import pytest
from src.label_alignment import align_labels_with_tokens_using_wordids


class MockEncoding:
    """Minimal mock of a HuggingFace BatchEncoding with word_ids() support."""
    def __init__(self, word_ids_list):
        self._word_ids = word_ids_list

    def word_ids(self):
        return self._word_ids


# ── Error cases ────────────────────────────────────────────────────────────────

def test_raises_without_word_ids_method():
    with pytest.raises(ValueError, match="word_ids"):
        align_labels_with_tokens_using_wordids([0, 1], object(), token_strings=['a', 'b'])

def test_raises_without_token_strings():
    enc = MockEncoding([None, 0, 0, None])
    with pytest.raises(ValueError, match="token_strings"):
        align_labels_with_tokens_using_wordids([0], enc, token_strings=None)


# ── Basic alignment ───────────────────────────────────────────────────────────

def test_all_special_tokens_ignored():
    # [CLS] [SEP] — both word_id=None
    enc = MockEncoding([None, None])
    token_strings = ['<s>', '</s>']
    result = align_labels_with_tokens_using_wordids([], enc, token_strings=token_strings)
    assert result == [-100, -100]

def test_single_word_single_token():
    # [CLS] word [SEP]
    enc = MockEncoding([None, 0, None])
    token_strings = ['<s>', '▁hello', '</s>']
    result = align_labels_with_tokens_using_wordids([1], enc, token_strings=token_strings)
    assert result == [-100, 1, -100]

def test_two_words_literal_and_metaphor():
    # [CLS] word1 word2 [SEP]
    enc = MockEncoding([None, 0, 1, None])
    token_strings = ['<s>', '▁They', '▁sailed', '</s>']
    result = align_labels_with_tokens_using_wordids([0, 1], enc, token_strings=token_strings)
    assert result == [-100, 0, 1, -100]

def test_subword_trailing_gets_minus100():
    # [CLS] word(split into 2 subwords) [SEP]
    enc = MockEncoding([None, 0, 0, None])
    token_strings = ['<s>', '▁strugg', 'led', '</s>']
    result = align_labels_with_tokens_using_wordids([1], enc, token_strings=token_strings)
    # First subword gets the label, trailing gets -100
    assert result == [-100, 1, -100, -100]

def test_punctuation_only_group_skipped():
    # [CLS] "." word [SEP] — punctuation group gets -100, doesn't consume label
    enc = MockEncoding([None, 0, 1, None])
    token_strings = ['<s>', '.', '▁hello', '</s>']
    result = align_labels_with_tokens_using_wordids([0], enc, token_strings=token_strings)
    # '.' is punctuation-only → -100, doesn't consume a label entry
    # '▁hello' is word_id=1, gets label[0]=0
    assert result == [-100, -100, 0, -100]

def test_leading_punctuation_subword_skipped():
    # A word group whose first subtoken is punctuation, second is alphanumeric
    # word_id=0 spans two tokens: '"' and '▁complex'
    enc = MockEncoding([None, 0, 0, None])
    token_strings = ['<s>', '"', '▁complex', '</s>']
    result = align_labels_with_tokens_using_wordids([1], enc, token_strings=token_strings)
    # '"' is punctuation-leading → -100; '▁complex' gets label
    assert result == [-100, -100, 1, -100]

def test_empty_word_labels():
    enc = MockEncoding([None, None])
    token_strings = ['<s>', '</s>']
    result = align_labels_with_tokens_using_wordids([], enc, token_strings=token_strings)
    assert result == [-100, -100]

def test_output_length_matches_input_tokens():
    enc = MockEncoding([None, 0, 1, 2, None])
    token_strings = ['<s>', '▁a', '▁b', '▁c', '</s>']
    result = align_labels_with_tokens_using_wordids([0, 0, 1], enc, token_strings=token_strings)
    assert len(result) == 5

def test_no_words_beyond_label_list_causes_no_error():
    # More word groups than labels — extra groups should just not get a label
    enc = MockEncoding([None, 0, 1, None])
    token_strings = ['<s>', '▁a', '▁b', '</s>']
    result = align_labels_with_tokens_using_wordids([0], enc, token_strings=token_strings)
    # word_id=0 → label[0]=0; word_id=1 → no label available → stays -100
    assert result[1] == 0
    assert result[2] == -100
