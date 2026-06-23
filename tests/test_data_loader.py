import pytest
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader

from src.data_loader import TokenClassificationDataset, create_data_loader, create_data_loaders


# ── Mock tokenizer ─────────────────────────────────────────────────────────────

SEQ_LEN = 10

def _make_tokenizer(seq_len=SEQ_LEN):
    """Return a mock tokenizer whose __call__ returns padded encodings with word_ids."""
    tok = MagicMock()

    def call_side_effect(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        # Return a mock BatchEncoding
        enc = MagicMock()
        enc.__getitem__ = lambda self, k: (
            torch.zeros(n, seq_len, dtype=torch.long) if k == 'input_ids' else
            torch.ones(n, seq_len, dtype=torch.long)
        )
        enc.word_ids = MagicMock(return_value=[None] + list(range(n - 2)) + [None])
        return enc

    tok.side_effect = call_side_effect

    def batch_call(texts, **kwargs):
        if isinstance(texts, list):
            n = len(texts)
        else:
            n = 1
        enc = MagicMock()
        enc.__getitem__ = lambda self, k: (
            torch.zeros(n, seq_len, dtype=torch.long) if k == 'input_ids' else
            torch.ones(n, seq_len, dtype=torch.long)
        )
        return enc

    # Two behaviours: batch call (list) returns padded tensors;
    # single call (str) returns per-sentence encoding with word_ids
    def smart_call(texts, truncation=True, padding=None, max_length=128,
                   return_tensors=None, return_offsets_mapping=False,
                   add_special_tokens=True):
        if isinstance(texts, list) and return_tensors == 'pt':
            n = len(texts)
            enc = MagicMock()
            enc.__getitem__ = lambda self, k: (
                torch.zeros(n, seq_len, dtype=torch.long) if k == 'input_ids' else
                torch.ones(n, seq_len, dtype=torch.long)
            )
            return enc
        else:
            # Per-sentence call (no return_tensors)
            enc = MagicMock()
            enc.__getitem__ = lambda self, k: (
                list(range(seq_len)) if k == 'input_ids' else
                [1] * seq_len
            )
            enc.word_ids = MagicMock(return_value=[None] + [0, 1, 2] + [None] + [-1] * (seq_len - 5))
            return enc

    tok.side_effect = None
    tok.__call__ = MagicMock(side_effect=smart_call)

    def convert_ids_to_tokens(ids):
        return ['▁word'] * len(ids)

    tok.convert_ids_to_tokens = MagicMock(side_effect=convert_ids_to_tokens)
    return tok


def _simple_align(word_labels, tokens, token_strings):
    """Trivial alignment: head token gets label, rest -100."""
    n = seq_len = SEQ_LEN
    result = [-100] * n
    for i, label in enumerate(word_labels[:n]):
        result[i] = label
    return result


# ── TokenClassificationDataset ─────────────────────────────────────────────────

def test_dataset_length():
    tok = _make_tokenizer()
    sentences   = ["Hello world.", "Another sentence."]
    words_only  = [["Hello", "world"], ["Another", "sentence"]]
    word_labels = [[0, 1], [0, 0]]
    ds = TokenClassificationDataset(sentences, words_only, word_labels, tok,
                                    max_length=SEQ_LEN,
                                    label_alignment_func=_simple_align)
    assert len(ds) == 2

def test_dataset_getitem_keys():
    tok = _make_tokenizer()
    sentences   = ["Hello world."]
    words_only  = [["Hello", "world"]]
    word_labels = [[0, 1]]
    ds = TokenClassificationDataset(sentences, words_only, word_labels, tok,
                                    max_length=SEQ_LEN,
                                    label_alignment_func=_simple_align)
    item = ds[0]
    assert 'input_ids' in item
    assert 'attention_mask' in item
    assert 'labels' in item

def test_dataset_labels_are_tensors():
    tok = _make_tokenizer()
    sentences   = ["Hello world."]
    words_only  = [["Hello", "world"]]
    word_labels = [[0, 1]]
    ds = TokenClassificationDataset(sentences, words_only, word_labels, tok,
                                    max_length=SEQ_LEN,
                                    label_alignment_func=_simple_align)
    item = ds[0]
    assert isinstance(item['labels'], torch.Tensor)

def test_dataset_labels_dtype_long():
    tok = _make_tokenizer()
    sentences   = ["Hello world."]
    words_only  = [["Hello", "world"]]
    word_labels = [[0, 1]]
    ds = TokenClassificationDataset(sentences, words_only, word_labels, tok,
                                    max_length=SEQ_LEN,
                                    label_alignment_func=_simple_align)
    item = ds[0]
    assert item['labels'].dtype == torch.long

def test_dataset_without_alignment_func():
    tok = _make_tokenizer()
    sentences   = ["Hello world."]
    words_only  = [["Hello", "world"]]
    word_labels = [[0, 1, -100, -100, -100, -100, -100, -100, -100, -100]]
    ds = TokenClassificationDataset(sentences, words_only, word_labels, tok,
                                    max_length=SEQ_LEN,
                                    label_alignment_func=None)
    item = ds[0]
    assert 'labels' in item


# ── create_data_loader ─────────────────────────────────────────────────────────

def test_create_data_loader_returns_dataloader():
    tok = _make_tokenizer()
    loader = create_data_loader(
        sentences=["Hello.", "World."],
        words_only=[["Hello"], ["World"]],
        word_labels=[[0], [1]],
        tokenizer=tok,
        batch_size=2,
        max_length=SEQ_LEN,
        label_alignment_func=_simple_align,
    )
    assert isinstance(loader, DataLoader)

def test_create_data_loader_batch_size():
    tok = _make_tokenizer()
    loader = create_data_loader(
        sentences=["Hello.", "World.", "Foo."],
        words_only=[["Hello"], ["World"], ["Foo"]],
        word_labels=[[0], [1], [0]],
        tokenizer=tok,
        batch_size=2,
        max_length=SEQ_LEN,
        label_alignment_func=_simple_align,
    )
    assert loader.batch_size == 2


# ── create_data_loaders ───────────────────────────────────────────────────────

def test_create_data_loaders_returns_tuple():
    tok = _make_tokenizer()
    train_l, val_l = create_data_loaders(
        train_sentences=["Hello."],
        train_words_only=[["Hello"]],
        train_word_labels=[[0]],
        val_sentences=["World."],
        val_words_only=[["World"]],
        val_word_labels=[[1]],
        tokenizer=tok,
        batch_size=1,
        max_length=SEQ_LEN,
        label_alignment_func=_simple_align,
    )
    assert isinstance(train_l, DataLoader)
    assert isinstance(val_l, DataLoader)

def test_create_data_loaders_no_val_returns_none():
    tok = _make_tokenizer()
    train_l, val_l = create_data_loaders(
        train_sentences=["Hello."],
        train_words_only=[["Hello"]],
        train_word_labels=[[0]],
        tokenizer=tok,
        batch_size=1,
        max_length=SEQ_LEN,
        label_alignment_func=_simple_align,
    )
    assert isinstance(train_l, DataLoader)
    assert val_l is None
