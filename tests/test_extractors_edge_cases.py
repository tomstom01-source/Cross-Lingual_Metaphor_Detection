"""
Edge-case tests for src/extractors.py.

A separate XML fixture file is built for each group so the minimal corpus
in test_extractors.py stays clean.  All XML is VUAMC/TEI-format so the
extractor's namespace logic and element routing are exercised exactly as
they are on the real corpus.
"""

import os
import xml.etree.ElementTree as etree
import pytest

from src.extractors import (
    clean_word_text,
    get_word_text,
    is_classifiable_word,
    extract_xml_subset,
    extract_vuamc_data,
)

NS     = 'http://www.tei-c.org/ns/1.0'
NS_MAP = {'tei': NS}


# ── XML template helpers ───────────────────────────────────────────────────────

def _wrap_sentences(raw_sentences_xml: str) -> str:
    """Wrap raw <s>…</s> blocks in the minimal TEI envelope."""
    return f"""\
<?xml version='1.0' encoding='utf-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader/>
  <text>
    <group>
      <text xml:id="t01">
        <body>
          <div1 n="test" type="u">
            <p>
{raw_sentences_xml}
            </p>
          </div1>
        </body>
      </text>
    </group>
  </text>
</TEI>
"""


def _write_xml(tmp_path, name: str, raw_sentences_xml: str) -> str:
    path = tmp_path / name
    path.write_text(_wrap_sentences(raw_sentences_xml), encoding='utf-8')
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════════
# clean_word_text — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_clean_word_text_only_newlines():
    assert clean_word_text('\n\n\n') == ''

def test_clean_word_text_tab_before_text():
    assert clean_word_text('\thello') == 'hello'

def test_clean_word_text_mixed_whitespace():
    assert clean_word_text('  \t hello \t ') == 'hello'

def test_clean_word_text_newline_in_middle():
    # Only the text before the first newline is kept
    assert clean_word_text('first\nsecond\nthird') == 'first'

def test_clean_word_text_unicode_word():
    assert clean_word_text('  Straße  ') == 'Straße'

def test_clean_word_text_numbers_and_symbols():
    assert clean_word_text('  42%  ') == '42%'


# ═══════════════════════════════════════════════════════════════════════════════
# is_classifiable_word — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_is_classifiable_hyphenated():
    assert is_classifiable_word('well-known') is True

def test_is_classifiable_unicode_letters():
    assert is_classifiable_word('naïve') is True

def test_is_classifiable_digits_only():
    assert is_classifiable_word('007') is True

def test_is_classifiable_symbols_only():
    # &, @, # — no alphanumeric chars
    assert is_classifiable_word('&@#') is False

def test_is_classifiable_single_char_alpha():
    assert is_classifiable_word('I') is True

def test_is_classifiable_single_char_punct():
    assert is_classifiable_word(',') is False

def test_is_classifiable_whitespace_only():
    assert is_classifiable_word('   ') is False

def test_is_classifiable_ellipsis():
    assert is_classifiable_word('...') is False

def test_is_classifiable_mixed_symbols_and_alpha():
    assert is_classifiable_word('$100') is True


# ═══════════════════════════════════════════════════════════════════════════════
# get_word_text — edge cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_get_word_text_trailing_space_in_seg():
    # seg.text has trailing whitespace — should be stripped
    elem = etree.fromstring('<w><seg function="mrw" type="met">sailed  </seg></w>')
    seg = elem.find('seg')
    assert get_word_text(elem, seg) == 'sailed'

def test_get_word_text_unicode_in_seg():
    elem = etree.fromstring('<w><seg function="mrw" type="met">naïve</seg></w>')
    seg = elem.find('seg')
    assert get_word_text(elem, seg) == 'naïve'

def test_get_word_text_numbers_in_word():
    elem = etree.fromstring('<w>42nd</w>')
    assert get_word_text(elem) == '42nd'

def test_get_word_text_hyphenated():
    elem = etree.fromstring('<w>well-known</w>')
    assert get_word_text(elem) == 'well-known'

def test_get_word_text_empty_seg():
    elem = etree.fromstring('<w><seg function="mrw" type="met"></seg></w>')
    seg = elem.find('seg')
    assert get_word_text(elem, seg) == ''

def test_get_word_text_only_whitespace_seg():
    elem = etree.fromstring('<w><seg function="mrw" type="met">   </seg></w>')
    seg = elem.find('seg')
    # rstrip keeps leading spaces; clean_word_text handles the outer text, not seg
    # The result should have the whitespace stripped only on the right
    result = get_word_text(elem, seg)
    assert result.strip() == ''


# ═══════════════════════════════════════════════════════════════════════════════
# extract_vuamc_data — edge-case XML fixtures
# ═══════════════════════════════════════════════════════════════════════════════

# ── Whitespace-only placeholder <w> elements (BNC artefacts) ─────────────────

def test_whitespace_placeholder_words_skipped(tmp_path):
    """Whitespace-only <w> elements must not appear in words_only."""
    xml = _write_xml(tmp_path, "ws.xml", """\
              <s>
                <w>Hello</w>
                <w>   </w>
                <w>world</w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    assert words_only[0] == ['Hello', 'world']
    assert word_labels[0] == [0, 0]


# ── Contraction / apostrophe merging ─────────────────────────────────────────

def test_contraction_nt_merged(tmp_path):
    """BNC stores 'do' + "n't" as two <w> elements; words_only merges them."""
    xml = _write_xml(tmp_path, "nt.xml", """\
              <s>
                <w>I</w>
                <w>do</w>
                <w>n't</w>
                <w>know</w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    # "n't" merges onto "do" -> "don't"; "do" disappears as standalone
    words = words_only[0]
    assert "don't" in words
    assert 'n\'t' not in words

def test_possessive_apostrophe_s_merged(tmp_path):
    """'s possessive suffix merges back onto the preceding word."""
    xml = _write_xml(tmp_path, "pos.xml", """\
              <s>
                <w>John</w>
                <w>'s</w>
                <w>car</w>
                <c>.</c>
              </s>""")
    _, words_only, _ = extract_vuamc_data(xml)
    words = words_only[0]
    # "John's" should appear; "'s" alone should not
    assert "John's" in words
    assert "'s" not in words

def test_possessive_pos_type_merged(tmp_path):
    """<w type="POS"> possessive markers are merged onto the preceding word."""
    xml = _write_xml(tmp_path, "postype.xml", """\
              <s>
                <w>Mary</w>
                <w type="POS">'s</w>
                <w>book</w>
                <c>.</c>
              </s>""")
    _, words_only, _ = extract_vuamc_data(xml)
    words = words_only[0]
    assert "Mary's" in words


# ── Punctuation-only sentences ────────────────────────────────────────────────

def test_sentence_with_only_punctuation_has_empty_words(tmp_path):
    """A <s> with only <c> elements appears in sentences but with empty words/labels."""
    xml = _write_xml(tmp_path, "punct_only.xml", """\
              <s>
                <c>...</c>
                <c>!</c>
              </s>
              <s>
                <w>Real</w>
                <w>sentence</w>
                <c>.</c>
              </s>""")
    sentences, words_only, word_labels = extract_vuamc_data(xml)
    # The punctuation-only sentence is still emitted (sentence_parts is non-empty)
    # but contributes no words or labels.
    assert len(sentences) == 2
    assert words_only[0] == []
    assert word_labels[0] == []
    # The second sentence has the real content
    assert words_only[1] == ['Real', 'sentence']


# ── Mixed non-alphanumeric characters within words ────────────────────────────

def test_hyphenated_word_treated_as_single(tmp_path):
    """Hyphenated words count as a single classifiable word."""
    xml = _write_xml(tmp_path, "hyphen.xml", """\
              <s>
                <w>well-known</w>
                <w>fact</w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    assert 'well-known' in words_only[0]
    assert word_labels[0][words_only[0].index('well-known')] == 0

def test_word_with_ampersand(tmp_path):
    """A standalone & element is punctuation and should not enter words_only."""
    xml = _write_xml(tmp_path, "amp.xml", """\
              <s>
                <w>bread</w>
                <c>&amp;</c>
                <w>butter</w>
                <c>.</c>
              </s>""")
    _, words_only, _ = extract_vuamc_data(xml)
    assert words_only[0] == ['bread', 'butter']

def test_numeric_word_included(tmp_path):
    """Words made of digits are classifiable and must appear in words_only."""
    xml = _write_xml(tmp_path, "num.xml", """\
              <s>
                <w>There</w>
                <w>were</w>
                <w>42</w>
                <w>soldiers</w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    assert '42' in words_only[0]
    assert len(words_only[0]) == len(word_labels[0])

def test_percent_sign_word(tmp_path):
    """A word like '50%' contains an alphanumeric character and is classifiable."""
    xml = _write_xml(tmp_path, "pct.xml", """\
              <s>
                <w>50%</w>
                <w>off</w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    assert '50%' in words_only[0]


# ── Unicode / special characters in word text ─────────────────────────────────

def test_unicode_word_in_sentence(tmp_path):
    """Non-ASCII words (e.g. German umlauts) are extracted correctly."""
    xml = _write_xml(tmp_path, "unicode.xml", """\
              <s>
                <w>Die</w>
                <w>Stra&#xDF;e</w>
                <w>ist</w>
                <w>sch&#xF6;n</w>
                <c>.</c>
              </s>""")
    _, words_only, _ = extract_vuamc_data(xml)
    assert 'Straße' in words_only[0]
    assert 'schön' in words_only[0]

def test_unicode_metaphor_word(tmp_path):
    """A metaphor label on a non-ASCII word is correctly assigned."""
    xml = _write_xml(tmp_path, "unicode_met.xml", """\
              <s>
                <w>Life</w>
                <w>is</w>
                <w><seg function="mrw" type="met">&#xE9;lan</seg></w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    idx = words_only[0].index('élan')
    assert word_labels[0][idx] == 1


# ── Multiple metaphors in one sentence ────────────────────────────────────────

def test_multiple_metaphors_same_sentence(tmp_path):
    """Two metaphor words in a single sentence both receive label 1."""
    xml = _write_xml(tmp_path, "multi_met.xml", """\
              <s>
                <w>He</w>
                <w><seg function="mrw" type="met">sailed</seg></w>
                <w>through</w>
                <w><seg function="mrw" type="met">darkness</seg></w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    assert word_labels[0][words_only[0].index('sailed')]   == 1
    assert word_labels[0][words_only[0].index('darkness')] == 1
    assert word_labels[0][words_only[0].index('through')]  == 0


# ── Opening bracket triggers space before it ──────────────────────────────────

def test_space_before_opening_bracket(tmp_path):
    """A word before an opening bracket '(' gets a trailing space."""
    xml = _write_xml(tmp_path, "bracket.xml", """\
              <s>
                <w>example</w>
                <c>(see</c>
                <w>below</w>
                <c>)</c>
                <c>.</c>
              </s>""")
    sentences, _, _ = extract_vuamc_data(xml)
    assert 'example (' in sentences[0]


# ── Labels aligned with words: invariant ────────────────────────────────────

def test_labels_always_align_complex_sentence(tmp_path):
    """For every extracted sentence, len(words) == len(labels)."""
    xml = _write_xml(tmp_path, "align.xml", """\
              <s>
                <w>The</w>
                <w><seg function="mrw" type="met">river</seg></w>
                <w>of</w>
                <w>   </w>
                <w>time</w>
                <w>flows</w>
                <w>n't</w>
                <c>,</c>
                <w>said</w>
                <w>no</w>
                <w>one</w>
                <c>.</c>
              </s>
              <s>
                <w>But</w>
                <w><seg function="mrw" type="met">shadows</seg></w>
                <w>do</w>
                <c>.</c>
              </s>""")
    _, words_only, word_labels = extract_vuamc_data(xml)
    for words, labels in zip(words_only, word_labels):
        assert len(words) == len(labels), \
            f"Mismatch: words={words}, labels={labels}"


# ── Sentence with nothing but whitespace-only <w> elements ────────────────────

def test_all_whitespace_words_sentence_excluded(tmp_path):
    """A <s> whose only <w> children are whitespace placeholders is dropped."""
    xml = _write_xml(tmp_path, "all_ws.xml", """\
              <s>
                <w>  </w>
                <w>  </w>
              </s>
              <s>
                <w>Good</w>
                <w>sentence</w>
                <c>.</c>
              </s>""")
    sentences, words_only, _ = extract_vuamc_data(xml)
    assert len(sentences) == 1
    assert words_only[0] == ['Good', 'sentence']


# ── extract_xml_subset edge cases ────────────────────────────────────────────

def test_extract_subset_zero_sentences(tmp_path):
    """Requesting 0 sentences writes an empty (but valid) XML file."""
    src = _write_xml(tmp_path, "src.xml", """\
              <s><w>Hello</w><c>.</c></s>""")
    out = str(tmp_path / "out.xml")
    extract_xml_subset(src, out, number_of_sentences=0)
    tree = etree.parse(out)
    sentences = tree.getroot().findall('.//tei:s', NS_MAP)
    assert len(sentences) == 0

def test_extract_subset_preserves_metaphor_annotation(tmp_path):
    """The extracted subset keeps <seg function="mrw" type="met"> intact."""
    src = _write_xml(tmp_path, "met.xml", """\
              <s>
                <w><seg function="mrw" type="met">sailed</seg></w>
                <c>.</c>
              </s>""")
    out = str(tmp_path / "out.xml")
    extract_xml_subset(src, out, number_of_sentences=1)
    _, words_only, word_labels = extract_vuamc_data(out)
    assert word_labels[0][0] == 1

def test_extract_subset_output_is_valid_xml(tmp_path):
    """The written subset can be parsed without errors by ElementTree."""
    src = _write_xml(tmp_path, "src.xml", """\
              <s><w>Test</w><c>.</c></s>""")
    out = str(tmp_path / "out.xml")
    extract_xml_subset(src, out, number_of_sentences=1)
    # Would raise if invalid XML
    tree = etree.parse(out)
    assert tree.getroot() is not None
