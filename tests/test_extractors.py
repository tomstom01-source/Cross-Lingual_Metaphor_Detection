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

# ── Minimal VUAMC-format XML for testing ──────────────────────────────────────
NS = 'http://www.tei-c.org/ns/1.0'

MINIMAL_XML = """\
<?xml version='1.0' encoding='utf-8'?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader/>
  <text>
    <group>
      <text xml:id="t01">
        <body>
          <div1 n="test" type="u">
            <p>
              <s>
                <w>They</w>
                <w>went</w>
                <c>.</c>
              </s>
              <s>
                <w>He<seg function="mrw" type="met">sailed</seg></w>
                <w>home</w>
                <c>.</c>
              </s>
              <s>
                <w>Simple</w>
                <w>sentence</w>
              </s>
            </p>
          </div1>
        </body>
      </text>
    </group>
  </text>
</TEI>
"""


@pytest.fixture
def vuamc_xml_file(tmp_path):
    """Write the minimal XML to a temp file and return the path."""
    path = tmp_path / "test_vuamc.xml"
    path.write_text(MINIMAL_XML, encoding='utf-8')
    return str(path)


# ── clean_word_text ────────────────────────────────────────────────────────────

def test_clean_word_text_none():
    assert clean_word_text(None) == ''

def test_clean_word_text_empty():
    assert clean_word_text('') == ''

def test_clean_word_text_strips_whitespace():
    assert clean_word_text('  hello  ') == 'hello'

def test_clean_word_text_strips_newlines():
    assert clean_word_text('hello\nworld') == 'hello'

def test_clean_word_text_plain():
    assert clean_word_text('word') == 'word'


# ── is_classifiable_word ───────────────────────────────────────────────────────

def test_is_classifiable_word_alpha():
    assert is_classifiable_word('hello') is True

def test_is_classifiable_word_numeric():
    assert is_classifiable_word('42') is True

def test_is_classifiable_word_punctuation_only():
    assert is_classifiable_word('.') is False

def test_is_classifiable_word_mixed():
    assert is_classifiable_word("don't") is True

def test_is_classifiable_word_empty():
    assert is_classifiable_word('') is False


# ── get_word_text ─────────────────────────────────────────────────────────────

def test_get_word_text_no_seg():
    elem = etree.fromstring('<w>hello</w>')
    assert get_word_text(elem) == 'hello'

def test_get_word_text_with_seg():
    elem = etree.fromstring('<w><seg function="mrw" type="met">sailed</seg></w>')
    seg = elem.find('seg')
    assert get_word_text(elem, seg) == 'sailed'

def test_get_word_text_prefix_and_seg():
    elem = etree.fromstring('<w>re<seg function="mrw" type="met">sailed</seg></w>')
    seg = elem.find('seg')
    assert get_word_text(elem, seg) == 'resailed'

def test_get_word_text_empty_elem():
    elem = etree.fromstring('<w/>')
    assert get_word_text(elem) == ''


# ── extract_xml_subset ────────────────────────────────────────────────────────

def test_extract_xml_subset_creates_file(vuamc_xml_file, tmp_path):
    out = str(tmp_path / "subset.xml")
    extract_xml_subset(vuamc_xml_file, out, number_of_sentences=2)
    assert os.path.exists(out)

def test_extract_xml_subset_sentence_count(vuamc_xml_file, tmp_path):
    out = str(tmp_path / "subset.xml")
    extract_xml_subset(vuamc_xml_file, out, number_of_sentences=2)
    tree = etree.parse(out)
    root = tree.getroot()
    ns = {'tei': NS}
    sentences = root.findall('.//tei:s', ns)
    assert len(sentences) == 2

def test_extract_xml_subset_fewer_than_requested(vuamc_xml_file, tmp_path):
    out = str(tmp_path / "subset.xml")
    extract_xml_subset(vuamc_xml_file, out, number_of_sentences=100)
    tree = etree.parse(out)
    root = tree.getroot()
    ns = {'tei': NS}
    sentences = root.findall('.//tei:s', ns)
    assert len(sentences) == 3  # only 3 in MINIMAL_XML


# ── extract_vuamc_data ────────────────────────────────────────────────────────

def test_extract_vuamc_data_returns_three_lists(vuamc_xml_file):
    sentences, words_only, word_labels = extract_vuamc_data(vuamc_xml_file)
    assert isinstance(sentences, list)
    assert isinstance(words_only, list)
    assert isinstance(word_labels, list)

def test_extract_vuamc_data_sentence_count(vuamc_xml_file):
    sentences, words_only, word_labels = extract_vuamc_data(vuamc_xml_file)
    assert len(sentences) == len(words_only) == len(word_labels) == 3

def test_extract_vuamc_data_literal_sentence(vuamc_xml_file):
    _, words_only, word_labels = extract_vuamc_data(vuamc_xml_file)
    # First sentence: "They went." — both literal
    assert words_only[0] == ['They', 'went']
    assert word_labels[0] == [0, 0]

def test_extract_vuamc_data_metaphor_sentence(vuamc_xml_file):
    _, words_only, word_labels = extract_vuamc_data(vuamc_xml_file)
    # Second sentence: "sailed home." — sailed is metaphor
    assert word_labels[1][0] == 1   # sailed is metaphor
    assert word_labels[1][1] == 0   # home is literal

def test_extract_vuamc_data_labels_align_with_words(vuamc_xml_file):
    _, words_only, word_labels = extract_vuamc_data(vuamc_xml_file)
    for words, labels in zip(words_only, word_labels):
        assert len(words) == len(labels)

def test_extract_vuamc_data_sentence_strings(vuamc_xml_file):
    sentences, _, _ = extract_vuamc_data(vuamc_xml_file)
    for s in sentences:
        assert isinstance(s, str)
        assert len(s) > 0
