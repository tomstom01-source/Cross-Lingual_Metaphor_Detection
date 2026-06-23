"""
End-to-end integration tests for the full metaphor-detection pipeline.

Each test exercises the complete chain:
  extract_xml_subset → extract_vuamc_data
  → create_data_loaders (with real tokenizer + label alignment)
  → train_model
  → evaluate
  → print_report

The real XLM-RoBERTa tokenizer is used (loaded once per module via a fixture).
The real XLM-RoBERTa model is created fresh per test to avoid state leakage.
A small N_SENTENCES and EPOCHS=1 keeps each test fast (~5–15 s on CPU).
"""

import os
import pytest
import torch

from src.extractors import extract_xml_subset, extract_vuamc_data
from src.initializers import (
    initialize_xlm_roberta_tokenizer,
    initialize_xlm_roberta_model_for_classification,
)
from src.label_alignment import align_labels_with_tokens_using_wordids
from src.data_loader import create_data_loaders
from src.train_metaphor_detector import calculate_class_weights, train_model
from run import evaluate, set_seed, print_report

# ── Constants ─────────────────────────────────────────────────────────────────

VUAMC_XML   = 'data/VUAMC/VUAMC.xml'
DEVICE      = torch.device('cpu')
N_SENTENCES = 30   # Small slice — keeps tests under ~15 s each
MAX_LENGTH  = 64
EPOCHS      = 1
BATCH_SIZE  = 4

pytestmark = pytest.mark.skipif(
    not os.path.exists(VUAMC_XML),
    reason="VUAMC XML not present at data/VUAMC/VUAMC.xml"
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    """Load the real XLM-R tokenizer once for the whole module."""
    return initialize_xlm_roberta_tokenizer()


@pytest.fixture(scope="module")
def extracted_data(tmp_path_factory):
    """
    Extract N_SENTENCES from the real VUAMC corpus once for all tests.
    Returns (sentences, words_only, word_labels).
    """
    tmp = tmp_path_factory.mktemp("vuamc")
    subset_xml = str(tmp / "subset.xml")
    extract_xml_subset(VUAMC_XML, subset_xml, N_SENTENCES)
    return extract_vuamc_data(subset_xml)


# ── Pipeline helper ───────────────────────────────────────────────────────────

def run_pipeline(tokenizer, sentences, words_only, word_labels,
                 train_split=0.8, no_class_weights=False,
                 save_path=None, seed=42):
    """
    Run the full pipeline and return the metrics dict from evaluate().
    The model is freshly initialised on every call.
    """
    set_seed(seed)
    train_size = int(train_split * len(sentences))

    tr_s = sentences[:train_size];  tr_w = words_only[:train_size];  tr_l = word_labels[:train_size]
    va_s = sentences[train_size:];  va_w = words_only[train_size:];  va_l = word_labels[train_size:]

    model = initialize_xlm_roberta_model_for_classification(num_labels=2)
    class_weights = None if no_class_weights else calculate_class_weights(tr_l)

    train_loader, val_loader = create_data_loaders(
        train_sentences=tr_s, train_words_only=tr_w, train_word_labels=tr_l,
        val_sentences=va_s,   val_words_only=va_w,   val_word_labels=va_l,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        label_alignment_func=align_labels_with_tokens_using_wordids,
    )

    train_model(model, train_loader, val_loader, DEVICE,
                epochs=EPOCHS, save_path=save_path)

    if save_path and os.path.exists(save_path):
        model.load_state_dict(
            torch.load(save_path, map_location=DEVICE, weights_only=True)
        )

    return evaluate(model, val_loader, DEVICE)


# ── Data extraction tests ─────────────────────────────────────────────────────

def test_extraction_sentence_count(extracted_data):
    sentences, words_only, word_labels = extracted_data
    assert len(sentences) == N_SENTENCES

def test_extraction_alignment(extracted_data):
    """Every word list must be the same length as its label list."""
    _, words_only, word_labels = extracted_data
    for words, labels in zip(words_only, word_labels):
        assert len(words) == len(labels)

def test_extraction_labels_binary(extracted_data):
    """All labels must be 0 or 1."""
    _, _, word_labels = extracted_data
    for labels in word_labels:
        assert all(l in (0, 1) for l in labels)

def test_extraction_contains_metaphors(extracted_data):
    """VUAMC should contain at least some metaphor labels in 30 sentences."""
    _, _, word_labels = extracted_data
    total_metaphors = sum(sum(ls) for ls in word_labels)
    assert total_metaphors > 0


# ── Pipeline metrics structure ────────────────────────────────────────────────

def test_metrics_keys_present(tokenizer, extracted_data):
    sentences, words_only, word_labels = extracted_data
    metrics = run_pipeline(tokenizer, sentences, words_only, word_labels)
    expected_keys = {'tp', 'tn', 'fp', 'fn', 'n_total', 'n_literal',
                     'n_metaphor', 'literal', 'metaphor', 'macro', 'weighted', 'accuracy'}
    assert expected_keys.issubset(metrics.keys())

def test_metrics_confusion_matrix_sums_to_n_total(tokenizer, extracted_data):
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels)
    assert m['tp'] + m['tn'] + m['fp'] + m['fn'] == m['n_total']

def test_metrics_accuracy_in_range(tokenizer, extracted_data):
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels)
    assert 0.0 <= m['accuracy'] <= 1.0

def test_metrics_per_class_tuples(tokenizer, extracted_data):
    """literal and metaphor values are (precision, recall, f1) tuples."""
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels)
    for key in ('literal', 'metaphor', 'macro', 'weighted'):
        assert len(m[key]) == 3
        assert all(0.0 <= v <= 1.0 for v in m[key])

def test_metrics_support_counts_correct(tokenizer, extracted_data):
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels)
    assert m['n_literal']  == m['tn'] + m['fp']
    assert m['n_metaphor'] == m['tp'] + m['fn']


# ── Edge case: no class weights ───────────────────────────────────────────────

def test_pipeline_no_class_weights_runs(tokenizer, extracted_data):
    """Pipeline should complete and return valid metrics when class weights are disabled."""
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels,
                     no_class_weights=True)
    assert 0.0 <= m['accuracy'] <= 1.0

def test_pipeline_no_class_weights_same_structure(tokenizer, extracted_data):
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels,
                     no_class_weights=True)
    assert m['tp'] + m['tn'] + m['fp'] + m['fn'] == m['n_total']


# ── Edge case: model checkpoint saving ───────────────────────────────────────

def test_pipeline_saves_checkpoint(tokenizer, extracted_data, tmp_path):
    sentences, words_only, word_labels = extracted_data
    save_path = str(tmp_path / "model.bin")
    run_pipeline(tokenizer, sentences, words_only, word_labels,
                 save_path=save_path)
    assert os.path.exists(save_path)

def test_pipeline_no_save_produces_no_file(tokenizer, extracted_data, tmp_path):
    sentences, words_only, word_labels = extracted_data
    run_pipeline(tokenizer, sentences, words_only, word_labels,
                 save_path=None)
    assert list(tmp_path.iterdir()) == []


# ── Edge case: different train/val splits ─────────────────────────────────────

def test_pipeline_split_70_30(tokenizer, extracted_data):
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels,
                     train_split=0.7)
    assert 0.0 <= m['accuracy'] <= 1.0

def test_pipeline_split_90_10(tokenizer, extracted_data):
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels,
                     train_split=0.9)
    assert 0.0 <= m['accuracy'] <= 1.0


# ── Edge case: reproducibility ────────────────────────────────────────────────

def test_pipeline_reproducible_with_same_seed(tokenizer, extracted_data):
    """Two runs with the same seed and no GPU should produce identical metrics."""
    sentences, words_only, word_labels = extracted_data
    m1 = run_pipeline(tokenizer, sentences, words_only, word_labels, seed=7)
    m2 = run_pipeline(tokenizer, sentences, words_only, word_labels, seed=7)
    assert m1['tp'] == m2['tp']
    assert m1['tn'] == m2['tn']
    assert m1['accuracy'] == pytest.approx(m2['accuracy'])


# ── print_report smoke test ───────────────────────────────────────────────────

def test_print_report_runs_without_error(tokenizer, extracted_data, capsys):
    sentences, words_only, word_labels = extracted_data
    m = run_pipeline(tokenizer, sentences, words_only, word_labels)
    print_report(m)  # Should not raise
    out = capsys.readouterr().out
    assert 'CONFUSION MATRIX' in out
    assert 'CLASSIFICATION METRICS' in out
    assert 'Accuracy' in out


# ── Edge case: subset size smaller than split boundary ────────────────────────

def test_pipeline_small_subset(tokenizer, tmp_path):
    """Pipeline should work with just 10 sentences (tiny dataset)."""
    subset_xml = str(tmp_path / "small_subset.xml")
    extract_xml_subset(VUAMC_XML, subset_xml, number_of_sentences=10)
    sentences, words_only, word_labels = extract_vuamc_data(subset_xml)

    m = run_pipeline(tokenizer, sentences, words_only, word_labels,
                     train_split=0.8)
    assert m['n_total'] >= 0
