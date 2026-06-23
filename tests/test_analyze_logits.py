"""
Tests for src/analyze_logits.py.

analyze_logits() orchestrates: data loading → model inference → metric printing.
We mock extract_vuamc_data and patch the DataLoader to return synthetic batches,
avoiding I/O and large model downloads.
"""

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.analyze_logits import analyze_logits


# ── Helpers ────────────────────────────────────────────────────────────────────

SEQ_LEN = 8
BATCH_SIZE = 2


def _fake_batch(n_literal=4, n_metaphor=2):
    """Return a fake DataLoader batch with some literal and metaphor labels."""
    total = SEQ_LEN * BATCH_SIZE
    labels = torch.full((BATCH_SIZE, SEQ_LEN), -100, dtype=torch.long)
    # Fill first few positions with real labels
    for row in range(BATCH_SIZE):
        labels[row, 0] = 0  # literal
        labels[row, 1] = 1  # metaphor
    return {
        'input_ids':      torch.zeros(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
        'attention_mask': torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
        'labels':         labels,
    }


class TinyModel(nn.Module):
    """2-class token classifier for deterministic test outputs."""
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        # Always predict class 0 with high confidence
        logits = torch.zeros(batch_size, seq_len, 2)
        logits[:, :, 0] = 2.0
        logits[:, :, 1] = -2.0
        out = MagicMock()
        out.logits = logits
        return out


def _make_data(n=10):
    sentences   = [f"Sentence {i}." for i in range(n)]
    words_only  = [["Sentence", str(i)] for i in range(n)]
    word_labels = [[0, 0] for _ in range(n)]
    return sentences, words_only, word_labels


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch('src.analyze_logits.extract_vuamc_data')
@patch('src.analyze_logits.DataLoader')
def test_analyze_logits_runs_without_error(mock_dl_cls, mock_extract):
    mock_extract.return_value = _make_data(10)
    mock_dl_cls.return_value = [_fake_batch()]

    model = TinyModel()
    tok   = MagicMock()
    analyze_logits(model, tok, xml_file='dummy.xml',
                   max_length=SEQ_LEN, batch_size=BATCH_SIZE)


@patch('src.analyze_logits.extract_vuamc_data')
@patch('src.analyze_logits.DataLoader')
def test_analyze_logits_calls_extract_with_correct_file(mock_dl_cls, mock_extract):
    mock_extract.return_value = _make_data(10)
    mock_dl_cls.return_value = [_fake_batch()]

    model = TinyModel()
    tok   = MagicMock()
    analyze_logits(model, tok, xml_file='my_corpus.xml',
                   max_length=SEQ_LEN, batch_size=BATCH_SIZE)

    mock_extract.assert_called_once_with('my_corpus.xml')


@patch('src.analyze_logits.extract_vuamc_data')
@patch('src.analyze_logits.DataLoader')
def test_analyze_logits_model_in_eval_mode_after_call(mock_dl_cls, mock_extract):
    mock_extract.return_value = _make_data(10)
    mock_dl_cls.return_value = [_fake_batch()]

    model = TinyModel()
    tok   = MagicMock()
    analyze_logits(model, tok, xml_file='dummy.xml',
                   max_length=SEQ_LEN, batch_size=BATCH_SIZE)

    assert not model.training


@patch('src.analyze_logits.extract_vuamc_data')
@patch('src.analyze_logits.DataLoader')
def test_analyze_logits_with_mixed_labels(mock_dl_cls, mock_extract):
    """Ensure the function handles data containing a mix of 0 and 1 labels."""
    sentences, words_only, _ = _make_data(10)
    word_labels = [[0, 1] for _ in range(10)]
    mock_extract.return_value = (sentences, words_only, word_labels)
    mock_dl_cls.return_value = [_fake_batch()]

    model = TinyModel()
    tok   = MagicMock()
    analyze_logits(model, tok, xml_file='dummy.xml',
                   max_length=SEQ_LEN, batch_size=BATCH_SIZE)


@patch('src.analyze_logits.extract_vuamc_data')
@patch('src.analyze_logits.DataLoader')
def test_analyze_logits_uses_80_percent_val_split(mock_dl_cls, mock_extract):
    """Verify only 20% of the data is used as validation."""
    n = 20
    sentences, words_only, word_labels = _make_data(n)
    mock_extract.return_value = (sentences, words_only, word_labels)
    mock_dl_cls.return_value = [_fake_batch()]

    model = TinyModel()
    tok   = MagicMock()
    analyze_logits(model, tok, xml_file='dummy.xml',
                   max_length=SEQ_LEN, batch_size=BATCH_SIZE)

    # DataLoader should be called with a dataset of len = 20% of n = 4
    dataset_passed = mock_dl_cls.call_args[0][0]
    assert len(dataset_passed) == n - int(0.8 * n)
