import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from src.train_metaphor_detector import calculate_class_weights, train_model


# ── calculate_class_weights ───────────────────────────────────────────────────

def test_class_weights_returns_tensor():
    labels = [[0, 1, 0], [0, 0, 1]]
    weights = calculate_class_weights(labels)
    assert isinstance(weights, torch.Tensor)

def test_class_weights_shape():
    labels = [[0, 1, 0], [0, 0, 1]]
    weights = calculate_class_weights(labels)
    assert weights.shape == (2,)

def test_class_weights_literal_is_baseline():
    labels = [[0, 1, 0], [0, 0, 1]]
    weights = calculate_class_weights(labels)
    assert weights[0].item() == pytest.approx(1.0)

def test_class_weights_metaphor_proportional():
    # 4 literal, 2 metaphor → weight_1 = 4/2 = 2.0
    labels = [[0, 0, 1], [0, 1, 0]]
    weights = calculate_class_weights(labels)
    assert weights[1].item() == pytest.approx(2.0)

def test_class_weights_no_metaphors_fallback():
    # Zero metaphors → weight_1 defaults to 1.0
    labels = [[0, 0, 0]]
    weights = calculate_class_weights(labels)
    assert weights[1].item() == pytest.approx(1.0)

def test_class_weights_all_metaphors():
    # Only metaphors → weight_1 = 0/3 = 0.0
    labels = [[1, 1, 1]]
    weights = calculate_class_weights(labels)
    assert weights[1].item() == pytest.approx(0.0)

def test_class_weights_float_dtype():
    labels = [[0, 1]]
    weights = calculate_class_weights(labels)
    assert weights.dtype == torch.float32


# ── train_model ───────────────────────────────────────────────────────────────

def _make_batch(seq_len=8, batch_size=2):
    """Create a fake batch with random input_ids, masks, and binary labels."""
    input_ids      = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels         = torch.randint(-1, 2, (batch_size, seq_len))
    labels[labels == -1] = -100  # simulate ignore index
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def _make_loader(n_batches=2, seq_len=8):
    batch = _make_batch(seq_len=seq_len)
    return [batch] * n_batches


class TinyModel(nn.Module):
    """Tiny 2-class token classifier for testing train_model without downloading XLM-R."""
    def __init__(self, vocab_size=100, hidden=16, num_labels=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.fc    = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None):
        logits = self.fc(self.embed(input_ids))
        out = MagicMock()
        out.logits = logits
        return out


def test_train_model_returns_model():
    model = TinyModel()
    device = torch.device('cpu')
    loader = _make_loader(n_batches=2)
    result = train_model(model, loader, loader, device, epochs=1, save_path=None)
    assert result is model

def test_train_model_saves_checkpoint(tmp_path):
    model = TinyModel()
    device = torch.device('cpu')
    loader = _make_loader(n_batches=2)
    save_path = str(tmp_path / "model.bin")
    train_model(model, loader, loader, device, epochs=1, save_path=save_path)
    import os
    assert os.path.exists(save_path)

def test_train_model_no_save_when_path_is_none(tmp_path):
    model = TinyModel()
    device = torch.device('cpu')
    loader = _make_loader(n_batches=2)
    train_model(model, loader, loader, device, epochs=1, save_path=None)
    # Nothing should be written to tmp_path
    assert list(tmp_path.iterdir()) == []

def test_train_model_with_class_weights():
    model = TinyModel()
    device = torch.device('cpu')
    loader = _make_loader(n_batches=2)
    weights = torch.tensor([1.0, 3.0])
    result = train_model(model, loader, loader, device, epochs=1,
                         class_weights=weights, save_path=None)
    assert result is model

def test_train_model_without_val_loader():
    model = TinyModel()
    device = torch.device('cpu')
    loader = _make_loader(n_batches=2)
    result = train_model(model, loader, None, device, epochs=1, save_path=None)
    assert result is model
