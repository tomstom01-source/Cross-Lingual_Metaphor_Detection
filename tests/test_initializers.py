import pytest
from unittest.mock import MagicMock, patch

from src.initializers import (
    initialize_xlm_roberta_tokenizer,
    initialize_xlm_roberta_model_for_classification,
)


# Both functions call HuggingFace's from_pretrained, which downloads weights.
# We patch it to keep tests fast and offline.

@patch('src.initializers.AutoTokenizer')
def test_tokenizer_returns_tokenizer_object(MockAutoTok):
    mock_tok = MagicMock()
    MockAutoTok.from_pretrained.return_value = mock_tok
    result = initialize_xlm_roberta_tokenizer()
    assert result is mock_tok

@patch('src.initializers.AutoTokenizer')
def test_tokenizer_uses_xlm_roberta_base(MockAutoTok):
    initialize_xlm_roberta_tokenizer()
    MockAutoTok.from_pretrained.assert_called_once_with('xlm-roberta-base')

@patch('src.initializers.AutoModelForTokenClassification')
def test_model_returns_model_object(MockAutoModel):
    mock_model = MagicMock()
    MockAutoModel.from_pretrained.return_value = mock_model
    result = initialize_xlm_roberta_model_for_classification()
    assert result is mock_model

@patch('src.initializers.AutoModelForTokenClassification')
def test_model_uses_xlm_roberta_base(MockAutoModel):
    initialize_xlm_roberta_model_for_classification()
    MockAutoModel.from_pretrained.assert_called_once_with('xlm-roberta-base', num_labels=2)

@patch('src.initializers.AutoModelForTokenClassification')
def test_model_passes_num_labels(MockAutoModel):
    initialize_xlm_roberta_model_for_classification(num_labels=3)
    MockAutoModel.from_pretrained.assert_called_once_with('xlm-roberta-base', num_labels=3)

@patch('src.initializers.AutoModelForTokenClassification')
def test_model_default_num_labels_is_2(MockAutoModel):
    initialize_xlm_roberta_model_for_classification()
    _, kwargs = MockAutoModel.from_pretrained.call_args
    assert kwargs.get('num_labels', 2) == 2
