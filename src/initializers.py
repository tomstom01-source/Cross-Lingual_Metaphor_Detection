from transformers import AutoTokenizer, AutoModelForTokenClassification

def initialize_xlm_roberta_tokenizer():
    """
    Initialize an AutoTokenizer for xlm-roberta-base.

    Returns:
        AutoTokenizer: Initialized tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer


def initialize_xlm_roberta_model_for_classification(num_labels=2):
    """
    Initialize an AutoModelForTokenClassification for xlm-roberta-base.

    Args:
        num_labels (int): Number of classification labels (default: 2 for binary classification)

    Returns:
        AutoModelForTokenClassification: Initialized model for token classification
    """
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
    return model
