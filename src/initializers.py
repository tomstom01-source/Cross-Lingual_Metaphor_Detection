from transformers import AutoTokenizer, AutoModelForTokenClassification

def initialize_xlm_roberta_tokenizer():
    """
    Initialize AutoTokenizer for xlm-roberta-base model.
    
    Returns:
        AutoTokenizer: Initialized tokenizer for xlm-roberta-base
    """
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer


def initialize_xlm_roberta_model_for_classification(num_labels=2):
    """
    Initialize AutoModelForTokenClassification for xlm-roberta-base model.
    
    Args:
        num_labels (int): Number of classification labels (default: 2 for binary classification)
    
    Returns:
        AutoModelForTokenClassification: Initialized model for token classification
    """
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
    return model

if __name__ == "__main__":
    # Test the tokenizer initialization
    tokenizer = initialize_xlm_roberta_tokenizer()
    print(f"Tokenizer initialized successfully: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")
    
    # Test tokenization
    test_text = "Hello, world!"
    tokens = tokenizer.encode(test_text)
    print(f"Test text: '{test_text}'")
    print(f"Token IDs: {tokens}")
    print(f"Decoded: '{tokenizer.decode(tokens)}'")
    
    print("\n" + "="*60)
    
    # Test the model initialization
    model = initialize_xlm_roberta_model_for_classification(num_labels=2)
    print(f"Model initialized successfully: {model.__class__.__name__}")
    print(f"Number of labels: {model.num_labels}")
    print(f"Model configuration: {model.config}")
    print(f"Total parameters: {model.num_parameters()}")
