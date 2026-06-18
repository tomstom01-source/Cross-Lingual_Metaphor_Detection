def align_labels_with_tokens(word_labels, tokens):
    """
    Align word-level labels with subword tokens.
    
    When SentencePiece splits a word (e.g., "struggled" -> "_strugg", "led"),
    assign the true label to the first subword token and -100 (PyTorch ignore index)
    to the trailing subword tokens.
    
    Args:
        word_labels (list): List of labels corresponding to each word
        tokens (list): List of tokenized subword tokens from the tokenizer
    
    Returns:
        list: Aligned labels at the token level, with -100 for trailing subwords
    """
    aligned_labels = []
    word_idx = 0
    
    for token in tokens:
        # Check if this token is a continuation of a previous word
        # In XLM-RoBERTa (SentencePiece), the first subword of a word starts with '▁' (underscore)
        # while continuation subwords do not
        if token.startswith('▁'):
            # First subword of a new word
            if word_idx < len(word_labels):
                aligned_labels.append(word_labels[word_idx])
                word_idx += 1
            else:
                # Handle case where we run out of word labels
                aligned_labels.append(-100)
        else:
            # Continuation subword or special token - assign -100
            aligned_labels.append(-100)
    
    return aligned_labels

# default to be used
def align_labels_with_tokens_using_wordids(word_labels, tokens):
    """
    Advanced label alignment using word_ids() method if available.
    
    This function aligns word-level labels with token-level labels:
    - Word head tokens (first subword of each word) get 1s and 0s from word_labels
    - Trailing subwords get -100
    - Punctuation tokens get -100
    - Special tokens (CLS, SEP, etc.) get -100
    
    Args:
        word_labels (list): List of labels corresponding to each word (from words_only)
        tokens: Tokenized output from tokenizer (with word_ids() method) from full sentence
    
    Returns:
        list: Aligned labels at the token level
    """
    if not hasattr(tokens, 'word_ids'):
        raise ValueError("Tokenizer output does not have word_ids() method")
    
    word_ids = tokens.word_ids()
    aligned_labels = []
    previous_word_id = None
    
    for word_id in word_ids:
        if word_id is None:
            # Special tokens (CLS, SEP, etc.) get -100
            aligned_labels.append(-100)
        elif word_id < len(word_labels) and word_id != previous_word_id:
            # First subword token of a word gets the true label
            aligned_labels.append(word_labels[word_id])
            previous_word_id = word_id
        else:
            # Continuation subword tokens or punctuation get -100
            aligned_labels.append(-100)
    
    return aligned_labels


def align_labels_with_tokens_simple(word_texts, word_labels, tokenizer):
    """
    Simple label alignment by tokenizing each word individually.
    
    Args:
        word_texts (list): List of original word strings
        word_labels (list): List of labels corresponding to each word
        tokenizer: The tokenizer object
    
    Returns:
        tuple: (all_tokens, aligned_labels) where all_tokens is the flattened
               list of subword tokens and aligned_labels is the corresponding labels
    """
    all_tokens = []
    aligned_labels = []
    
    for word_text, label in zip(word_texts, word_labels):
        # Tokenize individual word
        word_tokens = tokenizer.tokenize(word_text)
        
        if not word_tokens:
            continue
        
        # First subword gets the true label
        all_tokens.append(word_tokens[0])
        aligned_labels.append(label)
        
        # Remaining subwords get -100
        for remaining_token in word_tokens[1:]:
            all_tokens.append(remaining_token)
            aligned_labels.append(-100)
    
    return all_tokens, aligned_labels


if __name__ == "__main__":
    # Test the function with example data
    import sys
    sys.path.append('.')
    from src.tokenizer_initializer import initialize_xlm_roberta_tokenizer
    
    # Initialize tokenizer
    tokenizer = initialize_xlm_roberta_tokenizer()
    
    # Example sentence and word-level labels
    sentence = "They struggled with complex problems."
    word_texts = ["They", "struggled", "with", "complex", "problems"]
    word_labels = [0, 1, 0, 0, 1]  # Binary labels for example
    
    print("=" * 60)
    print("LABEL ALIGNMENT FUNCTION TEST")
    print("=" * 60)
    print(f"\nOriginal sentence: '{sentence}'")
    print("Original words:", word_texts)
    print("Original labels:", word_labels)
    
    # Test Function 1: align_labels_with_tokens (basic)
    print("\n" + "-" * 60)
    print("FUNCTION 1: align_labels_with_tokens (basic)")
    print("-" * 60)
    
    # Tokenize the full sentence
    tokenized = tokenizer(sentence, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    
    # Call basic alignment function
    basic_labels = align_labels_with_tokens(word_labels, tokens)
    
    print(f"Tokens: {[t.replace('▁', '_') for t in tokens]}")
    print(f"Labels: {[str(l) for l in basic_labels]}")
    
    # Test Function 2: align_labels_with_tokens_using_wordids (using word_ids)
    print("\n" + "-" * 60)
    print("FUNCTION 2: align_labels_with_tokens_using_wordids (using word_ids)")
    print("-" * 60)
    
    # Tokenize with return_offsets_mapping to enable word_ids
    tokenized_advanced = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    
    try:
        advanced_labels = align_labels_with_tokens_using_wordids(word_labels, tokenized_advanced)
        print(f"Tokens: {[t.replace('▁', '_') for t in tokenizer.convert_ids_to_tokens(tokenized_advanced['input_ids'])]}")
        print(f"Labels: {[str(l) for l in advanced_labels]}")
        print(f"Word IDs: {tokenized_advanced.word_ids()}")
    except ValueError as e:
        print(f"Error: {e}")
        print("This tokenizer may not support word_ids() method")
    
    # Test Function 3: align_labels_with_tokens_simple (individual word tokenization)
    print("\n" + "-" * 60)
    print("FUNCTION 3: align_labels_with_tokens_simple (individual word tokenization)")
    print("-" * 60)
    
    simple_tokens, simple_labels = align_labels_with_tokens_simple(word_texts, word_labels, tokenizer)
    
    print(f"Tokens: {[t.replace('▁', '_') for t in simple_tokens]}")
    print(f"Labels: {[str(l) for l in simple_labels]}")
    
    # Detailed breakdown for Function 3
    print("\nDetailed breakdown:")
    current_word_idx = 0
    for i, (token, label) in enumerate(zip(simple_tokens, simple_labels)):
        label_str = str(label)
        display_token = token.replace('▁', '_')
        
        # Determine which word this token belongs to
        if label != -100:  # First subword of a word
            word_info = f"(Word: '{word_texts[current_word_idx]}')"
            current_word_idx += 1
        else:  # Continuation subword
            word_info = f"(continuation of '{word_texts[current_word_idx-1]}')"
        
        print(f"  Token {i}: '{display_token}' -> Label: {label_str} {word_info}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original words: {len(word_texts)}")
    print(f"Function 1 tokens: {len(tokens)}")
    print(f"Function 2 tokens: {len(tokenizer.convert_ids_to_tokens(tokenized_advanced['input_ids']))}")
    print(f"Function 3 tokens: {len(simple_tokens)}")

