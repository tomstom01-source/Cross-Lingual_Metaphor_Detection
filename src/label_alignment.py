def align_labels_with_tokens_using_wordids(word_labels, tokens, token_strings=None):
    """
    Label alignment using word_ids() method if available.
    
    This function aligns word-level labels with token-level labels:
    - The first alphanumeric-containing subword of each word gets the actual label (1 or 0)
    - Leading punctuation-only (non-alphanumeric) subwords (e.g. the opening double quote in '"complex"') get -100
    - Trailing subwords get -100
    - Punctuation tokens get -100
    - Special tokens (CLS, SEP, etc.) get -100
    
    Args:
        word_labels (list): List of labels corresponding to each word (from words_only)
        tokens: Tokenized output from tokenizer (with word_ids() method) from full sentence
        token_strings (list, optional): List of token strings (from tokenizer.convert_ids_to_tokens).
            When provided, punctuation-only leading subwords are skipped so the label lands
            on the first subword that contains at least one alphanumeric character.
    
    Returns:
        list: Prefix-aligned labels at the token level
    """
    if not hasattr(tokens, 'word_ids'):
        raise ValueError("Tokenizer output does not have word_ids() method")

    word_ids = tokens.word_ids()
    n = len(word_ids)
    if token_strings is None:
        raise ValueError(
            "token_strings parameter is required. Without token strings, the function cannot "
            "distinguish punctuation-only tokens (e.g., opening quotes, brackets) from actual words. "
            "This would lead to incorrect classification of punctuation marks as literal/metaphor "
            "when classification should only be performed on words. "
            "Please provide token_strings from tokenizer.convert_ids_to_tokens()."
        )

    aligned_labels = [-100] * n

    # Walk the tokens grouped by word_id. The tokenizer assigns a *new* word_id to
    # every whitespace-separated chunk, INCLUDING standalone punctuation such as an
    # opening quote/bracket ("‘", "(") or an infix symbol ("&"). Those chunks are not
    # present in words_only, so we must skip them WITHOUT consuming a words_only entry.
    # We therefore advance our own pointer (`word_ptr`) into word_labels only when a
    # word-group actually contains alphanumeric text (i.e. a real word). This keeps the
    # labels aligned even when the tokenizer "sees" more words than words_only lists.
    word_ptr = 0
    i = 0
    while i < n:
        wid = word_ids[i]
        if wid is None:
            # Special tokens (CLS, SEP, pad) -> ignore
            i += 1
            continue

        # Gather the contiguous run of tokens sharing this word_id
        j = i
        first_alnum_idx = None
        while j < n and word_ids[j] == wid:
            clean = token_strings[j].lstrip('▁')
            if first_alnum_idx is None and any(c.isalnum() for c in clean):
                first_alnum_idx = j
            j += 1

        if first_alnum_idx is not None:
            # A real word: assign its label to the first alphanumeric subword.
            # Leading punctuation subwords (e.g. the quote in "‘word") stay -100.
            if word_ptr < len(word_labels):
                aligned_labels[first_alnum_idx] = word_labels[word_ptr]
                word_ptr += 1
        # else: punctuation-only group (e.g. "(", "‘", "&") -> leave all -100,
        # and do NOT advance word_ptr.

        i = j

    return aligned_labels
