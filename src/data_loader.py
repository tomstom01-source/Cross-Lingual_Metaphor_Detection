import torch
from torch.utils.data import Dataset, DataLoader


class TokenClassificationDataset(Dataset):
    """
    Custom Dataset for token classification tasks.
    
    This dataset handles tokenized inputs with input_ids, attention_mask, and labels
    for training/evaluating token classification models.
    """
    
    def __init__(self, sentences, words_only, word_labels, tokenizer, max_length=512, label_alignment_func=None):
        """
        Initialize the dataset.
        
        Args:
            sentences (list): List of sentence strings (with punctuation preserved)
            words_only (list): List of word lists (only words, no punctuation)
            word_labels (list): List of label lists corresponding to words_only
            tokenizer: Tokenizer object to use for tokenization
            max_length (int): Maximum sequence length for truncation/padding
            label_alignment_func (function): Function to align word-level labels with token-level labels
        """
        self.sentences = sentences
        self.words_only = words_only
        self.word_labels = word_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_alignment_func = label_alignment_func
        
        # Tokenize all sentences and prepare encodings
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding='max_length',  # Explicitly pad to max_length
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        # Align labels with tokens if alignment function is provided
        if label_alignment_func is not None:
            self.aligned_labels = []
            for i, (sentence, word_labels_for_sentence) in enumerate(zip(sentences, word_labels)):
                text_encoding = tokenizer(
                    sentence,
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                    return_offsets_mapping=True,
                    add_special_tokens=True
                )
                # Align labels using word_ids() method
                aligned = label_alignment_func(word_labels_for_sentence, text_encoding)
                # Use the actual tokenized length from the main encodings
                actual_length = len(self.encodings['input_ids'][i])
                # Pad/truncate to match actual length
                aligned_padded = aligned + [-100] * (actual_length - len(aligned))
                aligned_padded = aligned_padded[:actual_length]
                self.aligned_labels.append(aligned_padded)
        else:
            # Use labels as-is (assuming they're already token-aligned)
            self.aligned_labels = word_labels
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sentences)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
        }
        
        # Ensure tensor format
        if isinstance(self.aligned_labels[idx], list):
            item['labels'] = torch.tensor(self.aligned_labels[idx], dtype=torch.long)
        else:
            item['labels'] = self.aligned_labels[idx]
        
        return item


def create_data_loader(sentences, words_only, word_labels, tokenizer, batch_size=16, max_length=512, 
                       shuffle=True, num_workers=0, label_alignment_func=None):
    """
    Create a DataLoader for token classification.
    
    Args:
        sentences (list): List of sentence strings (with punctuation preserved)
        words_only (list): List of word lists (only words, no punctuation)
        word_labels (list): List of label lists corresponding to words_only
        tokenizer: Tokenizer object to use for tokenization
        batch_size (int): Batch size for the DataLoader
        max_length (int): Maximum sequence length
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes for data loading
        label_alignment_func (function): Function to align word-level labels with token-level labels
        
    Returns:
        DataLoader: Configured DataLoader for the dataset
    """
    # Create dataset
    dataset = TokenClassificationDataset(
        sentences=sentences,
        words_only=words_only,
        word_labels=word_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        label_alignment_func=label_alignment_func
    )
    
    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return data_loader


def create_data_loaders(train_sentences, train_words_only, train_word_labels, 
                       val_sentences=None, val_words_only=None, val_word_labels=None,
                       tokenizer=None, batch_size=16, max_length=512, 
                       label_alignment_func=None):
    """
    Create train and validation DataLoaders.
    
    Args:
        train_sentences (list): List of training sentence strings (with punctuation)
        train_words_only (list): List of training word lists (only words)
        train_word_labels (list): List of training label lists
        val_sentences (list): Optional list of validation sentence strings
        val_words_only (list): Optional list of validation word lists
        val_word_labels (list): Optional list of validation label lists
        tokenizer: Tokenizer object
        batch_size (int): Batch size for DataLoaders
        max_length (int): Maximum sequence length
        label_alignment_func (function): Function to align labels
        
    Returns:
        tuple: (train_loader, val_loader) where val_loader is None if no validation data provided
    """
    train_loader = create_data_loader(
        sentences=train_sentences,
        words_only=train_words_only,
        word_labels=train_word_labels,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=True,
        label_alignment_func=label_alignment_func
    )
    
    val_loader = None
    if val_sentences is not None and val_words_only is not None and val_word_labels is not None:
        val_loader = create_data_loader(
            sentences=val_sentences,
            words_only=val_words_only,
            word_labels=val_word_labels,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=False,
            label_alignment_func=label_alignment_func
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the DataLoader with sample data
    import sys
    sys.path.append('.')
    from src.initializers import initialize_xlm_roberta_tokenizer
    from src.label_alignment import align_labels_with_tokens_using_wordids
    
    # Initialize tokenizer
    tokenizer = initialize_xlm_roberta_tokenizer()
    
    # Sample data
    train_texts = [
        "They struggled with complex problems.",
        "The system worked perfectly yesterday.",
        "She enjoyed the beautiful sunset.",
        "I am running out of examples."
    ]
    
    # Word-level labels
    train_labels = [
        [0, 1, 0, 0, 1],  
        [0, 0, 0, 0, 0],      
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1]
    ]
    
    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)
    print(f"\nSample texts: {train_texts}")
    print(f"Sample labels: {train_labels}")
    
    # Test with label alignment
    print("\n" + "-" * 60)
    print("Testing with label alignment")
    print("-" * 60)
    
    train_loader, val_loader = create_data_loaders(
        train_texts=train_texts,
        train_labels=train_labels,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512,
        label_alignment_func=align_labels_with_tokens_using_wordids
    )
    
    print(f"Train loader batch size: {train_loader.batch_size}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Test first batch
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        
        # Display first sample in batch
        if batch_idx == 0:
            print(f"\n  First sample:")
            print(f"    input_ids: {batch['input_ids'][0][:10]}...")  # First 10 tokens
            print(f"    attention_mask: {batch['attention_mask'][0][:10]}...")
            print(f"    labels: {batch['labels'][0][:10]}...")
            
            # Decode first few tokens
            decoded = tokenizer.decode(batch['input_ids'][0][:10], skip_special_tokens=True)
            print(f"    Decoded (first 10 tokens): '{decoded}...'")
    
    # Test without label alignment (assuming pre-aligned labels)
    print("\n" + "-" * 60)
    print("Testing without label alignment (pre-aligned)")
    print("-" * 60)
    
    # Create pre-aligned labels (simplified example with consistent length)
    base_labels = [
        [-100, 0, 1, -100, 0, 0, 1, 0],
        [-100, 0, 0, 0, 0, 0, 0, 0],
        [-100, 0, 1, 0, 1, 0],
        [-100, 0, 1, 1, 0, 0, 1]
    ]
    # Pad to 512
    pre_aligned_labels = [labels + [-100] * (512 - len(labels)) for labels in base_labels]
    
    test_train_loader_pre_aligned_labels = create_data_loader(
        texts=train_texts,
        labels=pre_aligned_labels,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=512
    )
    
    print(f"Train loader batch size: {test_train_loader_pre_aligned_labels.batch_size}")
    print(f"Number of batches: {len(test_train_loader_pre_aligned_labels)}")
    
    print("\n" + "=" * 60)
    print("DATALOADER TEST COMPLETE")
    print("=" * 60)
