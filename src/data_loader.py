import torch
from torch.utils.data import Dataset, DataLoader


class TokenClassificationDataset(Dataset):
    """
    Custom Dataset for token classification tasks.
    
    This dataset handles tokenized inputs with input_ids, attention_mask, and labels
    for training/evaluating token classification models.
    """
    
    def __init__(self, sentences, words_only, word_labels, tokenizer, max_length=128, label_alignment_func=None):
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
            padding='max_length',
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
                token_strings = tokenizer.convert_ids_to_tokens(text_encoding['input_ids'])
                aligned = label_alignment_func(word_labels_for_sentence, text_encoding, token_strings)
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


def create_data_loader(sentences, words_only, word_labels, tokenizer, batch_size=16, max_length=128, 
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
                       tokenizer=None, batch_size=16, max_length=128, 
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
