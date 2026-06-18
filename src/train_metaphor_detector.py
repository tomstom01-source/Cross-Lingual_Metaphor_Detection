import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import sys
sys.path.append('.')

from src.extractors import extract_vuamc_data
from src.initializers import initialize_xlm_roberta_tokenizer, initialize_xlm_roberta_model_for_classification
from src.label_alignment import align_labels_with_tokens_using_wordids
from src.data_loader import create_data_loaders


def prepare_train_and_validation_split(xml_file, train_split_ratio=0.8):
    """
    Prepare training and validation split from XML file, alloting train_split_ratio of extracted sentences to training and the rest to validation.
    
    Args:
        xml_file (str): Path to XML file (VUAMC format)
        train_split_ratio (float): Ratio of sentences to use for training (default: 0.8)
        
    Returns:
        tuple: (train_sentences, train_words_only, train_word_labels, 
               val_sentences, val_words_only, val_word_labels)
    """
    # Extract data from XML
    sentences, words_only, word_labels = extract_vuamc_data(xml_file)
    
    print(f"Extracted {len(sentences)} sentences from XML file")
    print(f"Total words (for classification): {sum(len(labels) for labels in word_labels)}")
    print(f"Total metaphors: {sum(sum(labels) for labels in word_labels)}")
    
    # Split into train and validation
    train_size = int(train_split_ratio * len(sentences))
    
    train_sentences = sentences[:train_size]
    train_words_only = words_only[:train_size]
    train_word_labels = word_labels[:train_size]
    val_sentences = sentences[train_size:]
    val_words_only = words_only[train_size:]
    val_word_labels = word_labels[train_size:]
    
    print(f"Training samples: {len(train_sentences)}")
    print(f"Validation samples: {len(val_sentences)}")
    
    return (train_sentences, train_words_only, train_word_labels, 
            val_sentences, val_words_only, val_word_labels)


def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5):
    """
    Train the metaphor detection model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for AdamW
    """
    # Move model to device
    model.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Training batches per epoch: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader) if val_loader else 0}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(train_progress):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # Update progress bar
            train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader:
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
                
                for batch in val_progress:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    total_val_loss += loss.item()
                    val_progress.set_postfix({'val_loss': f"{loss.item():.4f}"})
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_metaphor_model.bin')
                print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    
    return model


def main():
    """
    Main training function.
    """
    print("="*60)
    print("METAPHOR DETECTION MODEL TRAINING")
    print("="*60)
    
    # Configuration
    XML_FILE = "data/VUAMC/2541/VUAMC_first_100_sentences.xml"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    print("\nInitializing tokenizer and model...")
    tokenizer = initialize_xlm_roberta_tokenizer()
    model = initialize_xlm_roberta_model_for_classification(num_labels=2)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {model.num_parameters()}")
    
    # Prepare data
    print("\nPreparing training and validation split on data...")
    (train_sentences, train_words_only, train_word_labels, 
     val_sentences, val_words_only, val_word_labels) = prepare_train_and_validation_split(
        XML_FILE)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_sentences=train_sentences,
        train_words_only=train_words_only,
        train_word_labels=train_word_labels,
        val_sentences=val_sentences,
        val_words_only=val_words_only,
        val_word_labels=val_word_labels,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        label_alignment_func=align_labels_with_tokens_using_wordids
    )
    
    # Train model
    print("\nStarting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Save final model
    final_model_path = 'final_metaphor_model.bin'
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Save tokenizer
    tokenizer.save_pretrained('./metaphor_tokenizer')
    print("Tokenizer saved to: ./metaphor_tokenizer")


if __name__ == "__main__":
    main()
