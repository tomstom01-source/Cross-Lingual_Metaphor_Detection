import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

def calculate_class_weights(train_word_labels):
    """
    Calculate class weights for handling class imbalance.
    
    The weight for the metaphor class (1) is calculated as the ratio of literal labels (0)
    to metaphor labels (1) in the training data. This penalizes any tendency of the model to bias towards predicting the majority label (literals).
    
    Args:
        train_word_labels (list): List of word label lists for training data
        
    Returns:
        torch.Tensor: Weight tensor for classes [weight_0, weight_1]
    """
    # Count literal (0) and metaphor (1) labels
    literal_count = 0
    metaphor_count = 0
    
    for labels in train_word_labels:
        literal_count += sum(1 for label in labels if label == 0)
        metaphor_count += sum(1 for label in labels if label == 1)
    
    print(f"\nClass distribution in training data:")
    print(f"  Literal (0): {literal_count}")
    print(f"  Metaphor (1): {metaphor_count}")
    
    # Calculate class 0 weight as 1.0 (baseline)
    weight_0 = 1.0
    # Calculate class 1 weight as (#class_0 / #class_1)
    weight_1 = literal_count / metaphor_count if metaphor_count > 0 else 1.0
    
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)
    
    print(f"Class weights: [{weight_0:.2f}, {weight_1:.2f}]")
    print(f"  Literal weight: {weight_0:.2f} (baseline)")
    print(f"  Metaphor weight: {weight_1:.2f} (punishes misclassification {weight_1:.2f}x more)")
    
    return class_weights


def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5,
                class_weights=None, save_path='metaphor_model.bin'):
    """
    Train the metaphor detection model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (cuda/cpu)
        epochs (int): Number of training epochs
        learning_rate (float): Learning rate for AdamW
        class_weights (torch.Tensor): Class weights for handling imbalance [weight_0, weight_1]
        save_path (str|None): Where to save the best-by-val-loss checkpoint.
            If None, the model is not saved to disk.
    """
    # Move model to device
    model.to(device)
    
    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Loss function with class weights
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        print(f"Using weighted cross-entropy loss")
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        print(f"Using standard cross-entropy loss")
    
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
        
        for _, batch in enumerate(train_progress):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass (get logits only, not internal loss)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss manually with class weights
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
            
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
                    
                    # Forward pass (get logits only, not internal loss)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Compute loss manually with class weights
                    loss = loss_fn(logits.view(-1, 2), labels.view(-1))
                    
                    total_val_loss += loss.item()
                    val_progress.set_postfix({'val_loss': f"{loss.item():.4f}"})
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved to {save_path} (validation loss: {avg_val_loss:.4f})")
                else:
                    print(f"New best validation loss: {avg_val_loss:.4f} (not saved)")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}")
    
    return model
