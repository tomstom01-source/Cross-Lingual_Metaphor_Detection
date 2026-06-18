import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import sys
sys.path.append('.')

from src.extractors import extract_vuamc_data
from src.initializers import initialize_xlm_roberta_tokenizer, initialize_xlm_roberta_model_for_classification
from src.data_loader import TokenClassificationDataset
from src.label_alignment import align_labels_with_tokens_using_wordids

def analyze_logits(model, tokenizer, xml_file, max_length=512, batch_size=8):
    """
    Extract and analyze pre-softmax logits from model outputs.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        xml_file: Path to XML file
        max_length: Maximum sequence length
        batch_size: Batch size for processing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Extract data
    sentences, words_only, word_labels = extract_vuamc_data(xml_file)
    
    # Use validation set (last 20%)
    train_size = int(0.8 * len(sentences))
    val_sentences = sentences[train_size:]
    val_words_only = words_only[train_size:]
    val_word_labels = word_labels[train_size:]
    
    # Create dataset
    dataset = TokenClassificationDataset(
        sentences=val_sentences,
        words_only=val_words_only,
        word_labels=val_word_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        label_alignment_func=align_labels_with_tokens_using_wordids
    )
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_logits = []
    all_predictions = []
    all_labels = []
    
    print("Extracting logits from validation set...")
    print(f"Processing {len(val_sentences)} sentences...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass without labels to get logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions (argmax over logits)
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to CPU for analysis
            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Filter out padding tokens (-100)
    valid_mask = all_labels != -100
    
    # Get logits and labels for valid tokens only
    valid_logits = all_logits[valid_mask]
    valid_predictions = all_predictions[valid_mask]
    valid_labels = all_labels[valid_mask]
    
    print(f"\n{'='*60}")
    print("LOGIT ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total tokens (including padding): {all_logits.numel()}")
    print(f"Valid tokens (excluding padding): {valid_logits.numel()}")
    print(f"Token labels (0=non-metaphor, 1=metaphor)")
    
    # Count label distribution
    label_counts = torch.bincount(valid_labels)
    print(f"\nLabel distribution in validation set:")
    print(f"  Label 0 (non-metaphor): {label_counts[0].item()}")
    print(f"  Label 1 (metaphor): {label_counts[1].item()}")
    
    # Calculate softmax probabilities from logits
    probs = softmax(valid_logits, dim=-1)
    
    # Get probabilities for predicted class
    predicted_probs = probs[range(len(valid_predictions)), valid_predictions]
    
    # Separate logits by true label
    non_metaphor_logits = valid_logits[valid_labels == 0]
    metaphor_logits = valid_logits[valid_labels == 1]
    
    print(f"\n{'='*60}")
    print("LOGIT STATISTICS BY TRUE LABEL")
    print(f"{'='*60}")
    
    # Non-metaphor token logits
    print(f"\nNon-metaphor tokens (label=0): {len(non_metaphor_logits)} tokens")
    print(f"  Logit for class 0: mean={non_metaphor_logits[:, 0].mean():.4f}, std={non_metaphor_logits[:, 0].std():.4f}")
    print(f"  Logit for class 1: mean={non_metaphor_logits[:, 1].mean():.4f}, std={non_metaphor_logits[:, 1].std():.4f}")
    
    # Metaphor token logits
    print(f"\nMetaphor tokens (label=1): {len(metaphor_logits)} tokens")
    print(f"  Logit for class 0: mean={metaphor_logits[:, 0].mean():.4f}, std={metaphor_logits[:, 1].std():.4f}")
    print(f"  Logit for class 1: mean={metaphor_logits[:, 1].mean():.4f}, std={metaphor_logits[:, 1].std():.4f}")
    
    # Softmax probabilities
    print(f"\n{'='*60}")
    print("SOFTMAX PROBABILITIES BY TRUE LABEL")
    print(f"{'='*60}")
    
    non_metaphor_probs = probs[valid_labels == 0]
    metaphor_probs = probs[valid_labels == 1]
    
    print(f"\nNon-metaphor tokens (label=0):")
    print(f"  P(class=0): mean={non_metaphor_probs[:, 0].mean():.4f}, std={non_metaphor_probs[:, 0].std():.4f}")
    print(f"  P(class=1): mean={non_metaphor_probs[:, 1].mean():.4f}, std={non_metaphor_probs[:, 1].std():.4f}")
    
    print(f"\nMetaphor tokens (label=1):")
    print(f"  P(class=0): mean={metaphor_probs[:, 0].mean():.4f}, std={metaphor_probs[:, 0].std():.4f}")
    print(f"  P(class=1): mean={metaphor_probs[:, 1].mean():.4f}, std={metaphor_probs[:, 1].std():.4f}")
    
    # Calculate accuracy metrics
    correct = (valid_predictions == valid_labels).sum().item()
    accuracy = correct / len(valid_predictions)
    
    # Calculate confusion matrix
    tp = ((valid_predictions == 1) & (valid_labels == 1)).sum().item()
    tn = ((valid_predictions == 0) & (valid_labels == 0)).sum().item()
    fp = ((valid_predictions == 1) & (valid_labels == 0)).sum().item()
    fn = ((valid_predictions == 0) & (valid_labels == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION METRICS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(valid_predictions)})")
    print(f"Precision (metaphor): {precision:.4f} (TP={tp}, FP={fp})")
    print(f"Recall (metaphor): {recall:.4f} (TP={tp}, FN={fn})")
    print(f"F1 Score (metaphor): {f1:.4f}")
    
    # Show sample logits for first 10 valid tokens
    print(f"\n{'='*60}")
    print("SAMPLE LOGITS (first 10 valid tokens)")
    print(f"{'='*60}")
    
    sample_idx = 0
    for i in range(valid_logits.shape[0]):
        if sample_idx >= 10:
            break
        true_label = valid_labels[i].item()
        pred_label = valid_predictions[i].item()
        logit_0 = valid_logits[i, 0].item()
        logit_1 = valid_logits[i, 1].item()
        prob_0 = probs[i, 0].item()
        prob_1 = probs[i, 1].item()
        
        print(f"Token {i}: True={true_label}, Pred={pred_label}")
        print(f"  Logits: [{logit_0:.4f}, {logit_1:.4f}]")
        print(f"  Probs: [{prob_0:.4f}, {prob_1:.4f}]")
        print(f"  Confidence: {max(prob_0, prob_1):.4f}")
        
        sample_idx += 1
    
    print(f"\n{'='*60}")
    print("LOSS INTERPRETATION")
    print(f"{'='*60}")
    print(f"The cross-entropy loss of {0.2809} indicates:")
    print(f"1. Model is learning but has room for improvement")
    print(f"2. Non-metaphor tokens have ~{non_metaphor_probs[:, 0].mean()*100:.1f}% confidence for class 0")
    print(f"3. Metaphor tokens have ~{metaphor_probs[:, 1].mean()*100:.1f}% confidence for class 1")
    print(f"4. The model makes {fp} false positives and {fn} false negatives")
    print(f"5. Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    if accuracy > 0.8:
        print(f"\n[!] High accuracy but check class balance")
        print(f"    Model predicts {label_counts[0].item()}/{label_counts.sum().item()} as non-metaphor")
        if label_counts[1].item() > 0 and tp == 0:
            print(f"    WARNING: Zero metaphors detected! Model learned to predict majority class only.")
    elif accuracy > 0.7:
        print(f"\n[*] Moderate performance (70-80% accuracy)")
    else:
        print(f"\n[!] Low performance (<70% accuracy)")

if __name__ == "__main__":
    XML_FILE = "data/VUAMC/2541/VUAMC_first_100_sentences.xml"
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    
    print("="*60)
    print("LOGIT ANALYSIS FOR METAPHOR DETECTION MODEL")
    print("="*60)
    
    # Load tokenizer and model
    print("\nLoading trained model...")
    tokenizer = initialize_xlm_roberta_tokenizer()
    model = initialize_xlm_roberta_model_for_classification(num_labels=2)
    model.load_state_dict(torch.load('best_metaphor_model.bin', map_location='cpu'))
    print("Model loaded successfully!")
    
    # Analyze logits
    analyze_logits(model, tokenizer, XML_FILE, MAX_LENGTH, BATCH_SIZE)
