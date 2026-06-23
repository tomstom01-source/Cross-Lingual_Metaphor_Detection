import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import sys
sys.path.append('.')

from src.extractors import extract_vuamc_data
from src.data_loader import TokenClassificationDataset
from src.label_alignment import align_labels_with_tokens_using_wordids


def analyze_logits(model, tokenizer, xml_file, max_length=128, batch_size=8):
    """
    Extract and analyze pre-softmax logits from model outputs on the validation
    split.  Reports:
      - Logit/probability statistics by true label
      - Confidence distribution across prediction-probability buckets
      - Full confusion matrix (TP / TN / FP / FN)
      - Per-class and macro-averaged precision, recall, F1
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # ── Data preparation ──────────────────────────────────────────────────────
    sentences, words_only, word_labels = extract_vuamc_data(xml_file)

    train_size = int(0.8 * len(sentences))
    val_sentences   = sentences[train_size:]
    val_words_only  = words_only[train_size:]
    val_word_labels = word_labels[train_size:]

    dataset = TokenClassificationDataset(
        sentences=val_sentences,
        words_only=val_words_only,
        word_labels=val_word_labels,
        tokenizer=tokenizer,
        max_length=max_length,
        label_alignment_func=align_labels_with_tokens_using_wordids
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_logits      = []
    all_predictions = []
    all_labels      = []

    print(f"Extracting logits from validation set ({len(val_sentences)} sentences)...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            logits      = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    all_logits      = torch.cat(all_logits,      dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels      = torch.cat(all_labels,      dim=0)

    # Mask out padding / special tokens (label == -100)
    valid_mask       = all_labels != -100
    valid_logits     = all_logits[valid_mask]
    valid_predictions = all_predictions[valid_mask]
    valid_labels     = all_labels[valid_mask]

    probs = softmax(valid_logits, dim=-1)   # (N, 2)

    n_total     = valid_labels.numel()
    label_counts = torch.bincount(valid_labels, minlength=2)
    n_literal   = label_counts[0].item()
    n_metaphor  = label_counts[1].item()

    # Separate by true class
    non_metaphor_logits = valid_logits[valid_labels == 0]
    metaphor_logits     = valid_logits[valid_labels == 1]
    non_metaphor_probs  = probs[valid_labels == 0]
    metaphor_probs      = probs[valid_labels == 1]

    # ── Confusion matrix ──────────────────────────────────────────────────────
    tp = int(((valid_predictions == 1) & (valid_labels == 1)).sum())
    tn = int(((valid_predictions == 0) & (valid_labels == 0)).sum())
    fp = int(((valid_predictions == 1) & (valid_labels == 0)).sum())
    fn = int(((valid_predictions == 0) & (valid_labels == 1)).sum())

    # Per-class metrics
    prec_metaphor  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_metaphor   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_metaphor    = (2 * prec_metaphor * rec_metaphor / (prec_metaphor + rec_metaphor)
                      if (prec_metaphor + rec_metaphor) > 0 else 0.0)

    prec_literal   = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_literal    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_literal     = (2 * prec_literal * rec_literal / (prec_literal + rec_literal)
                      if (prec_literal + rec_literal) > 0 else 0.0)

    # Macro average (unweighted)
    macro_prec = (prec_literal + prec_metaphor) / 2
    macro_rec  = (rec_literal  + rec_metaphor)  / 2
    macro_f1   = (f1_literal   + f1_metaphor)   / 2

    # Weighted average
    w_prec = (prec_literal * n_literal + prec_metaphor * n_metaphor) / n_total
    w_rec  = (rec_literal  * n_literal + rec_metaphor  * n_metaphor) / n_total
    w_f1   = (f1_literal   * n_literal + f1_metaphor   * n_metaphor) / n_total

    accuracy = (tp + tn) / n_total

    # Confidence of the predicted class for each token
    confidence = probs[range(n_total), valid_predictions]

    # Confidence by correctness
    correct_mask   = valid_predictions == valid_labels
    conf_correct   = confidence[correct_mask].mean().item()
    conf_incorrect = confidence[~correct_mask].mean().item() if (~correct_mask).any() else 0.0

    # ── Print ─────────────────────────────────────────────────────────────────
    SEP = '=' * 60

    print(f"\n{SEP}")
    print("LOGIT / PROBABILITY STATISTICS BY TRUE CLASS")
    print(SEP)
    print(f"\nLiteral tokens (label=0):  {n_literal} tokens")
    print(f"  Logit class-0 : mean={non_metaphor_logits[:, 0].mean():.4f}"
          f"  std={non_metaphor_logits[:, 0].std():.4f}")
    print(f"  Logit class-1 : mean={non_metaphor_logits[:, 1].mean():.4f}"
          f"  std={non_metaphor_logits[:, 1].std():.4f}")
    print(f"  P(literal)    : mean={non_metaphor_probs[:, 0].mean():.4f}"
          f"  std={non_metaphor_probs[:, 0].std():.4f}")
    print(f"  P(metaphor)   : mean={non_metaphor_probs[:, 1].mean():.4f}"
          f"  std={non_metaphor_probs[:, 1].std():.4f}")

    print(f"\nMetaphor tokens (label=1): {n_metaphor} tokens")
    print(f"  Logit class-0 : mean={metaphor_logits[:, 0].mean():.4f}"
          f"  std={metaphor_logits[:, 0].std():.4f}")
    print(f"  Logit class-1 : mean={metaphor_logits[:, 1].mean():.4f}"
          f"  std={metaphor_logits[:, 1].std():.4f}")
    print(f"  P(literal)    : mean={metaphor_probs[:, 0].mean():.4f}"
          f"  std={metaphor_probs[:, 0].std():.4f}")
    print(f"  P(metaphor)   : mean={metaphor_probs[:, 1].mean():.4f}"
          f"  std={metaphor_probs[:, 1].std():.4f}")

    # ── Confidence distribution ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PREDICTION CONFIDENCE DISTRIBUTION")
    print(SEP)
    buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80),
               (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]
    print(f"\n{'Bucket':>12}  {'Count':>6}  {'%':>6}  {'Correct':>8}  {'Acc':>6}")
    print("-" * 48)
    for lo, hi in buckets:
        mask = (confidence >= lo) & (confidence < hi)
        cnt  = int(mask.sum())
        if cnt == 0:
            print(f" [{lo:.0%}-{min(hi, 1.0):.0%})  {cnt:>6}  {'':>6}  {'':>8}")
            continue
        acc_b = float((correct_mask & mask).sum()) / cnt
        label = f"[{lo:.0%}-{'100%' if hi > 1 else f'{hi:.0%}'})"
        print(f" {label:>12}  {cnt:>6}  {cnt/n_total:>6.1%}  "
              f"{int((correct_mask & mask).sum()):>8}  {acc_b:>6.1%}")

    print(f"\nOverall confidence  — correct  : {conf_correct:.4f}")
    print(f"Overall confidence  — incorrect: {conf_incorrect:.4f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("CONFUSION MATRIX  (rows = true, cols = predicted)")
    print(SEP)
    w = max(len(str(max(tp, tn, fp, fn))), 6)
    print(f"\n{'':>16}  {'Pred Literal':>{w+2}}  {'Pred Metaphor':>{w+2}}")
    print(f"  {'True Literal':>14}  {tn:>{w+2}}  {fp:>{w+2}}")
    print(f"  {'True Metaphor':>14}  {fn:>{w+2}}  {tp:>{w+2}}")
    print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # ── Per-class and aggregate metrics ──────────────────────────────────────
    print(f"\n{SEP}")
    print("CLASSIFICATION METRICS")
    print(SEP)
    print(f"\n  {'Class':>12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}")
    print(f"  {'-'*52}")
    print(f"  {'Literal (0)':>12}  {prec_literal:>10.4f}  {rec_literal:>8.4f}"
          f"  {f1_literal:>8.4f}  {n_literal:>8}")
    print(f"  {'Metaphor (1)':>12}  {prec_metaphor:>10.4f}  {rec_metaphor:>8.4f}"
          f"  {f1_metaphor:>8.4f}  {n_metaphor:>8}")
    print(f"  {'-'*52}")
    print(f"  {'Macro avg':>12}  {macro_prec:>10.4f}  {macro_rec:>8.4f}"
          f"  {macro_f1:>8.4f}  {n_total:>8}")
    print(f"  {'Weighted avg':>12}  {w_prec:>10.4f}  {w_rec:>8.4f}"
          f"  {w_f1:>8.4f}  {n_total:>8}")
    print(f"\n  Accuracy: {accuracy:.4f}  ({tp + tn}/{n_total} correct tokens)")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("INTERPRETATION")
    print(SEP)
    print(f"\n  Val set: {n_literal} literal + {n_metaphor} metaphor tokens"
          f"  ({n_metaphor/n_total:.1%} metaphor rate)")
    print(f"  Literal tokens — model assigns P(literal)={non_metaphor_probs[:, 0].mean():.1%} on average")
    print(f"  Metaphor tokens — model assigns P(metaphor)={metaphor_probs[:, 1].mean():.1%} on average")
    if tp == 0:
        print("\n  WARNING: zero metaphors detected — model is predicting the majority class only.")
    elif rec_metaphor < 0.30:
        print(f"\n  Low metaphor recall ({rec_metaphor:.1%}): model still misses most metaphors.")
    elif f1_metaphor < 0.50:
        print(f"\n  Metaphor F1 of {f1_metaphor:.2f} — moderate; further training or data may help.")
    else:
        print(f"\n  Metaphor F1 of {f1_metaphor:.2f} — reasonable performance for a small training set.")
