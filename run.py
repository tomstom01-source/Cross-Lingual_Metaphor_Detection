"""
Train and validate a metaphor-detection token classifier on the VUAMC corpus.

Command line configuration example:

    python run.py --num-sentences 500 --max-length 128 --epochs 3

Run `python run.py --help` for the full list of options.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

# Make stdout UTF-8 tolerant so the script never crashes on consoles whose
# default encoding (e.g. Windows cp1252) cannot represent a stray character.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

sys.path.append('.')

from src.extractors import extract_xml_subset, extract_vuamc_data
from src.initializers import (
    initialize_xlm_roberta_tokenizer,
    initialize_xlm_roberta_model_for_classification,
)
from src.label_alignment import align_labels_with_tokens_using_wordids
from src.data_loader import create_data_loaders
from src.train_metaphor_detector import calculate_class_weights, train_model


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    """
    Make a run reproducible across random / numpy / torch (CPU & CUDA). 
    May not be exhaustive
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    """
    Run inference over `loader` and compute token-level classification metrics.

    Padding/sub-word/special tokens (label == -100) are excluded.

    Returns a dict with the confusion-matrix counts and per-class metrics.
    """
    model.to(device)
    model.eval()

    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds  = torch.argmax(logits, dim=-1)

            mask   = labels != -100
            preds  = preds[mask]
            labels = labels[mask]

            tp += int(((preds == 1) & (labels == 1)).sum())
            tn += int(((preds == 0) & (labels == 0)).sum())
            fp += int(((preds == 1) & (labels == 0)).sum())
            fn += int(((preds == 0) & (labels == 1)).sum())

    n_total    = tp + tn + fp + fn
    n_literal  = tn + fp
    n_metaphor = tp + fn

    def prf(p_num, p_den, r_num, r_den):
        p = p_num / p_den if p_den else 0.0
        r = r_num / r_den if r_den else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    prec_m, rec_m, f1_m = prf(tp, tp + fp, tp, tp + fn)   # metaphor
    prec_l, rec_l, f1_l = prf(tn, tn + fn, tn, tn + fp)   # literal

    macro = ((prec_l + prec_m) / 2, (rec_l + rec_m) / 2, (f1_l + f1_m) / 2)
    if n_total:
        weighted = (
            (prec_l * n_literal + prec_m * n_metaphor) / n_total,
            (rec_l  * n_literal + rec_m  * n_metaphor) / n_total,
            (f1_l   * n_literal + f1_m   * n_metaphor) / n_total,
        )
    else:
        weighted = (0.0, 0.0, 0.0)
    accuracy = (tp + tn) / n_total if n_total else 0.0

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'n_total': n_total, 'n_literal': n_literal, 'n_metaphor': n_metaphor,
        'literal':  (prec_l, rec_l, f1_l),
        'metaphor': (prec_m, rec_m, f1_m),
        'macro': macro, 'weighted': weighted, 'accuracy': accuracy,
    }


def print_report(metrics):
    """Pretty-print the confusion matrix and metric tables from evaluate()."""
    sep = '=' * 60
    print(f"\n{sep}\nCONFUSION MATRIX\n{sep}")
    cw = 15  # column width for the count columns
    print(f"\n{'':>16}{'Pred Literal':>{cw}}{'Pred Metaphor':>{cw}}")
    print(f"{'True Literal':>16}{metrics['tn']:>{cw}}{metrics['fp']:>{cw}}")
    print(f"{'True Metaphor':>16}{metrics['fn']:>{cw}}{metrics['tp']:>{cw}}")
    print(f"\n  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}  TP={metrics['tp']}")

    print(f"\n{sep}\nCLASSIFICATION METRICS\n{sep}")
    print(f"\n  {'Class':>13}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'Support':>7}")
    print(f"  {'-' * 52}")
    pl, rl, fl = metrics['literal']
    pm, rm, fm = metrics['metaphor']
    print(f"  {'Literal (0)':>13}  {pl:>9.4f}  {rl:>7.4f}  {fl:>7.4f}  {metrics['n_literal']:>7}")
    print(f"  {'Metaphor (1)':>13}  {pm:>9.4f}  {rm:>7.4f}  {fm:>7.4f}  {metrics['n_metaphor']:>7}")
    print(f"  {'-' * 52}")
    pa, ra, fa = metrics['macro']
    pw, rw, fw = metrics['weighted']
    print(f"  {'Macro avg':>13}  {pa:>9.4f}  {ra:>7.4f}  {fa:>7.4f}  {metrics['n_total']:>7}")
    print(f"  {'Weighted avg':>13}  {pw:>9.4f}  {rw:>7.4f}  {fw:>7.4f}  {metrics['n_total']:>7}")
    print(f"\n  Accuracy: {metrics['accuracy']:.4f}  ({metrics['tp'] + metrics['tn']}/{metrics['n_total']} tokens)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    """Parse command line arguments for training configuration."""
    p = argparse.ArgumentParser(
        description="Train & validate a VUAMC metaphor-detection token classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('-n', '--num-sentences', type=int, default=500,
                   help="Number of sentences to extract from the VUAMC corpus.")
    p.add_argument('--max-length', type=int, default=128,
                   help="Max token sequence length (context window).")
    p.add_argument('--epochs', type=int, default=3, help="Training epochs.")
    p.add_argument('--batch-size', type=int, default=8, help="Batch size.")
    p.add_argument('--learning-rate', '--lr', type=float, default=2e-5,
                   help="AdamW learning rate.")
    p.add_argument('--train-split', type=float, default=0.8,
                   help="Fraction of sentences used for training (rest for validation).")
    p.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument('--source-xml', type=str, default='data/VUAMC/VUAMC.xml',
                   help="Path to the VUAMC TEI XML corpus (included in repo).")
    p.add_argument('--no-class-weights', action='store_true',
                   help="Disable the weighted loss that compensates for class imbalance.")
    p.add_argument('--save-model', type=str, default='metaphor_model.bin',
                   help="Filename to save in models/ directory. Set to '' to skip saving.")
    p.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                   help="Force a device. Defaults to CUDA when available.")
    return p.parse_args()


def main():
    """Main training and validation pipeline."""
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("VUAMC METAPHOR DETECTION - TRAIN & VALIDATE")
    print("=" * 60)
    print(f"  sentences={args.num_sentences}  max_length={args.max_length}  "
          f"epochs={args.epochs}  batch_size={args.batch_size}")
    print(f"  lr={args.learning_rate}  train_split={args.train_split}  seed={args.seed}")
    print(f"  model=xlm-roberta-base  device={device}  "
          f"class_weights={not args.no_class_weights}")

    # ── 1. Extract the first N sentences into a subset XML, then parse it ──────
    subset_xml = f"data/corpus_extracts/VUAMC_first_{args.num_sentences}_sentences.xml"
    print(f"\n[1/4] Extracting first {args.num_sentences} sentences -> {subset_xml}")
    extract_xml_subset(args.source_xml, subset_xml, args.num_sentences)
    sentences, words_only, word_labels = extract_vuamc_data(subset_xml)
    print(f"      parsed {len(sentences)} sentences")

    # ── 2. Train / validation split ───────────────────────────────────────────
    train_size = int(args.train_split * len(sentences))
    tr_s, tr_w, tr_l = sentences[:train_size], words_only[:train_size], word_labels[:train_size]
    va_s, va_w, va_l = sentences[train_size:], words_only[train_size:], word_labels[train_size:]
    print(f"\n[2/4] Split: {len(tr_s)} train / {len(va_s)} val sentences")

    # ── 3. Tokenizer, model, loaders ──────────────────────────────────────────
    print(f"\n[3/4] Loading 'xlm-roberta-base' and building data loaders...")
    tokenizer = initialize_xlm_roberta_tokenizer()
    model     = initialize_xlm_roberta_model_for_classification(num_labels=2)

    class_weights = None if args.no_class_weights else calculate_class_weights(tr_l)

    train_loader, val_loader = create_data_loaders(
        train_sentences=tr_s, train_words_only=tr_w, train_word_labels=tr_l,
        val_sentences=va_s,  val_words_only=va_w,  val_word_labels=va_l,
        tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length,
        label_alignment_func=align_labels_with_tokens_using_wordids,
    )

    # ── 4. Train, then validate with full metrics ─────────────────────────────
    save_path = None
    if args.save_model:
        save_path = os.path.join('models', args.save_model)
        os.makedirs('models', exist_ok=True)

    print(f"\n[4/4] Training...")
    train_model(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=args.epochs, learning_rate=args.learning_rate,
        class_weights=class_weights, save_path=save_path,
    )

    # Evaluate the best checkpoint if it was saved, else the current model state.
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        print(f"\nLoaded best model across all training epochs from {save_path} for final evaluation.")

    print("\nEvaluating on the validation set...")
    metrics = evaluate(model, val_loader, device)
    print_report(metrics)


if __name__ == "__main__":
    main()
