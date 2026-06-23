# VUAMC Metaphor Detection

Token-level **metaphor detection**: Given a sentence, classify each word (via tokens) as
*literal* (`0`) or *metaphorically used* (`1`). The model is a fine-tuned
[`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) token classifier
trained on (a subset of) the VU Amsterdam Metaphor Corpus (VUAMC).

## Approach

- **Data Extraction** (`src/extractors.py`) parses the VUAMC TEI-XML, rebuilding
  each sentence's surface text while reading per-word metaphor labels by looking for
  `<seg function="mrw" type="met">` annotations. Handles BNC quirks like excess whitespaces,
  opening quotes, possessives (`'s`), negation clitics (`do` + `n't` → `don't`)
  and multi-word units, so that the reconstructed text stays mostly consistent with expected sentence formatting and the labels stay aligned.
- **Label Alignment** (`src/label_alignment.py`) maps each word label onto the
  *first sub-word* produced by the tokenizer; continuation sub-words, punctuation
  and special tokens are masked with `-100` so they don't contribute to the loss.
- **Training** (`src/train_metaphor_detector.py`) uses class-weighted
  cross-entropy (metaphors are often the minority class) with AdamW and a linear
  warmup schedule. The best model (by validation loss) across training epochs is saved.
- **Evaluation** reports a confusion matrix plus precision, recall and F1.

## Setup

```bash
pip install -r requirements.txt
```

The VUAMC corpus is included in this repository at `data/VUAMC/VUAMC.xml`.

## Usage

Everything runs through `run.py`, which extracts the data, trains, validates and
prints metrics:

```bash
# Defaults: 500 sentences, context window 128, 3 epochs
python run.py

# Customise the run
python run.py --num-sentences 1000 --max-length 128 --epochs 5 --batch-size 16 --lr 3e-5
```

Key options (`python run.py --help` for all):

| Flag | Default | Meaning |
|------|---------|---------|
| `-n`, `--num-sentences` | 500 | Sentences to extract from the VUAMC corpus |
| `--max-length` | 128 | Token context window |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--train-split` | 0.8 | Train% of data in train:validation split |
| `--seed` | 42 | Reproducibility |
| `--no-class-weights` | off | Disable the imbalance-correcting weighted loss |
| `--save-model FILENAME` | metaphor_model.bin | Filename for best model (by validation loss) in models/ directory. Set to '' to skip saving. |
