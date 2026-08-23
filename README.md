# GloSeGAT

GloSeGAT is the reference implementation for **Gloss-Sense Graph Attention
Networks for Multilingual Word-in-Context Disambiguation**. The repository
contains the training code, Colab entry points, raw experiment outputs, and
post-processing artifacts used to evaluate mBERT-based Word-in-Context (WiC)
models on the MCL-WiC benchmark.

The main model, **GloSeGAT**, augments contextual target-word representations
with gloss and sense-graph information retrieved from WordNet and Open
Multilingual WordNet. The repository also includes the sentence-pair and
target-centered baselines used in the paper.

## Repository Contents

```text
.
|-- Bert-base/
|   |-- mcl_wic_colab_common.py       # Shared data loading, models, training, evaluation, and CLI logic
|   |-- colab_bert_cls.py             # PairCLS baseline entry point
|   |-- colab_bert_marked.py          # MarkCLS baseline entry point
|   |-- colab_glowic_target.py        # TargetSim baseline entry point
|   |-- colab_glowic_target_pos.py    # TargetPOS baseline entry point
|   |-- colab_glowic_gat.py           # GloSeGAT entry point
|   |-- *_colab.ipynb                 # Google Colab notebooks for the same variants
|   `-- Results/                      # Per-variant Optuna trials, seed runs, and summaries
|-- analysis_outputs_mbert/           # Consolidated tables, reports, and critical-difference diagrams
|-- consolidate_mbert_results.ipynb   # Notebook for consolidating mBERT experiment outputs
|-- consolidate_mcl_wic_results.ipynb # Notebook for additional MCL-WiC result consolidation
`-- glosegat_mbert_analysis.ipynb     # Statistical analysis and reporting notebook
```

## Model Variants

The implementation supports five variants:

| Variant name | Script | Description |
| --- | --- | --- |
| `bert_cls` | `Bert-base/colab_bert_cls.py` | PairCLS baseline. Encodes both contexts jointly and classifies from the `[CLS]` representation. |
| `bert_marked` | `Bert-base/colab_bert_marked.py` | MarkCLS baseline. Inserts `[TGT]` and `[/TGT]` markers around both target occurrences before joint encoding. |
| `glowic_target` | `Bert-base/colab_glowic_target.py` | TargetSim baseline. Encodes contexts independently and compares the two contextual target vectors. |
| `glowic_target_pos` | `Bert-base/colab_glowic_target_pos.py` | TargetPOS baseline. Extends TargetSim with a part-of-speech embedding. |
| `glowic_gat` | `Bert-base/colab_glowic_gat.py` | Full GloSeGAT model. Adds candidate gloss embeddings and graph-refined sense representations. |

All variants use `bert-base-multilingual-cased` by default.

## Requirements

The code is written in Python and uses PyTorch, Hugging Face Transformers,
Optuna, scikit-learn, tqdm, NumPy, and NLTK.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers optuna scikit-learn tqdm numpy nltk
```

For GPU execution, install the PyTorch build that matches your CUDA runtime
from the official PyTorch installation instructions. The scripts also support
Apple Silicon through `mps` and CPU execution for small/debug runs.

## Dataset

Experiments use **MCL-WiC**, the SemEval-2021 multilingual and cross-lingual
Word-in-Context benchmark.

By default, the training code expects the dataset under:

```text
/content/mcl-wic/extracted
```

If the expected files are not present and data downloading is enabled, the code
clones the official MCL-WiC repository and extracts:

```text
SemEval-2021_MCL-WiC_all-datasets.zip
SemEval-2021_MCL-WiC_test-gold-data.zip
```

The code uses the official English-English training split, English-English
development split, and the following test splits:

```text
test.ar-ar
test.en-en
test.fr-fr
test.ru-ru
test.zh-zh
test.en-ar
test.en-fr
test.en-ru
test.en-zh
```

To use an existing local dataset, pass `--data-root` and optionally disable
automatic download:

```bash
python Bert-base/colab_glowic_gat.py \
  --data-root /path/to/mcl-wic/extracted \
  --no-download-data
```

## Running Experiments

Each entry-point script runs the full experimental protocol for one model
variant:

1. Load MCL-WiC train/dev/test splits.
2. Run Optuna hyperparameter search on the development split.
3. Retrain the selected configuration over multiple random seeds.
4. Evaluate each seed on the development split and all test splits.
5. Save per-seed JSON files, Optuna trials, and an aggregate summary.

Example full GloSeGAT run:

```bash
python Bert-base/colab_glowic_gat.py \
  --data-root /content/mcl-wic/extracted \
  --output-dir ./Bert-base/Results/gat \
  --device auto
```

Run the baselines:

```bash
python Bert-base/colab_bert_cls.py --output-dir ./Bert-base/Results/cls
python Bert-base/colab_bert_marked.py --output-dir ./Bert-base/Results/marked
python Bert-base/colab_glowic_target.py --output-dir ./Bert-base/Results/target
python Bert-base/colab_glowic_target_pos.py --output-dir ./Bert-base/Results/target_pos
```

For quick smoke tests, use smaller limits and fewer trials:

```bash
python Bert-base/colab_glowic_gat.py \
  --output-dir ./tmp/gat-smoke \
  --n-trials 1 \
  --n-seed-runs 1 \
  --max-epochs 1 \
  --train-limit 64 \
  --eval-limit 64 \
  --device auto
```

## Command-Line Arguments

The entry points share the same command-line interface:

| Argument | Default | Description |
| --- | --- | --- |
| `--data-root` | `/content/mcl-wic/extracted` | Directory containing the extracted MCL-WiC data. |
| `--output-dir` | `/content/mcl_wic_results/<variant>` | Directory where JSON outputs are written. |
| `--model-name` | `bert-base-multilingual-cased` | Hugging Face encoder name or local model path. |
| `--n-trials` | `15` | Number of Optuna hyperparameter-search trials. |
| `--n-seed-runs` | `15` | Number of independent final training/evaluation runs. |
| `--max-epochs` | `5` | Maximum epochs per trial or seed run. |
| `--patience` | `2` | Early-stopping patience measured on development accuracy. |
| `--max-len` | `128` | Maximum tokenized sequence length. |
| `--seed` | `42` | Optuna/search seed. |
| `--seed-start` | `1000` | First seed for final repeated runs. |
| `--train-limit` | unset | Optional cap for training examples, useful for debugging. |
| `--eval-limit` | unset | Optional cap for dev/test examples, useful for debugging. |
| `--num-workers` | `2` | PyTorch DataLoader worker count. |
| `--device` | `auto` | One of `auto`, `cuda`, `mps`, or `cpu`. |
| `--freeze-encoder` | disabled | Freeze the Transformer encoder and train only task-specific layers. |
| `--no-download-data` | disabled | Require data to already exist at `--data-root`. |

## Output Files

Each experiment output directory contains:

| File | Description |
| --- | --- |
| `optuna_trials.json` | Full hyperparameter-search record, best trial, and best hyperparameters. |
| `seed_<seed>.json` | Metrics, training history, and final evaluation for one repeated run. |
| `summary.json` | Best hyperparameters, all seed runs, and aggregate metrics by split. |

The consolidated analysis directory contains:

| File pattern | Description |
| --- | --- |
| `accuracy_by_dataset_*` | Accuracy tables in numeric, percentage, CSV, and LaTeX-ready formats. |
| `best_hyperparameters_*` | Best hyperparameters selected for each variant. |
| `seed_level_metrics.csv` | Per-seed metric table used for statistical analysis. |
| `split_metric_summary_long.csv` | Long-format metric summary by model and split. |
| `autorank_*_report.txt` | Statistical ranking reports. |
| `autorank_*_cd_diagram.png` | Critical-difference diagrams. |

## Reproduced mBERT Results

The consolidated mBERT accuracy table reports mean and standard deviation over
15 independent seeds:

| Model | En-En | Ar-Ar | Fr-Fr | Ru-Ru | Zh-Zh | En-Ar | En-Fr | En-Ru | En-Zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mBERT (PairCLS) | 81.34 +/- 0.67 | 73.89 +/- 0.61 | 76.53 +/- 1.22 | 70.85 +/- 0.96 | 69.84 +/- 1.22 | 55.51 +/- 1.73 | 65.90 +/- 2.38 | 61.74 +/- 2.06 | 55.31 +/- 1.57 |
| mBERT (MarkCLS) | 83.77 +/- 1.04 | 75.10 +/- 0.90 | 79.09 +/- 0.64 | 73.23 +/- 0.74 | 74.34 +/- 2.10 | 59.52 +/- 2.16 | 71.03 +/- 2.37 | 67.00 +/- 1.58 | 61.49 +/- 3.04 |
| mBERT (TargetSim) | 85.15 +/- 0.93 | 72.73 +/- 1.79 | 80.51 +/- 1.13 | 76.68 +/- 0.79 | 79.96 +/- 0.92 | 66.21 +/- 4.42 | 80.20 +/- 2.68 | 78.25 +/- 2.67 | 80.79 +/- 3.09 |
| mBERT (TargetPOS) | 84.65 +/- 1.52 | 74.15 +/- 1.21 | 80.51 +/- 1.04 | 77.20 +/- 0.85 | 79.67 +/- 1.07 | 67.60 +/- 2.62 | 80.43 +/- 1.18 | 78.40 +/- 1.51 | 80.97 +/- 2.27 |
| mBERT (GloSeGAT) | 84.89 +/- 0.72 | 72.77 +/- 1.13 | 78.18 +/- 1.00 | 76.40 +/- 0.92 | 78.53 +/- 1.14 | 72.11 +/- 2.87 | 82.47 +/- 1.50 | 81.14 +/- 1.29 | 84.47 +/- 2.03 |

GloSeGAT obtains the strongest results on all four cross-lingual test splits
in this table.

## Best Hyperparameters

The best Optuna-selected hyperparameters used for the repeated mBERT runs are:

| Variant | Batch size | Dropout | Encoder LR | Head LR | Warmup ratio | Weight decay | Top-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PairCLS | 16 | 0.2487566853 | 1.0569064414e-05 | 4.0577978380e-04 | 0.0820065419 | 0.0258779982 | - |
| MarkCLS | 8 | 0.1475990992 | 1.1530645081e-05 | 7.8515041894e-05 | 0.1243106264 | 0.0045227289 | - |
| TargetSim | 4 | 0.3227961206 | 1.2170293884e-05 | 1.5636765184e-04 | 0.0780102032 | 0.0034388521 | - |
| TargetPOS | 8 | 0.2925192044 | 1.1103735609e-05 | 4.4448339535e-04 | 0.0660228741 | 0.0965632033 | - |
| GloSeGAT | 8 | 0.1087948587 | 2.6176655097e-05 | 4.1768053777e-04 | 0.0407023548 | 0.0088492502 | 7 |

## Implementation Notes

GloSeGAT retrieves candidate synsets using the target lemma, part of speech,
and language metadata. Language codes are mapped to Open Multilingual WordNet
codes for English, French, Russian, Chinese, and Arabic. If no candidate is
found for the split language, the implementation falls back to English.

Candidate glosses are encoded with the same Transformer encoder used for the
contexts. A small graph-attention layer refines the candidate sense vectors
using WordNet relations and shared lexicographer file information. The final
classifier combines contextual target features, gloss-attended features,
graph-attended features, element-wise comparisons, and cosine similarities.

## Citation

If you use this code, cite the accompanying paper:

Final citation metadata will be added after publication.
