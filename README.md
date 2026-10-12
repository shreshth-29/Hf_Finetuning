# BERT Fine-tuning Pipeline

Repository created in September 2024. Reorganized and cleaned up in October 2026.

Fine-tunes a pretrained transformer for 3-class text classification using the Hugging Face `Trainer` API. The project started as a Kaggle notebook (September 2024) and the same pipeline is also provided as a small Python package so it can be run from the command line.

## What it does

1. Loads a train CSV and a validation CSV (columns `text` and `label`) with `datasets.load_dataset("csv", ...)`.
2. Tokenizes with the `google-bert/bert-base-cased` tokenizer (max length padding, truncation).
3. Shuffles both splits with seed 42 and selects 6187 training rows and 2063 validation rows (the full splits).
4. Loads `wu981526092/Sentence-Level-Stereotype-Detector` with `AutoModelForSequenceClassification`, replacing its 9-way classification head with a freshly initialized 3-way head (`num_labels=3`, `ignore_mismatched_sizes=True`).
5. Trains with `Trainer`, evaluating accuracy on the validation split after every epoch.
6. Pushes the final model to the Hugging Face Hub.

## Dataset

A private Kaggle dataset named `finetuning` with two files, `responses_train.csv` (6187 rows) and `responses_val.csv` (2063 rows). Each row has a `text` field and an integer `label` in {0, 1, 2}. The notebook output directory is `age_gender_ft_1`, so the labels relate to age and gender stereotype classes. The dataset is not included in this repository.

## Model and training setup

| Item | Value |
| --- | --- |
| Base checkpoint | `wu981526092/Sentence-Level-Stereotype-Detector` (a DistilBERT sequence classifier) |
| Tokenizer | `google-bert/bert-base-cased` |
| Classification head | 3 labels, randomly initialized |
| Trainer | `transformers.Trainer` with `TrainingArguments` |
| Epochs | 10 |
| Learning rate | 5e-5 (AdamW) |
| Batch size | `Trainer` default, 8 per device (the Kaggle run used 2 x T4, 16 per step, 387 steps per epoch) |
| Eval strategy | every epoch |
| Metric | accuracy (`evaluate.load("accuracy")`) |
| Hardware | Kaggle, NVIDIA T4 |

## Results

Recorded in the notebook outputs for the Kaggle run (3870 steps, about 31 minutes):

| Epoch | Training loss | Validation loss | Accuracy |
| --- | --- | --- | --- |
| 1 | no log | 0.2058 | 0.9501 |
| 2 | 0.2945 | 0.1557 | 0.9632 |
| 3 | 0.1004 | 0.1389 | 0.9714 |
| 4 | 0.0681 | 0.1481 | 0.9695 |
| 5 | 0.0681 | 0.1211 | 0.9733 |
| 6 | 0.0361 | 0.1796 | 0.9729 |
| 7 | 0.0160 | 0.1419 | 0.9782 |
| 8 | 0.0070 | 0.1687 | 0.9762 |
| 9 | 0.0070 | 0.1721 | 0.9777 |
| 10 | 0.0017 | 0.1719 | 0.9782 |

Final validation accuracy after 10 epochs: 0.9782. Average training loss over the run: 0.0677.

## Repository layout

```
.
├── notebooks/
│   └── bert_finetuning.ipynb   # original Kaggle notebook
├── src/
│   └── finetune/
│       ├── __init__.py
│       ├── data.py             # CSV loading, tokenization, shuffling and subsampling
│       ├── metrics.py          # accuracy compute_metrics for Trainer
│       └── train.py            # model, TrainingArguments, Trainer, CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

### Notebook (Kaggle)

1. Upload the notebook to Kaggle and attach the dataset so the files sit at `/kaggle/input/finetuning/responses_train.csv` and `/kaggle/input/finetuning/responses_val.csv`.
2. Enable a GPU accelerator (the original run used a T4) and internet access.
3. Run all cells. The `notebook_login()` cell asks for a Hugging Face token because `push_to_hub=True`. The training cell asks for a Weights and Biases key if `wandb` is installed; you can skip that prompt by setting `report_to="none"` in `TrainingArguments`.

### Script

```bash
pip install -r requirements.txt
cd src
python -m finetune.train \
    --train-file ../data/responses_train.csv \
    --val-file ../data/responses_val.csv \
    --model-name wu981526092/Sentence-Level-Stereotype-Detector \
    --num-labels 3 \
    --epochs 10 \
    --lr 5e-5 \
    --batch-size 8 \
    --output-dir ../outputs/age_gender_ft_1
```

Add `--push-to-hub` to upload the model like the notebook does (run `huggingface-cli login` first). Use `--num-train-samples` and `--num-eval-samples` to train on a subset. Checkpoints go to `--output-dir`, and the final model is saved there as well. Per-epoch validation accuracy is printed by the `Trainer` during training.

## Note on the two versions

`notebooks/bert_finetuning.ipynb` is the original Kaggle notebook, kept as it was run (API tokens that were pasted into comments have been redacted). `src/finetune/` is the same pipeline rewritten as a script with the hyperparameters exposed as command line arguments. No modelling changes were made in the rewrite.
