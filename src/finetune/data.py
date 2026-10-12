"""Dataset loading and tokenization.

Mirrors the "Load and Process Dataset" section of notebooks/bert_finetuning.ipynb.
The CSV files are expected to have a "text" column and a "label" column.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "google-bert/bert-base-cased"
SEED = 42


def load_tokenizer(name=DEFAULT_TOKENIZER):
    return AutoTokenizer.from_pretrained(name)


def load_and_tokenize(train_file, val_file, tokenizer, num_train_samples=None, num_eval_samples=None):
    """Load train/val CSVs, tokenize with max_length padding, shuffle and optionally subsample.

    Returns (train_dataset, eval_dataset).
    """
    data_files = {"train": train_file, "test": val_file}
    dataset = load_dataset("csv", data_files=data_files)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized = dataset.map(tokenize_function, batched=True)

    train_ds = tokenized["train"].shuffle(seed=SEED)
    eval_ds = tokenized["test"].shuffle(seed=SEED)

    if num_train_samples is not None:
        train_ds = train_ds.select(range(num_train_samples))
    if num_eval_samples is not None:
        eval_ds = eval_ds.select(range(num_eval_samples))

    return train_ds, eval_ds
