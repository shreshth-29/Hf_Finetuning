"""Fine-tune a sequence classification model with the Hugging Face Trainer.

Script version of notebooks/bert_finetuning.ipynb. Defaults match the notebook.

Example:
    python -m finetune.train \
        --train-file data/responses_train.csv \
        --val-file data/responses_val.csv \
        --output-dir outputs/age_gender_ft_1
"""

import argparse

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from finetune.data import DEFAULT_TOKENIZER, load_and_tokenize, load_tokenizer
from finetune.metrics import build_compute_metrics

DEFAULT_MODEL = "wu981526092/Sentence-Level-Stereotype-Detector"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face sequence classifier.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
                        help="Hugging Face model id or local path for the base checkpoint.")
    parser.add_argument("--tokenizer-name", default=DEFAULT_TOKENIZER,
                        help="Tokenizer id. The notebook uses google-bert/bert-base-cased.")
    parser.add_argument("--train-file", required=True, help="Path to the training CSV (columns: text, label).")
    parser.add_argument("--val-file", required=True, help="Path to the validation CSV (columns: text, label).")
    parser.add_argument("--num-labels", type=int, default=3, help="Number of classes.")
    parser.add_argument("--epochs", type=float, default=10, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for AdamW.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device train and eval batch size.")
    parser.add_argument("--output-dir", default="outputs/age_gender_ft_1", help="Where checkpoints are written.")
    parser.add_argument("--num-train-samples", type=int, default=None,
                        help="Use only the first N shuffled training rows. Default: all rows.")
    parser.add_argument("--num-eval-samples", type=int, default=None,
                        help="Use only the first N shuffled validation rows. Default: all rows.")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push checkpoints and the final model to the Hugging Face Hub (requires huggingface-cli login).")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_name)
    train_ds, eval_ds = load_and_tokenize(
        args.train_file,
        args.val_file,
        tokenizer,
        num_train_samples=args.num_train_samples,
        num_eval_samples=args.num_eval_samples,
    )

    # ignore_mismatched_sizes lets us replace a classification head of a different size.
    # The new head is randomly initialized and trained on the downstream task.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_labels,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=args.push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=build_compute_metrics("accuracy"),
    )

    train_result = trainer.train()
    print(train_result)

    eval_result = trainer.evaluate()
    print(eval_result)

    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
