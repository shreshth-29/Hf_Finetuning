"""Evaluation metric passed to the Hugging Face Trainer.

Mirrors the compute_metrics helper in notebooks/bert_finetuning.ipynb.
The module is named metrics.py rather than evaluate.py so it does not shadow
the `evaluate` library it depends on.
"""

import evaluate
import numpy as np


def build_compute_metrics(metric_name="accuracy"):
    metric = evaluate.load(metric_name)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics
