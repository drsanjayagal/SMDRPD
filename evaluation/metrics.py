"""Multiclass metrics for synthetic-data experiments."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }

    per_class_precision = precision_score(y_true, y_pred, average=None, labels=list(labels), zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, labels=list(labels), zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(labels), zero_division=0)

    support = confusion_matrix(y_true, y_pred, labels=list(labels))
    specificities = []
    for i in range(len(labels)):
        true_negative = support.sum() - support[i, :].sum() - support[:, i].sum() + support[i, i]
        false_positive = support[:, i].sum() - support[i, i]
        specificities.append(true_negative / (true_negative + false_positive + 1e-12))

    metrics["specificity_macro"] = float(np.mean(specificities))
    metrics["sensitivity_macro"] = float(np.mean(per_class_recall))
    metrics["per_class"] = {
        int(label): {
            "precision": float(per_class_precision[i]),
            "recall_sensitivity": float(per_class_recall[i]),
            "specificity": float(specificities[i]),
            "f1": float(per_class_f1[i]),
            "support": int(support[i, :].sum()),
        }
        for i, label in enumerate(labels)
    }

    if y_proba is not None:
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=list(labels))
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=list(labels)))
        except ValueError:
            metrics["log_loss"] = None
    else:
        metrics["roc_auc_ovr_macro"] = None
        metrics["log_loss"] = None

    return metrics


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start