"""Evaluation helpers for binary clinical risk models."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_probability: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return common discrimination, calibration, and threshold metrics."""
    y_predicted = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(y_true, y_probability),
        "brier_score": brier_score_loss(y_true, y_probability),
        "precision": precision_score(y_true, y_predicted, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_predicted, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_predicted, zero_division=0),
    }
