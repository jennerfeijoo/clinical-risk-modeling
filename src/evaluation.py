"""Evaluation helpers for binary clinical risk models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
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


def compute_classification_metrics(
    y_true: pd.Series,
    y_predicted: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_predicted),
        "precision": precision_score(y_true, y_predicted, zero_division=0),
        "recall": recall_score(y_true, y_predicted, zero_division=0),
        "f1": f1_score(y_true, y_predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "average_precision": average_precision_score(y_true, y_probability),
    }


def classification_report_table(
    y_true: pd.Series,
    y_predicted: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Return sklearn's classification report as a DataFrame."""
    report = classification_report(
        y_true,
        y_predicted,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def plot_confusion_matrix(
    y_true: pd.Series,
    y_predicted: pd.Series | np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a confusion matrix plot."""
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_predicted)
    display.ax_.set_title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a ROC curve plot."""
    display = RocCurveDisplay.from_predictions(y_true, y_probability)
    display.ax_.set_title("Logistic Regression ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    output_path: str | Path,
) -> None:
    """Save a precision-recall curve plot."""
    display = PrecisionRecallDisplay.from_predictions(y_true, y_probability)
    display.ax_.set_title("Logistic Regression Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_logistic_coefficients(
    coefficients: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Save a bar plot of the largest absolute logistic regression coefficients."""
    coefficient_subset = (
        coefficients.assign(abs_coefficient=lambda data: data["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .head(top_n)
        .sort_values("coefficient")
    )
    ax = coefficient_subset.plot.barh(
        x="feature",
        y="coefficient",
        legend=False,
        figsize=(8, 6),
    )
    ax.set_title("Largest Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
