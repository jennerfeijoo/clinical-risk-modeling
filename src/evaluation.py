"""Evaluation helpers for binary clinical risk models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
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

CLASSIFICATION_SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def validate_probabilities(
    probabilities: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return validated one-dimensional probabilities without mutating input."""
    values = np.asarray(probabilities, dtype=float).copy()
    if values.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional.")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities must contain only finite values.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")
    return values


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return common discrimination, calibration, and threshold metrics."""
    probabilities = validate_probabilities(y_probability)
    if len(y_true) != len(probabilities):
        raise ValueError("Targets and probabilities must have equal lengths.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    y_predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_predicted, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
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
    probabilities = validate_probabilities(y_probability)
    if len(y_true) != len(y_predicted) or len(y_true) != len(probabilities):
        raise ValueError("Targets, predictions, and probabilities must have equal lengths.")

    return {
        "accuracy": accuracy_score(y_true, y_predicted),
        "precision": precision_score(y_true, y_predicted, zero_division=0),
        "recall": recall_score(y_true, y_predicted, zero_division=0),
        "f1": f1_score(y_true, y_predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
    }


def summarize_cross_validation(
    estimator: object,
    features: pd.DataFrame,
    target: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Summarize stratified cross-validation metrics on training data."""
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    scores = cross_validate(
        estimator,
        features,
        target,
        cv=splitter,
        scoring=CLASSIFICATION_SCORING,
        return_train_score=False,
    )
    rows = [
        {
            "metric": metric,
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"], ddof=1)),
        }
        for metric in CLASSIFICATION_SCORING
    ]
    return pd.DataFrame(rows)


def threshold_metrics_table(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    thresholds: list[float] | tuple[float, ...],
) -> pd.DataFrame:
    """Compute classification metrics across probability thresholds."""
    probabilities = validate_probabilities(y_probability)
    if len(y_true) != len(probabilities):
        raise ValueError("Targets and probabilities must have equal lengths.")

    rows = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be between 0 and 1.")
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(y_true, predictions),
                "precision": precision_score(
                    y_true, predictions, zero_division=0
                ),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "specificity": specificity,
                "f1": f1_score(y_true, predictions, zero_division=0),
                "predicted_positive_rate": float(np.mean(predictions)),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )
    return pd.DataFrame(rows)


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
    title: str = "Logistic Regression Confusion Matrix",
) -> None:
    """Save a confusion matrix plot."""
    display = ConfusionMatrixDisplay.from_predictions(y_true, y_predicted)
    display.ax_.set_title(title)
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


def plot_cross_validation_metrics(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save cross-validation means with one-standard-deviation error bars."""
    display_summary = summary.copy()
    display_summary["metric"] = display_summary["metric"].replace(
        {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-score",
            "roc_auc": "ROC-AUC",
            "average_precision": "PR-AUC",
        }
    )
    ax = display_summary.plot.bar(
        x="metric",
        y="mean",
        yerr="std",
        legend=False,
        capsize=4,
        color="#4c78a8",
        figsize=(9, 5),
    )
    ax.set_title("Training-Set Stratified Cross-Validation")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_assessment(
    y_true: pd.Series,
    y_probability: pd.Series | np.ndarray,
    output_path: str | Path,
    n_bins: int = 8,
) -> None:
    """Save a held-out calibration curve with a perfect-calibration reference."""
    probabilities = validate_probabilities(y_probability)
    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=n_bins,
        strategy="quantile",
    )
    _, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        color="#4c78a8",
        label="Logistic regression",
    )
    ax.set_title("Held-Out Test Calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_threshold_metrics(
    threshold_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save selected threshold-dependent metrics."""
    ax = threshold_table.plot(
        x="threshold",
        y=["precision", "recall", "specificity", "f1"],
        marker="o",
        figsize=(8, 5),
    )
    ax.set_title("Held-Out Test Metrics Across Thresholds")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
