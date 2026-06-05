import pandas as pd
import pytest

from src.evaluation import (
    compute_classification_metrics,
    summarize_cross_validation,
    threshold_metrics_table,
    validate_probabilities,
)
from src.features import FeatureColumns
from src.modeling import create_logistic_regression_pipeline


def test_classification_metrics_return_expected_keys() -> None:
    metrics = compute_classification_metrics(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0, 1, 0, 0]),
        pd.Series([0.1, 0.8, 0.2, 0.4]),
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }


def test_cross_validation_summary_returns_expected_metrics() -> None:
    features = pd.DataFrame(
        {
            "age": [40, 50, 60, 70, 45, 55, 65, 75, 42, 52],
            "cp": [1, 2, 1, 3, 2, 3, 1, 2, 3, 1],
        }
    )
    target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    columns = FeatureColumns(numeric=["age"], categorical=["cp"])
    model = create_logistic_regression_pipeline(columns)

    summary = summarize_cross_validation(
        model, features, target, n_splits=2, random_state=42
    )

    assert set(summary["metric"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    }


def test_threshold_metric_table_has_one_row_per_threshold() -> None:
    thresholds = [0.25, 0.5, 0.75]

    table = threshold_metrics_table(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0.1, 0.8, 0.4, 0.6]),
        thresholds,
    )

    assert len(table) == len(thresholds)
    assert table["threshold"].tolist() == thresholds


def test_probability_validation_returns_copy() -> None:
    probabilities = pd.Series([0.2, 0.8])

    validated = validate_probabilities(probabilities)
    validated[0] = 0.5

    assert probabilities.tolist() == [0.2, 0.8]


def test_probability_validation_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_probabilities(pd.Series([0.2, 1.1]))
