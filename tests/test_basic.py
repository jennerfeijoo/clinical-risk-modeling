import pandas as pd
import pytest

from src.data_processing import (
    HEART_DISEASE_COLUMNS,
    binarize_heart_disease_target,
    split_features_target,
    standardize_column_names,
    validate_expected_columns,
)
from src.evaluation import (
    compute_classification_metrics,
    summarize_cross_validation,
    threshold_metrics_table,
    validate_probabilities,
)
from src.features import FeatureColumns
from src.modeling import create_logistic_regression_pipeline, split_train_test


def test_standardize_column_names_and_split_target() -> None:
    data = pd.DataFrame(
        {
            " Patient Age ": [55, 72],
            "Risk-Group": ["low", "high"],
            "Outcome": [0, 1],
        }
    )

    cleaned = standardize_column_names(data)
    features, target = split_features_target(cleaned, "outcome")

    assert cleaned.columns.tolist() == ["patient_age", "risk_group", "outcome"]
    assert features.columns.tolist() == ["patient_age", "risk_group"]
    assert target.tolist() == [0, 1]


def test_heart_disease_columns_has_14_columns() -> None:
    assert len(HEART_DISEASE_COLUMNS) == 14


def test_validate_expected_columns_passes_with_expected_columns() -> None:
    data = pd.DataFrame(columns=HEART_DISEASE_COLUMNS)

    validate_expected_columns(data, HEART_DISEASE_COLUMNS)


def test_validate_expected_columns_fails_when_columns_are_missing() -> None:
    data = pd.DataFrame(columns=HEART_DISEASE_COLUMNS[:-1])

    with pytest.raises(ValueError, match="target"):
        validate_expected_columns(data, HEART_DISEASE_COLUMNS)


def test_binarize_heart_disease_target_maps_positive_values_to_one() -> None:
    data = pd.DataFrame({"target": [0, 1, 2, 3, 4]})

    cleaned = binarize_heart_disease_target(data)

    assert cleaned["target"].tolist() == [0, 1, 1, 1, 1]


def test_train_test_split_keeps_matching_lengths() -> None:
    features = pd.DataFrame({"age": [40, 50, 60, 70, 45, 55]})
    target = pd.Series([0, 1, 0, 1, 0, 1])

    x_train, x_test, y_train, y_test = split_train_test(
        features, target, test_size=0.5, random_state=42
    )

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == len(features)


def test_logistic_regression_pipeline_fits_tiny_dataframe() -> None:
    features = pd.DataFrame(
        {
            "age": [40, 50, 60, 70, 45, 55],
            "cp": [1, 2, 1, 3, 2, 3],
        }
    )
    target = pd.Series([0, 1, 0, 1, 0, 1])
    columns = FeatureColumns(numeric=["age"], categorical=["cp"])

    model = create_logistic_regression_pipeline(columns)
    model.fit(features, target)

    assert len(model.predict(features)) == len(target)


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


def test_probability_validation_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_probabilities(pd.Series([0.2, 1.1]))


def test_binarize_target_does_not_mutate_input_dataframe() -> None:
    data = pd.DataFrame({"target": [0, 2], "age": [50, 60]})
    original = data.copy(deep=True)

    binarize_heart_disease_target(data)

    pd.testing.assert_frame_equal(data, original)
