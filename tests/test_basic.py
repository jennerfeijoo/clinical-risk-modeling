import pandas as pd
import pytest

from src.data_processing import (
    HEART_DISEASE_COLUMNS,
    binarize_heart_disease_target,
    split_features_target,
    standardize_column_names,
    validate_expected_columns,
)


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
