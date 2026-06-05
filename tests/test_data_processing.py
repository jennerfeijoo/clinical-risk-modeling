import pandas as pd
import pytest

from src.data_processing import (
    HEART_DISEASE_COLUMNS,
    add_binary_target_column,
    binarize_heart_disease_target,
    clean_missing_values,
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


def test_add_binary_target_column_preserves_original_target() -> None:
    data = pd.DataFrame({"target": [0, 1, 2, 4]})

    result = add_binary_target_column(data)

    assert result["target"].tolist() == [0, 1, 2, 4]
    assert result["target_binary"].tolist() == [0, 1, 1, 1]


def test_add_binary_target_column_does_not_mutate_input() -> None:
    data = pd.DataFrame({"target": [0, 2], "age": [50, 60]})
    original = data.copy(deep=True)

    add_binary_target_column(data)

    pd.testing.assert_frame_equal(data, original)


def test_add_binary_target_column_raises_when_target_is_missing() -> None:
    data = pd.DataFrame({"age": [50, 60]})

    with pytest.raises(ValueError, match="Target column not found"):
        add_binary_target_column(data)


def test_add_binary_target_column_raises_when_target_values_are_missing() -> None:
    data = pd.DataFrame({"target": [0, pd.NA]})

    with pytest.raises(ValueError, match="contains missing values"):
        add_binary_target_column(data)


def test_binarize_heart_disease_target_remains_backward_compatible() -> None:
    data = pd.DataFrame({"target": [0, 1, 2, 3, 4]})

    cleaned = binarize_heart_disease_target(data)

    assert cleaned["target"].tolist() == [0, 1, 1, 1, 1]


def test_clean_missing_values_does_not_mutate_input_dataframe() -> None:
    data = pd.DataFrame({"ca": ["?", "1.0"]})
    original = data.copy(deep=True)

    cleaned = clean_missing_values(data)

    pd.testing.assert_frame_equal(data, original)
    assert cleaned["ca"].isna().sum() == 1
