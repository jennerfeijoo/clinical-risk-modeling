import pandas as pd

from src.data_processing import split_features_target, standardize_column_names


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
