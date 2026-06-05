"""Data loading and preprocessing helpers for clinical risk modeling.

These utilities intentionally avoid downloading data. They operate on local,
permitted files only.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd

HEART_DISEASE_COLUMNS: list[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def load_csv(path: str | Path, **read_csv_kwargs: object) -> pd.DataFrame:
    """Load a local CSV file with a clear error if the file is missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, **read_csv_kwargs)


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lower-case, snake_case column names."""
    cleaned = data.copy()
    cleaned.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in cleaned.columns
    ]
    return cleaned


def drop_columns(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop columns if present, leaving the input DataFrame unchanged."""
    return data.drop(columns=[col for col in columns if col in data.columns]).copy()


def split_features_target(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into predictors and target."""
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}")

    features = data.drop(columns=[target_column]).copy()
    target = data[target_column].copy()
    return features, target


def validate_expected_columns(
    data: pd.DataFrame, expected_columns: Iterable[str]
) -> None:
    """Raise an error if a DataFrame is missing expected columns."""
    missing_columns = [column for column in expected_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing expected columns: {missing}")


def clean_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Replace UCI missing-value markers with pandas missing values."""
    return data.replace("?", pd.NA).copy()


def load_heart_disease_data(path: str | Path) -> pd.DataFrame:
    """Load the local UCI processed Cleveland Heart Disease file.

    The expected raw file is not downloaded by this project. Place the official
    UCI file at data/raw/processed.cleveland.data before calling this helper.
    """
    data = load_csv(
        path,
        header=None,
        names=HEART_DISEASE_COLUMNS,
        na_values="?",
    )
    validate_expected_columns(data, HEART_DISEASE_COLUMNS)
    return data


def binarize_heart_disease_target(
    data: pd.DataFrame, target_column: str = "target"
) -> pd.DataFrame:
    """Convert the original target coding to 0 for 0 and 1 for values above 0."""
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}")

    cleaned = data.copy()
    cleaned[target_column] = (cleaned[target_column] > 0).astype(int)
    return cleaned
