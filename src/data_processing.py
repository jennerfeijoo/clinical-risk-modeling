"""Data loading and preprocessing helpers for clinical risk modeling.

These utilities intentionally avoid downloading data. They operate on local,
permitted files only.
"""

from pathlib import Path
from typing import Any, Iterable

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


def load_csv(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a local CSV file without modifying or downloading source data."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, **read_csv_kwargs)


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with trimmed, lower-case, snake_case column names."""
    cleaned = data.copy()
    cleaned.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in cleaned.columns
    ]
    return cleaned


def drop_columns(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy with the requested columns removed when present."""
    return data.drop(columns=[col for col in columns if col in data.columns]).copy()


def split_features_target(
    data: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return independent copies of predictors and the selected target."""
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}")

    features = data.drop(columns=[target_column]).copy()
    target = data[target_column].copy()
    return features, target


def validate_expected_columns(
    data: pd.DataFrame, expected_columns: Iterable[str]
) -> None:
    """Raise ``ValueError`` when any required column is absent."""
    missing_columns = [column for column in expected_columns if column not in data.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing expected columns: {missing}")


def clean_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with UCI ``?`` markers replaced by pandas missing values."""
    return data.replace("?", pd.NA).copy()


def load_heart_disease_data(path: str | Path) -> pd.DataFrame:
    """Load the local UCI processed Cleveland file with documented columns.

    The helper never downloads data. The official file must be placed at
    ``data/raw/processed.cleveland.data`` by the user.
    """
    data = load_csv(
        path,
        header=None,
        names=HEART_DISEASE_COLUMNS,
        na_values="?",
    )
    validate_expected_columns(data, HEART_DISEASE_COLUMNS)
    return data


def _validated_target_values(
    data: pd.DataFrame,
    target_column: str,
) -> pd.Series:
    """Return a validated numeric target series for binary recoding."""
    if target_column not in data.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if data[target_column].isna().any():
        raise ValueError(f"Target column contains missing values: {target_column}")

    try:
        target = pd.to_numeric(data[target_column], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Target values must be numeric.") from error

    if (target < 0).any():
        raise ValueError("Target values must be zero or positive.")
    return target


def add_binary_target_column(
    data: pd.DataFrame,
    target_column: str = "target",
    output_column: str = "target_binary",
) -> pd.DataFrame:
    """Return a copy with a binary target while preserving the original target.

    Original target values equal to ``0`` map to ``0``. Values greater than
    ``0`` map to ``1``.
    """
    if output_column == target_column:
        raise ValueError("Output column must differ from the original target column.")

    target = _validated_target_values(data, target_column)
    cleaned = data.copy()
    cleaned[output_column] = (target > 0).astype(int)
    return cleaned


def binarize_heart_disease_target(
    data: pd.DataFrame, target_column: str = "target"
) -> pd.DataFrame:
    """Return a copy with the selected target replaced by its binary coding.

    This compatibility helper preserves the original API. New analysis code
    should prefer :func:`add_binary_target_column` so the source target remains
    available for documentation and audit.
    """
    target = _validated_target_values(data, target_column)
    cleaned = data.copy()
    cleaned[target_column] = (target > 0).astype(int)
    return cleaned
