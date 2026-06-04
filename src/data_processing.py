"""Data loading and preprocessing helpers for clinical risk modeling.

These utilities intentionally avoid downloading data. They operate on local,
permitted files only.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd


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
