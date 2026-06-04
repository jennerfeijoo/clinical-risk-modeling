"""Feature preparation utilities.

Fit preprocessing objects on training data only, then reuse them on validation
or test data to reduce leakage risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureColumns:
    """Column groups used by the preprocessing pipeline."""

    numeric: list[str]
    categorical: list[str]


def infer_feature_columns(data: pd.DataFrame) -> FeatureColumns:
    """Infer numeric and categorical columns from pandas dtypes."""
    numeric = data.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in data.columns if column not in numeric]
    return FeatureColumns(numeric=numeric, categorical=categorical)


def build_preprocessor(columns: FeatureColumns) -> ColumnTransformer:
    """Build a baseline preprocessing pipeline for tabular clinical data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if columns.numeric:
        transformers.append(("numeric", numeric_pipeline, columns.numeric))
    if columns.categorical:
        transformers.append(("categorical", categorical_pipeline, columns.categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop")
