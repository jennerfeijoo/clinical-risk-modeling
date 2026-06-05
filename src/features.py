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

HEART_DISEASE_NUMERIC_FEATURES: tuple[str, ...] = (
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
)

HEART_DISEASE_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
)


@dataclass(frozen=True)
class FeatureColumns:
    """Explicit numeric and categorical column groups for preprocessing."""

    numeric: list[str]
    categorical: list[str]


def define_heart_disease_feature_columns() -> FeatureColumns:
    """Return explicit feature groups for the processed Cleveland dataset.

    Continuous measurements are treated as numeric. Integer-coded categories
    are treated as categorical so their codes are not assumed to be linear.
    """
    return FeatureColumns(
        numeric=list(HEART_DISEASE_NUMERIC_FEATURES),
        categorical=list(HEART_DISEASE_CATEGORICAL_FEATURES),
    )


def infer_feature_columns(data: pd.DataFrame) -> FeatureColumns:
    """Infer feature groups from pandas dtypes for generic tabular data."""
    numeric = data.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in data.columns if column not in numeric]
    return FeatureColumns(numeric=numeric, categorical=categorical)


def feature_column_names(columns: FeatureColumns) -> list[str]:
    """Return numeric and categorical feature names in modeling order."""
    return columns.numeric + columns.categorical


def build_preprocessor(columns: FeatureColumns) -> ColumnTransformer:
    """Build leakage-aware preprocessing for numeric and categorical features."""
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
