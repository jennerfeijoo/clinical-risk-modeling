import numpy as np
import pandas as pd

from src.features import (
    HEART_DISEASE_CATEGORICAL_FEATURES,
    HEART_DISEASE_NUMERIC_FEATURES,
    FeatureColumns,
    build_preprocessor,
    define_heart_disease_feature_columns,
    feature_column_names,
)


def test_heart_disease_feature_lists_are_explicit_and_non_overlapping() -> None:
    columns = define_heart_disease_feature_columns()

    assert tuple(columns.numeric) == HEART_DISEASE_NUMERIC_FEATURES
    assert tuple(columns.categorical) == HEART_DISEASE_CATEGORICAL_FEATURES
    assert set(columns.numeric).isdisjoint(columns.categorical)
    assert len(feature_column_names(columns)) == 13


def test_preprocessor_fits_synthetic_dataframe_with_missing_values() -> None:
    data = pd.DataFrame(
        {
            "age": [40.0, 50.0, np.nan, 70.0],
            "cp": [1.0, 2.0, 1.0, np.nan],
        }
    )
    columns = FeatureColumns(numeric=["age"], categorical=["cp"])

    transformed = build_preprocessor(columns).fit_transform(data)

    assert transformed.shape[0] == len(data)
