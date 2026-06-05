"""Baseline modeling helpers for clinical risk prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features import FeatureColumns, build_preprocessor


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified split while preserving X/y alignment."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def create_preprocessing_pipeline(columns: FeatureColumns) -> ColumnTransformer:
    """Create the preprocessing transformer used by baseline classifiers."""
    return build_preprocessor(columns)


def build_logistic_regression_model(
    columns: FeatureColumns,
    random_state: int = 42,
) -> Pipeline:
    """Build an interpretable, leakage-aware logistic regression pipeline.

    Imputation, scaling, and encoding are fitted inside the pipeline so they can
    be learned independently within training folds.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(columns)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )


def create_logistic_regression_pipeline(
    columns: FeatureColumns,
    random_state: int = 42,
) -> Pipeline:
    """Build the reusable baseline logistic regression pipeline."""
    return build_logistic_regression_model(columns, random_state=random_state)


def create_dummy_classifier(random_state: int = 42) -> DummyClassifier:
    """Build a majority-class comparator for contextualizing model metrics."""
    return DummyClassifier(strategy="most_frequent", random_state=random_state)
