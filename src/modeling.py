"""Baseline modeling helpers for clinical risk prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.features import FeatureColumns, build_preprocessor


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split for binary classification."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def create_preprocessing_pipeline(columns: FeatureColumns):
    """Create the preprocessing pipeline used by baseline classifiers."""
    return build_preprocessor(columns)


def build_logistic_regression_model(columns: FeatureColumns) -> Pipeline:
    """Build an interpretable baseline classifier.

    Logistic regression is a reasonable first model for many binary clinical
    risk prediction tasks. More complex models should be added only after this
    baseline is documented and evaluated.
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
                    random_state=42,
                ),
            ),
        ]
    )


def create_logistic_regression_pipeline(columns: FeatureColumns) -> Pipeline:
    """Build the Phase 4 baseline logistic regression pipeline."""
    return build_logistic_regression_model(columns)


def create_dummy_classifier(random_state: int = 42) -> DummyClassifier:
    """Build a simple majority-class baseline classifier."""
    return DummyClassifier(strategy="most_frequent", random_state=random_state)
