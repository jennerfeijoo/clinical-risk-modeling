"""Baseline modeling helpers for clinical risk prediction."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import FeatureColumns, build_preprocessor


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
