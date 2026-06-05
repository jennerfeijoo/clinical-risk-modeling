import pandas as pd

from src.features import FeatureColumns
from src.modeling import (
    create_dummy_classifier,
    create_logistic_regression_pipeline,
    split_train_test,
)


def test_train_test_split_keeps_matching_lengths() -> None:
    features = pd.DataFrame({"age": [40, 50, 60, 70, 45, 55]})
    target = pd.Series([0, 1, 0, 1, 0, 1])

    x_train, x_test, y_train, y_test = split_train_test(
        features, target, test_size=0.5, random_state=42
    )

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) + len(x_test) == len(features)


def test_logistic_regression_pipeline_fits_tiny_dataframe() -> None:
    features = pd.DataFrame(
        {
            "age": [40, 50, 60, 70, 45, 55],
            "cp": [1, 2, 1, 3, 2, 3],
        }
    )
    target = pd.Series([0, 1, 0, 1, 0, 1])
    columns = FeatureColumns(numeric=["age"], categorical=["cp"])

    model = create_logistic_regression_pipeline(columns)
    model.fit(features, target)

    assert len(model.predict(features)) == len(target)


def test_dummy_classifier_fits_without_raw_dataset() -> None:
    features = pd.DataFrame({"age": [40, 50, 60, 70]})
    target = pd.Series([0, 0, 0, 1])

    model = create_dummy_classifier().fit(features, target)

    assert model.predict(features).tolist() == [0, 0, 0, 0]
