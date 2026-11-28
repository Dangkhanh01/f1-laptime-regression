"""F1-06 event and lap-number OLS baseline."""

from functools import partial

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _observed_column(frame: pd.DataFrame, name: str) -> list[str]:
    """ColumnTransformer calls its selectors during fit and saves the selection."""
    return [name] if frame[name].notna().any() else []


def build_m1(config: dict) -> Pipeline:
    if config["features"] != ["EventName", "LapNumber"] or config["scaler"] != "StandardScaler":
        raise ValueError("M1 expects EventName, LapNumber and StandardScaler")
    numeric = make_pipeline(SimpleImputer(**config["numeric_imputer"]), StandardScaler())
    event = make_pipeline(
        SimpleImputer(missing_values=pd.NA, **config["categorical_imputer"]),
        OneHotEncoder(**config["encoder"]),
    )
    preprocess = ColumnTransformer(
        [
            ("lap", numeric, partial(_observed_column, name="LapNumber")),
            ("event", event, partial(_observed_column, name="EventName")),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("ols", LinearRegression(**config["estimator"]))])
