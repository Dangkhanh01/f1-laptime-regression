import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from f1_lap_time.m1 import build_m1


@pytest.fixture
def pipeline():
    config = Path(__file__).resolve().parents[1] / "configs/models/m1.json"
    return build_m1(json.loads(config.read_text("utf-8")))


def test_train_statistics_unseen_event_and_reload(pipeline, tmp_path):
    train = pd.DataFrame({"EventName": ["A", "B", pd.NA, "A"], "LapNumber": [2, 3, 7, np.nan]})
    pipeline.fit(train, [95, 80, 75, 90])
    validation = pd.DataFrame({"EventName": ["New", "A"], "LapNumber": [50, 50]})
    with pytest.warns(UserWarning, match="unknown categories"):
        prediction = pipeline.predict(validation)
    assert np.isfinite(prediction).all()
    assert prediction[0] == prediction[1]  # Unknown and dropped reference have zero event dummies.
    pre = pipeline.named_steps["preprocess"]
    imputer, scaler = pre.named_transformers_["lap"].steps
    assert imputer[1].statistics_.tolist() == [3]
    assert scaler[1].mean_.tolist() == [3.75]
    assert pre.named_transformers_["event"][-1].categories_[0].tolist() == ["A", "B", "UNKNOWN"]
    path = tmp_path / "m1.joblib"
    joblib.dump(pipeline, path)
    with pytest.warns(UserWarning, match="unknown categories"):
        np.testing.assert_array_equal(joblib.load(path).predict(validation), prediction)


@pytest.mark.parametrize("empty_column", ["EventName", "LapNumber"])
def test_feature_missing_throughout_fit_is_ignored_at_predict(pipeline, empty_column):
    train = pd.DataFrame({"EventName": ["A", "B", "A"], "LapNumber": [2, 3, 4]})
    validation = train.copy()
    train[empty_column] = np.nan
    pipeline.fit(train, [80, 81, 83])
    modified = validation.copy()
    modified[empty_column] = "New" if empty_column == "EventName" else 500
    np.testing.assert_array_equal(pipeline.predict(validation), pipeline.predict(modified))
    assert all(empty_column not in name for name in pipeline[0].get_feature_names_out())
