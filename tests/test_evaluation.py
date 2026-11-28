import numpy as np
import pandas as pd
import pytest

from f1_lap_time.evaluation import read_predictions, score_predictions


@pytest.fixture
def laps():
    reference = pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d", "e"],
            "split": ["train", "train", "train", "validation", "validation"],
            "LapTimeSec": [2.0, 4.0, 6.0, 10.0, 14.0],
        }
    )
    predictions = reference.rename(columns={"LapTimeSec": "y_true"}).copy()
    predictions["y_pred"] = [3.0, 2.0, 6.0, 12.0, 12.0]
    return reference, predictions


def score(reference, predictions):
    return score_predictions(predictions, reference=reference, model="M2", run_id="hand")


def test_saved_shuffled_predictions_match_hand_calculation(laps, tmp_path):
    reference, predictions = laps
    path = tmp_path / "predictions.csv"
    predictions.iloc[[3, 0, 4, 2, 1]].to_csv(path, index=False)
    result = score(reference, read_predictions(path)).set_index("split")
    assert result["model"].tolist() == ["M2", "M2"]
    assert result["run_id"].tolist() == ["hand", "hand"]
    assert result["n_laps"].tolist() == [3, 2]
    # Train errors +1, -2, 0: SSE=5, total sum of squares=8. Validation SSE=SST=8.
    cols = ["mae_s", "rmse_s", "mse_s2", "r2"]
    assert list(result.loc["train", cols]) == pytest.approx([1, np.sqrt(5 / 3), 5 / 3, 3 / 8])
    assert list(result.loc["validation", cols]) == pytest.approx([2, 2, 4, 0])


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda df: pd.concat([df, df.iloc[[0]]]), "Duplicate row_id: a"),
        (lambda df: df.iloc[1:], "missing 1.*a"),
        (lambda df: pd.concat([df, df.iloc[[0]].assign(row_id="z")]), "extra 1.*z"),
        (lambda df: df.assign(split="train"), "Wrong split at d"),
        (lambda df: df.assign(split=pd.NA), "Wrong split"),
        (lambda df: df.assign(split="test"), "Wrong split"),
        (lambda df: df.assign(row_id=pd.NA), "Missing row_id"),
        (lambda df: df.assign(y_true=df["y_true"].to_numpy()[::-1]), "Target mismatch"),
        (lambda df: df.assign(y_true=df["y_true"] + 1e-12), "Target mismatch"),
        (lambda df: df.assign(y_true=np.nan), "non-finite y_true"),
        (lambda df: df.assign(y_pred=-np.inf), "non-finite y_pred"),
        (lambda df: df.assign(y_pred="bad"), "Non-numeric"),
        (lambda df: df.drop(columns="y_pred"), "Missing prediction columns"),
    ],
)
def test_reject_incomparable_tables(laps, corrupt, message):
    reference, predictions = laps
    with pytest.raises(ValueError, match=message):
        score(reference, corrupt(predictions))
