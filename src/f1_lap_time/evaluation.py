"""Score saved lap predictions against the released F1-05 train and validation rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from f1_lap_time.data_preparation import TARGET_COLUMN, load_prepared_data
from f1_lap_time.protocol import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation")
PREDICTION_COLUMNS = ["row_id", "split", "y_true", "y_pred"]


def load_evaluation_data(root: Path = ROOT) -> tuple[dict[str, pd.DataFrame], dict]:
    """Pin the loader's verified files to the release used by all four models."""
    release = json.loads((root / "reports/data-preparation/checks.json").read_text("utf-8"))
    if (
        not release["all_passed"]
        or sha256_file(root / "configs/experiment.json") != release["config_sha256"]
    ):
        raise ValueError("F1-05 reference report or experiment config differs from the release")
    frames = {}
    for split in SPLITS:
        frames[split], meta = load_prepared_data(root, split=split)
        for file in ("base_csv", "splits_csv"):
            if meta["processed_outputs"][file]["sha256"] != release["hashes"][file + "_sha256"]:
                raise ValueError(f"{file} differs from the F1-05 release")
    return frames, release


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, dtype={"row_id": "string", "split": "string"}, float_precision="round_trip"
    )


def score_predictions(
    predictions: pd.DataFrame, *, model: str, run_id: str, reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Require one finite prediction per reference lap. Return two unrounded metric rows."""
    if reference is None:
        frames, _ = load_evaluation_data()
        reference = pd.concat(frames.values(), ignore_index=True)
    if missing := set(PREDICTION_COLUMNS).difference(predictions.columns):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions["row_id"].isna().any():
        raise ValueError("Missing row_id value")
    duplicates = predictions.loc[predictions["row_id"].duplicated(), "row_id"]
    if not duplicates.empty:
        raise ValueError(f"Duplicate row_id: {duplicates.iloc[0]}")
    expected = reference.set_index("row_id", verify_integrity=True)
    actual = predictions.set_index("row_id")
    missing_ids = expected.index.difference(actual.index)
    extra_ids = actual.index.difference(expected.index)
    if len(missing_ids) or len(extra_ids):
        raise ValueError(
            f"row_id mismatch: missing {len(missing_ids)} {missing_ids[:2].tolist()}, "
            f"extra {len(extra_ids)} {extra_ids[:2].tolist()}"
        )
    actual = actual.loc[expected.index]
    wrong_split = actual["split"].ne(expected["split"]).fillna(True) | actual["split"].isna()
    if wrong_split.any():
        row = expected.index[wrong_split][0]
        raise ValueError(
            f"Wrong split at {row}: expected {expected.at[row, 'split']}, got {actual.at[row, 'split']}"
        )
    for col in ("y_true", "y_pred"):
        values = pd.to_numeric(actual[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        invalid = ~np.isfinite(values)
        if invalid.any():
            raise ValueError(f"Non-numeric or non-finite {col} at {actual.index[invalid][0]}")
        actual[col] = values
    mismatch = actual["y_true"].ne(expected[TARGET_COLUMN])
    if mismatch.any():
        row = actual.index[mismatch][0]
        raise ValueError(
            f"Target mismatch at {row}: expected {expected.at[row, TARGET_COLUMN]}, got {actual.at[row, 'y_true']}"
        )

    records = []
    for split in SPLITS:
        laps = actual.loc[actual["split"] == split]
        y, pred = laps["y_true"], laps["y_pred"]
        mse = mean_squared_error(y, pred)
        metrics = dict(
            mae_s=mean_absolute_error(y, pred),
            rmse_s=np.sqrt(mse),
            mse_s2=mse,
            r2=r2_score(y, pred),
        )
        if not np.isfinite(list(metrics.values())).all():
            raise ValueError(f"Non-finite metrics for {split}")
        records.append(dict(model=model, run_id=run_id, split=split, n_laps=len(laps), **metrics))
    return pd.DataFrame(records)
