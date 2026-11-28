"""Fit the F1-06 M1 baseline on train and score its saved predictions."""

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from f1_lap_time.data_preparation import TARGET_COLUMN
from f1_lap_time.evaluation import ROOT, load_evaluation_data, read_predictions, score_predictions
from f1_lap_time.m1 import build_m1
from f1_lap_time.protocol import sha256_file
from scripts.prepare_data import get_git_info

MODEL_CONFIG = "configs/models/m1.json"
COMMAND = "uv run python -m scripts.train_m1"


def train_m1(root: Path) -> Path:
    frames, release = load_evaluation_data(root)
    config = json.loads((root / MODEL_CONFIG).read_text("utf-8"))
    features = config["features"]
    train, val = frames["train"], frames["validation"]
    now = datetime.now(timezone.utc)
    run_id = f"m1-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    commit, dirty = get_git_info(root)
    pipeline = build_m1(config).fit(train[features], train[TARGET_COLUMN])
    artifact_dir = root / "artifacts/M1" / run_id
    report_dir = root / "reports/models/M1" / run_id
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    report_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=artifact_dir.parent) as work:
        stage = Path(work)
        binary = stage / "artifacts"
        reports = stage / "reports"
        binary.mkdir()
        reports.mkdir()
        joblib.dump(pipeline, binary / "pipeline.joblib")
        tables = []
        for frame in frames.values():
            table = frame[["row_id", "split", TARGET_COLUMN]].rename(
                columns={TARGET_COLUMN: "y_true"}
            )
            table["y_pred"] = pipeline.predict(frame[features])
            tables.append(table)
        pd.concat(tables).to_csv(binary / "predictions.csv", index=False, lineterminator="\n")
        saved = read_predictions(binary / "predictions.csv")
        scores = score_predictions(
            saved, model=config["model"], run_id=run_id, reference=pd.concat(frames.values())
        )
        scores.to_csv(reports / "scores.csv", index=False, lineterminator="\n")
        compared = val[["row_id", "EventName"]].merge(
            saved[saved["split"] == "validation"], validate="1:1", on="row_id"
        )
        compared["error"] = compared["y_pred"] - compared["y_true"]
        compared["absolute_error"] = compared["error"].abs()
        events = (
            compared.groupby("EventName")
            .agg(
                n_laps=("error", "size"),
                mae_s=("absolute_error", "mean"),
                mean_error_s=("error", "mean"),
            )
            .sort_values("mae_s", ascending=False)
        )
        events["seen_in_train"] = events.index.isin(train["EventName"])
        events.assign(run_id=run_id).to_csv(
            reports / "validation_by_event.csv", lineterminator="\n"
        )

        preprocess = pipeline.named_steps["preprocess"]
        ols = pipeline.named_steps["ols"]
        coefficients = pd.DataFrame(
            {"feature": preprocess.get_feature_names_out(), "coefficient_s": ols.coef_}
        )
        coefficients.loc[len(coefficients)] = ["intercept", ols.intercept_]
        coefficients.assign(run_id=run_id).to_csv(
            reports / "coefficients.csv", index=False, lineterminator="\n"
        )
        sample = val.iloc[[0]]
        reloaded = joblib.load(binary / "pipeline.joblib")
        np.testing.assert_array_equal(
            reloaded.predict(val[features]), tables[1]["y_pred"].to_numpy()
        )
        example = float(reloaded.predict(sample[features])[0])
        code_paths = [
            "src/f1_lap_time/m1.py",
            "src/f1_lap_time/evaluation.py",
            "scripts/train_m1.py",
            "scripts/score_predictions.py",
            "src/f1_lap_time/data_preparation.py",
            "src/f1_lap_time/protocol.py",
            "scripts/prepare_data.py",
        ]
        metadata = {
            "model": config["model"],
            "run_id": run_id,
            "timestamp_utc": now.isoformat(),
            "data_source_sha256": release["source_sha256"],
            "processed_base_sha256": release["hashes"]["base_csv_sha256"],
            "processed_splits_sha256": release["hashes"]["splits_csv_sha256"],
            "split_version": "F1-05:" + release["hashes"]["splits_csv_sha256"],
            "config_path": "configs/experiment.json",
            "config_sha256": release["config_sha256"],
            "model_config_path": MODEL_CONFIG,
            "model_config_sha256": sha256_file(root / MODEL_CONFIG),
            "model_config": config,
            "command": COMMAND,
            "git_commit_hash": commit,
            "git_dirty_flag": dirty,
            "code_files_sha256": {p: sha256_file(root / p) for p in code_paths},
            "uv_lock_sha256": sha256_file(root / "uv.lock"),
            "python_version": platform.python_version(),
            "library_versions": {
                p: version(p) for p in ("scikit-learn", "pandas", "numpy", "scipy", "joblib")
            },
            "model_class": type(ols).__name__,
            "hyperparameters": ols.get_params(),
            "random_seed": None,
            "features_list": features,
            "metrics": scores.to_dict("records"),
            "reference_event": preprocess.named_transformers_["event"][-1].categories_[0][0],
            "lap_number_scale": preprocess.named_transformers_["lap"][-1].scale_[0],
            "artifact_path": (artifact_dir / "pipeline.joblib").relative_to(root).as_posix(),
            "artifact_sha256": sha256_file(binary / "pipeline.joblib"),
            "predictions_sha256": sha256_file(binary / "predictions.csv"),
            "validation_example": {
                "row_id": sample["row_id"].iat[0],
                "input": sample[features].to_dict("records")[0],
                "y_pred_s": example,
                "y_true_s": sample[TARGET_COLUMN].iat[0],
            },
        }
        (reports / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        binary.rename(artifact_dir)
        try:
            reports.rename(report_dir)
        except OSError:
            artifact_dir.rename(binary)
            raise
    return report_dir


if __name__ == "__main__":
    print(train_m1(ROOT))
