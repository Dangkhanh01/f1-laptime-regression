"""Minimal reproducible survey script for F1-03 protocol evidence strictly on Train (2022-2023)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from f1_lap_time.protocol import (
    apply_protocol_filters,
    extract_split_subset,
    load_experiment_config,
    sha256_file,
)


def compute_train_survey_metrics(train_df: pd.DataFrame) -> dict[str, Any]:
    """Compute protocol survey metrics strictly on the Train subset.

    Args:
        train_df: DataFrame containing only clean Train split records.

    Returns:
        Dictionary containing survey measurements.
    """
    n_train = len(train_df)
    if n_train == 0:
        raise ValueError("Train subset is empty, cannot compute survey metrics")

    # 1. TyreLife survey on Train
    tyrelife_missing = int(train_df["TyreLife"].isna().sum())
    tyrelife_valid = train_df["TyreLife"].dropna()
    tyrelife_min = float(tyrelife_valid.min()) if not tyrelife_valid.empty else None
    tyrelife_max = float(tyrelife_valid.max()) if not tyrelife_valid.empty else None

    # 2. Compound placeholders on Train
    compound_placeholders = {
        "None": int((train_df["Compound"] == "None").sum()),
        "nan": int((train_df["Compound"] == "nan").sum()),
    }

    # 3. Lags gap analysis on Train
    driver_keys = ["Year", "EventName", "Session", "Driver"]
    sorted_train = train_df.sort_values([*driver_keys, "LapNumber"]).copy()
    sorted_train["Lap_Diff"] = sorted_train.groupby(driver_keys)["LapNumber"].diff()
    first_observed = int(sorted_train["Lap_Diff"].isna().sum())
    gap_after_filter = int((sorted_train["Lap_Diff"] > 1).sum())
    strict_l1_available = int((sorted_train["Lap_Diff"] == 1).sum())

    # 4. Weather intra-lap variation observation strictly on Train
    weather_variation = train_df.groupby(["Year", "EventName", "Session", "LapNumber"])[
        "TrackTemp"
    ].nunique()
    multi_weather_laps = int((weather_variation > 1).sum())
    total_lap_groups = len(weather_variation)

    return {
        "train_scope": {
            "years": sorted([int(y) for y in train_df["Year"].unique()]),
            "total_clean_train_rows": n_train,
            "distinct_event_names": int(train_df["EventName"].nunique()),
            "event_session_groups": int(train_df.groupby(["Year", "EventName", "Session"]).ngroups),
        },
        "tyrelife_train_survey": {
            "missing_count": tyrelife_missing,
            "missing_pct": round(tyrelife_missing * 100.0 / n_train, 4),
            "min_value": tyrelife_min,
            "max_value": tyrelife_max,
        },
        "compound_placeholders_train": compound_placeholders,
        "lags_train_survey": {
            "first_observed_per_driver_event": first_observed,
            "gaps_lap_diff_greater_than_1": gap_after_filter,
            "total_missing_strict_l_minus_1": first_observed + gap_after_filter,
            "strict_l_minus_1_available": strict_l1_available,
            "strict_l_minus_1_available_pct": round(strict_l1_available * 100.0 / n_train, 4),
        },
        "weather_intra_lap_observation": {
            "total_lap_groups": total_lap_groups,
            "laps_with_multiple_tracktemp_values": multi_weather_laps,
            "multiple_tracktemp_pct": round(multi_weather_laps * 100.0 / total_lap_groups, 4)
            if total_lap_groups > 0
            else 0.0,
            "note": "TrackTemp có thể khác nhau giữa các xe có cùng số vòng trong cùng sự kiện/phiên; quan sát này chưa xác định thời điểm lấy mẫu hoặc cách ghép thời tiết.",
        },
    }


def build_survey_evidence(repo_root: Path) -> dict[str, Any]:
    """Validate the source and build a train-only report without writing files."""
    config_path = repo_root / "configs" / "experiment.json"
    config = load_experiment_config(config_path)

    raw_rel = config["data_source"]["primary_file"]
    raw_path = repo_root / raw_rel
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    # Verify SHA-256 against config and manifest (read-only)
    actual_hash = sha256_file(raw_path)
    expected_hash = config["data_source"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 mismatch with config: {actual_hash} != {expected_hash}")

    manifest_path = repo_root / "data" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")
    selected = manifest.get("selected_file")
    if not isinstance(selected, dict):
        raise ValueError("Manifest selected_file must be a JSON object")
    expected_record = {
        "filename": raw_path.name,
        "sha256": actual_hash,
        "size_bytes": raw_path.stat().st_size,
    }
    for key, value in expected_record.items():
        if selected.get(key) != value:
            raise ValueError(f"Manifest selected_file.{key} missing or inconsistent with source")
    if "path" in selected and selected["path"] != raw_rel:
        raise ValueError("Manifest selected_file.path differs from config")
    source = manifest.get("source")
    if not isinstance(source, dict) or any(
        source.get(key) != config["data_source"][cfg_key]
        for key, cfg_key in [("doi", "source_doi"), ("version", "version")]
    ):
        raise ValueError("Manifest source DOI/version differs from config")

    df = pd.read_parquet(raw_path)

    # Use shared protocol filter helper
    clean_df = apply_protocol_filters(df, config)

    # Extract Train subset strictly via protocol split rules
    train_df = extract_split_subset(clean_df, config, "train")

    metrics = compute_train_survey_metrics(train_df)

    # Git absence is explicit (e.g. a source export), never reported as a clean checkout.
    commit, dirty = None, None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None

    return {
        "provenance": {
            "survey_version": 1,
            "config_path": config_path.relative_to(repo_root).as_posix(),
            "config_sha256": sha256_file(config_path),
            "config_snapshot": config,
            "git_commit_hash": commit,
            "git_dirty_flag": dirty,
            "code_sha256": {
                relative: sha256_file(repo_root / relative)
                for relative in [
                    "scripts/survey_protocol_evidence.py",
                    "src/f1_lap_time/protocol.py",
                ]
                if (repo_root / relative).is_file()
            },
            "reproduce_command": "uv run python -m scripts.survey_protocol_evidence",
            "lap_group_keys": ["Year", "EventName", "Session", "LapNumber"],
            "driver_group_keys": ["Year", "EventName", "Session", "Driver"],
        },
        "raw_source": {
            "path": raw_rel,
            "sha256": actual_hash,
            "total_rows": len(df),
        },
        **metrics,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        evidence = build_survey_evidence(repo_root)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1
    metrics = evidence
    out_path = repo_root / "reports" / "protocol_survey.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    print(f"Evidence survey written successfully to: {out_path.relative_to(repo_root)}")
    n_train = evidence["train_scope"]["total_clean_train_rows"]
    print(f"Train rows analyzed: {n_train:,}")
    print(
        f"TyreLife missing on Train: {metrics['tyrelife_train_survey']['missing_count']} ({metrics['tyrelife_train_survey']['missing_pct']}%)"
    )
    print(
        f"Total missing strict L-1 on Train: {metrics['lags_train_survey']['total_missing_strict_l_minus_1']} ({round(metrics['lags_train_survey']['total_missing_strict_l_minus_1'] * 100 / n_train, 2)}%)"
    )
    print(
        f"Weather intra-lap variation on Train: {metrics['weather_intra_lap_observation']['laps_with_multiple_tracktemp_values']}/{metrics['weather_intra_lap_observation']['total_lap_groups']} ({metrics['weather_intra_lap_observation']['multiple_tracktemp_pct']}%)"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
