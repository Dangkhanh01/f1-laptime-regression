"""Comprehensive tests for experiment protocol configuration loading, validation, and error handling."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1_lap_time.protocol import (
    apply_protocol_filters,
    extract_split_subset,
    load_experiment_config,
    validate_config_structure,
    verify_config_against_raw_data,
)
from scripts import survey_protocol_evidence as survey
from scripts.survey_protocol_evidence import compute_train_survey_metrics


def get_sample_valid_config() -> dict:
    """Return an in-memory deep copy of a fully valid configuration dict."""
    return {
        "problem_definition": {
            "task_type": "regression",
            "objective": "Test predict lap time",
            "observation_unit": "One lap",
            "target_column": "LapTimeSec",
            "target_unit": "seconds",
            "primary_metric": "MAE",
            "reported_metrics": ["MAE", "MSE", "R2", "RMSE"],
            "metric_units": {
                "MAE": "seconds",
                "MSE": "seconds^2",
                "R2": "unitless",
                "RMSE": "seconds",
            },
        },
        "data_source": {
            "primary_file": "data/raw/f1_research_dataset_with_risk_safetycar.parquet",
            "source_doi": "10.17632/test",
            "version": 1,
            "sha256": "0" * 64,
            "raw_rows": 6,
            "raw_columns": 35,
            "candidate_key": ["Year", "EventName", "Session", "Driver", "LapNumber"],
        },
        "scope_and_filters": {
            "session_scope": "R",
            "expected_clean_rows": 6,
            "preseason_filter": {
                "action": "exclude",
                "condition": "starts_with_any",
                "match_prefixes": ["Pre-Season"],
                "matched_events": ["Pre-Season Test"],
                "rows_excluded": 0,
            },
            "safety_car_filter": {
                "action": "exclude",
                "condition": "any_equals_1",
                "columns_checked": ["IsSafetyCar", "IsVSC"],
                "rows_excluded_outside_preseason": 0,
            },
            "pit_laps_policy": {
                "action": "excluded_at_source",
                "explanation": "Excluded at source",
            },
            "target_validity": {
                "target_column": "LapTimeSec",
                "condition": "numeric_positive_finite",
            },
            "duplicate_handling": {
                "key": ["Year", "EventName", "Session", "Driver", "LapNumber"],
                "action": "raise_on_conflict_else_drop",
            },
        },
        "split_protocol": {
            "strategy": "temporal_by_year",
            "valid_years": [2022, 2023, 2024, 2025],
            "train": {
                "years": [2022, 2023],
                "expected_rows": 3,
                "distinct_event_names": 2,
                "event_session_groups": 3,
                "purpose": "Train",
            },
            "validation": {
                "years": [2024],
                "expected_rows": 2,
                "distinct_event_names": 1,
                "event_session_groups": 1,
                "purpose": "Validation",
            },
            "final_test": {
                "years": [2025],
                "expected_rows": 1,
                "distinct_event_names": 1,
                "event_session_groups": 1,
                "purpose": "Final test",
            },
        },
        "features": {
            "target": "LapTimeSec",
            "allowed_candidate_features": [
                "Driver",
                "Team",
                "EventName",
                "LapNumber",
                "Compound",
                "FreshTyre",
                "Stint",
                "TyreLife",
            ],
            "allowed_derived_features_in_pipeline": [
                {
                    "name": "TyreLife_start",
                    "formula_type": "subtract_constant",
                    "formula": "TyreLife - 1.0",
                    "condition": "TyreLife >= 1.0",
                    "input_feature": "TyreLife",
                    "missing_handling": "SimpleImputer",
                    "meaning": "Laps on tyre prior to current lap",
                }
            ],
            "pending_verification_features": [
                "AirTemp",
                "TrackTemp",
                "Humidity",
                "Pressure",
                "Rainfall",
                "WindSpeed",
                "WindDirection",
            ],
            "forbidden_features": [
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Time",
                "Position",
                "TrackStatus",
                "IsSafetyCar",
                "IsVSC",
                "PitInTime",
                "PitOutTime",
                "HasPitIn",
                "HasPitOut",
                "RaceProgress",
                "Risk_SafetyCar",
            ],
            "tracking_and_split_columns": [
                "Year",
                "Session",
                "EventKey",
                "IsAccurate",
            ],
        },
        "preprocessing_and_pipeline": {
            "pseudo_missing_strings": ["None", "nan"],
            "missing_policy": {
                "numeric_columns": {"strategy": "median"},
                "categorical_columns": {"strategy": "constant_unknown"},
                "all_missing_on_train_rule": "drop",
            },
            "unseen_category_policy": {"encoder": "OneHotEncoder", "explanation": "unseen ok"},
            "team_naming_policy": "keep raw",
            "lagged_features_policy": "disallowed",
            "leakage_boundary": "fit on train only",
        },
        "evaluation": {
            "model_selection": {
                "primary_criterion": "MAE",
                "tie_breaking_threshold_seconds": 0.005,
                "tie_breaking_rules": ["RMSE", "Simpler"],
            },
            "provenance_metadata": ["run_id", "timestamp_utc"],
            "interpretation_rules": ["Based on real output"],
        },
    }


def setup_mini_valid_dataset(tmp_path: Path) -> tuple[dict, Path]:
    """Create a self-consistent mini dataset strictly within tmp_path."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "f1_research_dataset_with_risk_safetycar.parquet"

    data = {
        "Year": [2022, 2023, 2023, 2024, 2024, 2025],
        "EventName": [
            "Grand Prix A",
            "Grand Prix A",
            "Grand Prix B",
            "Grand Prix C",
            "Grand Prix C",
            "Grand Prix D",
        ],
        "Session": ["R", "R", "R", "R", "R", "R"],
        "Driver": ["VER", "VER", "HAM", "VER", "HAM", "NOR"],
        "LapNumber": [2.0, 3.0, 2.0, 2.0, 3.0, 2.0],
        "LapTimeSec": [80.5, 81.2, 80.9, 79.1, 79.8, 77.4],
        "LapTime": [80.5, 81.2, 80.9, 79.1, 79.8, 77.4],
        "Sector1Time": [25.0, 25.1, 25.0, 24.5, 24.6, 24.0],
        "Sector2Time": [30.0, 30.2, 30.1, 29.8, 29.9, 29.0],
        "Sector3Time": [25.5, 25.9, 25.8, 24.8, 25.3, 24.4],
        "Team": [
            "Red Bull Racing",
            "Red Bull Racing",
            "Mercedes",
            "Red Bull Racing",
            "Mercedes",
            "McLaren",
        ],
        "Compound": ["MEDIUM", "MEDIUM", "HARD", "SOFT", "SOFT", "HARD"],
        "TyreLife": [5.0, 6.0, 4.0, 3.0, 4.0, 5.0],
        "FreshTyre": [True, False, True, True, False, True],
        "Stint": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "IsAccurate": [True, True, True, True, True, True],
        "TrackStatus": ["1", "1", "1", "1", "1", "1"],
        "PitInTime": [float("nan")] * 6,
        "PitOutTime": [float("nan")] * 6,
        "Position": [1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
        "Time": [3000.0, 3081.2, 3000.9, 3000.0, 3079.8, 3000.0],
        "AirTemp": [25.0, 25.1, 25.0, 26.0, 26.1, 24.0],
        "TrackTemp": [35.0, 35.2, 35.0, 36.0, 36.2, 34.0],
        "Humidity": [50.0, 50.0, 51.0, 45.0, 45.0, 55.0],
        "Pressure": [1013.0, 1013.0, 1013.0, 1012.0, 1012.0, 1014.0],
        "Rainfall": [False, False, False, False, False, False],
        "WindSpeed": [2.0, 2.1, 2.0, 1.5, 1.5, 3.0],
        "WindDirection": [180, 180, 180, 90, 90, 270],
        "IsSafetyCar": [0, 0, 0, 0, 0, 0],
        "IsVSC": [0, 0, 0, 0, 0, 0],
        "RaceProgress": [0.1, 0.2, 0.1, 0.1, 0.2, 0.1],
        "HasPitIn": [0, 0, 0, 0, 0, 0],
        "HasPitOut": [0, 0, 0, 0, 0, 0],
        "EventKey": ["E1", "E1", "E2", "E3", "E3", "E4"],
        "Risk_SafetyCar": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    }
    df = pd.DataFrame(data)
    df.to_parquet(raw_file)

    with raw_file.open("rb") as f:
        file_sha256 = hashlib.file_digest(f, "sha256").hexdigest()

    manifest_file = tmp_path / "data" / "manifest.json"
    manifest_data = {
        "source": {"doi": "10.17632/test", "version": 1},
        "selected_file": {
            "filename": "f1_research_dataset_with_risk_safetycar.parquet",
            "size_bytes": raw_file.stat().st_size,
            "sha256": file_sha256,
        },
        "extracted": {
            "files": [
                {
                    "path": "f1_safetycar/datasets/f1_research_dataset_with_risk_safetycar.parquet",
                    "size_bytes": raw_file.stat().st_size,
                    "sha256": file_sha256,
                }
            ]
        },
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    cfg = get_sample_valid_config()
    cfg["data_source"]["sha256"] = file_sha256

    return cfg, raw_file


def update_mini_dataset_and_manifest(tmp_path: Path, df: pd.DataFrame, cfg: dict) -> None:
    """Save updated df, recompute SHA-256, and update both cfg and manifest.json to isolate data errors."""
    raw_file = tmp_path / cfg["data_source"]["primary_file"]
    df.to_parquet(raw_file)

    with raw_file.open("rb") as f:
        new_hash = hashlib.file_digest(f, "sha256").hexdigest()

    cfg["data_source"]["sha256"] = new_hash
    manifest_file = tmp_path / "data" / "manifest.json"
    manifest_data = {
        "source": {
            "doi": cfg["data_source"]["source_doi"],
            "version": cfg["data_source"]["version"],
        },
        "selected_file": {
            "filename": "f1_research_dataset_with_risk_safetycar.parquet",
            "size_bytes": raw_file.stat().st_size,
            "sha256": new_hash,
        },
        "extracted": {
            "files": [
                {
                    "path": "f1_safetycar/datasets/f1_research_dataset_with_risk_safetycar.parquet",
                    "size_bytes": raw_file.stat().st_size,
                    "sha256": new_hash,
                }
            ]
        },
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")


def test_mini_valid_dataset_passes_verification(tmp_path: Path) -> None:
    cfg, _ = setup_mini_valid_dataset(tmp_path)
    res = verify_config_against_raw_data(cfg, tmp_path)
    assert res["raw_rows"] == 6
    assert res["clean_rows"] == 6
    assert res["splits"]["train"] == 3


def test_verify_catches_target_nan(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df.loc[0, "LapTimeSec"] = np.nan
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="contains NaN values"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_target_positive_inf(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df.loc[0, "LapTimeSec"] = np.inf
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="contains infinite values"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_target_negative_inf(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df.loc[0, "LapTimeSec"] = -np.inf
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="contains infinite values"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_target_non_numeric(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df["LapTimeSec"] = ["80.5", "81.2", "80.9", "79.1", "79.8", "77.4"]
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="not numeric"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_target_non_positive(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df.loc[0, "LapTimeSec"] = 0.0
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="non-positive"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_unassigned_split_row(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df.loc[0, "Year"] = 2026
    cfg["split_protocol"]["valid_years"].append(2026)
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="not assigned to any split"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_unexpected_year_outside_valid_years(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df.loc[0, "Year"] = 2030
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="unexpected years outside valid_years"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_empty_split(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df = df[df["Year"] != 2025]
    cfg["data_source"]["raw_rows"] = len(df)
    cfg["scope_and_filters"]["expected_clean_rows"] = len(df)
    cfg["split_protocol"]["final_test"]["expected_rows"] = 0
    cfg["split_protocol"]["valid_years"] = [2022, 2023, 2024, 2025]
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="is empty"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_manifest_hash_mismatch(tmp_path: Path) -> None:
    cfg, _ = setup_mini_valid_dataset(tmp_path)
    manifest_file = tmp_path / "data" / "manifest.json"
    manifest_data = {
        "source": {"doi": "10.17632/test", "version": 1},
        "selected_file": {
            "filename": "f1_research_dataset_with_risk_safetycar.parquet",
            "size_bytes": 100,
            "sha256": "0" * 64,  # Intentionally corrupted hash
        },
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest selected_file SHA-256 mismatch"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_verify_catches_filter_mismatch_when_preseason_prefix_changes(tmp_path: Path) -> None:
    cfg, raw_file = setup_mini_valid_dataset(tmp_path)
    df = pd.read_parquet(raw_file)
    df.loc[0, "EventName"] = "Pre-Season Test Event"
    cfg["scope_and_filters"]["preseason_filter"]["rows_excluded"] = 0
    update_mini_dataset_and_manifest(tmp_path, df, cfg)

    with pytest.raises(ValueError, match="Pre-Season excluded count mismatch"):
        verify_config_against_raw_data(cfg, tmp_path)


def test_validate_catches_null_allowed_features() -> None:
    cfg = get_sample_valid_config()
    cfg["features"]["allowed_candidate_features"] = None
    errors = validate_config_structure(cfg)
    assert any("must be a list" in e for e in errors)


def test_validate_catches_bool_as_split_year() -> None:
    cfg = get_sample_valid_config()
    cfg["split_protocol"]["train"]["years"] = [True, 2023]
    errors = validate_config_structure(cfg)
    assert any("must be an integer" in e for e in errors)


def test_validate_catches_duplicate_years_in_split() -> None:
    cfg = get_sample_valid_config()
    cfg["split_protocol"]["train"]["years"] = [2022, 2022]
    errors = validate_config_structure(cfg)
    assert any("duplicate integer elements" in e for e in errors)


def test_validate_catches_year_overlap_between_splits() -> None:
    cfg = get_sample_valid_config()
    cfg["split_protocol"]["validation"]["years"] = [2023]
    errors = validate_config_structure(cfg)
    assert any("Year overlap between train and validation" in e for e in errors)


def test_validate_catches_empty_scope_and_filters() -> None:
    cfg = get_sample_valid_config()
    cfg["scope_and_filters"] = {}
    errors = validate_config_structure(cfg)
    assert any("Missing required field 'scope_and_filters." in e for e in errors)


def test_validate_catches_empty_preprocessing_and_pipeline() -> None:
    cfg = get_sample_valid_config()
    cfg["preprocessing_and_pipeline"] = {}
    errors = validate_config_structure(cfg)
    assert any("Missing required field 'preprocessing_and_pipeline." in e for e in errors)


def test_validate_catches_unsupported_preseason_condition() -> None:
    cfg = get_sample_valid_config()
    cfg["scope_and_filters"]["preseason_filter"]["condition"] = "regex_match"
    errors = validate_config_structure(cfg)
    assert any("Unsupported preseason condition" in e for e in errors)


def test_validate_catches_unsupported_derived_formula() -> None:
    cfg = get_sample_valid_config()
    cfg["features"]["allowed_derived_features_in_pipeline"][0]["formula_type"] = "arbitrary_eval"
    errors = validate_config_structure(cfg)
    assert any("Unsupported formula_type" in e for e in errors)


def test_validate_catches_target_in_allowed_features() -> None:
    cfg = get_sample_valid_config()
    cfg["features"]["allowed_candidate_features"].append("LapTimeSec")
    errors = validate_config_structure(cfg)
    assert any("must not be in allowed_candidate_features" in e for e in errors)


def test_validate_catches_feature_overlap() -> None:
    cfg = get_sample_valid_config()
    cfg["features"]["forbidden_features"].append("Driver")
    errors = validate_config_structure(cfg)
    assert any("Feature set overlap between allowed and forbidden" in e for e in errors)


def test_load_default_real_experiment_config() -> None:
    config = load_experiment_config()
    assert isinstance(config, dict)
    assert config["problem_definition"]["task_type"] == "regression"
    assert config["problem_definition"]["target_column"] == "LapTimeSec"
    assert config["problem_definition"]["primary_metric"] == "MAE"

    assert config["split_protocol"]["train"]["years"] == [2022, 2023]
    assert config["split_protocol"]["validation"]["years"] == [2024]
    assert config["split_protocol"]["final_test"]["years"] == [2025]

    allowed = set(config["features"]["allowed_candidate_features"])
    forbidden = set(config["features"]["forbidden_features"])
    pending = set(config["features"]["pending_verification_features"])
    tracking = set(config["features"]["tracking_and_split_columns"])

    assert len(allowed & forbidden) == 0
    assert len(allowed & pending) == 0
    assert len(allowed & tracking) == 0
    assert len(forbidden & pending) == 0

    assert "RaceProgress" in forbidden
    assert "Driver" in allowed
    assert "AirTemp" in pending


def test_cli_structure_only_prints_notice() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "check_experiment_config.py"),
        "--check-structure-only",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0
    assert "HAVE NOT been verified" in res.stdout


def test_cli_missing_config_returns_error(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bad_cfg = tmp_path / "non_existent.json"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "check_experiment_config.py"),
        "--config",
        str(bad_cfg),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode != 0


def test_apply_protocol_filters_logic() -> None:
    cfg = get_sample_valid_config()
    df = pd.DataFrame(
        [
            # Valid lap 1
            {
                "Year": 2022,
                "EventName": "Bahrain GP",
                "Session": "R",
                "Driver": "VER",
                "LapNumber": 2.0,
                "IsSafetyCar": 0,
                "IsVSC": 0,
            },
            # Invalid session
            {
                "Year": 2022,
                "EventName": "Bahrain GP",
                "Session": "FP1",
                "Driver": "VER",
                "LapNumber": 3.0,
                "IsSafetyCar": 0,
                "IsVSC": 0,
            },
            # Pre-season
            {
                "Year": 2022,
                "EventName": "Pre-Season Test",
                "Session": "R",
                "Driver": "VER",
                "LapNumber": 4.0,
                "IsSafetyCar": 0,
                "IsVSC": 0,
            },
            # Safety car
            {
                "Year": 2022,
                "EventName": "Bahrain GP",
                "Session": "R",
                "Driver": "VER",
                "LapNumber": 5.0,
                "IsSafetyCar": 1,
                "IsVSC": 0,
            },
            # VSC
            {
                "Year": 2022,
                "EventName": "Bahrain GP",
                "Session": "R",
                "Driver": "VER",
                "LapNumber": 6.0,
                "IsSafetyCar": 0,
                "IsVSC": 1,
            },
            # Valid lap 2
            {
                "Year": 2023,
                "EventName": "Monaco GP",
                "Session": "R",
                "Driver": "HAM",
                "LapNumber": 10.0,
                "IsSafetyCar": 0,
                "IsVSC": 0,
            },
        ]
    )
    clean = apply_protocol_filters(df, cfg)
    assert len(clean) == 2
    assert set(clean["LapNumber"]) == {2.0, 10.0}


def test_extract_split_subset_logic() -> None:
    cfg = get_sample_valid_config()
    df = pd.DataFrame(
        [
            {"Year": 2022, "Val": 1},
            {"Year": 2023, "Val": 2},
            {"Year": 2024, "Val": 3},
            {"Year": 2025, "Val": 4},
        ]
    )
    train_sub = extract_split_subset(df, cfg, "train")
    assert sorted(train_sub["Year"].tolist()) == [2022, 2023]

    val_sub = extract_split_subset(df, cfg, "validation")
    assert val_sub["Year"].tolist() == [2024]

    test_sub = extract_split_subset(df, cfg, "final_test")
    assert test_sub["Year"].tolist() == [2025]

    with pytest.raises(ValueError, match="Unknown split name 'unknown_split'"):
        extract_split_subset(df, cfg, "unknown_split")


def test_compute_train_survey_metrics_strictly_isolated_from_validation() -> None:
    train_data = pd.DataFrame(
        [
            {
                "Year": 2022,
                "EventName": "Bahrain GP",
                "Session": "R",
                "Driver": "VER",
                "LapNumber": 2.0,
                "TyreLife": 2.0,
                "Compound": "SOFT",
                "TrackTemp": 30.0,
            },
            {
                "Year": 2022,
                "EventName": "Bahrain GP",
                "Session": "R",
                "Driver": "PER",
                "LapNumber": 2.0,
                "TyreLife": 2.0,
                "Compound": "None",
                "TrackTemp": 31.0,
            },
            {
                "Year": 2023,
                "EventName": "Monaco GP",
                "Session": "R",
                "Driver": "HAM",
                "LapNumber": 5.0,
                "TyreLife": np.nan,
                "Compound": "HARD",
                "TrackTemp": 25.0,
            },
        ]
    )

    metrics_initial = compute_train_survey_metrics(train_data)
    assert metrics_initial["tyrelife_train_survey"]["missing_count"] == 1
    assert metrics_initial["compound_placeholders_train"]["None"] == 1
    assert metrics_initial["weather_intra_lap_observation"]["total_lap_groups"] == 2
    assert (
        metrics_initial["weather_intra_lap_observation"]["laps_with_multiple_tracktemp_values"] == 1
    )

    # Now verify that passing full data through extract_split_subset guarantees no leakage from val_data
    val_data = pd.DataFrame(
        [
            {
                "Year": 2024,
                "EventName": "Bahrain GP",
                "Session": "R",
                "Driver": "VER",
                "LapNumber": 2.0,
                "TyreLife": 99.0,
                "Compound": "None",
                "TrackTemp": 50.0,
            }
        ]
    )
    all_data = pd.concat([train_data, val_data], ignore_index=True)
    extracted_train = extract_split_subset(all_data, get_sample_valid_config(), "train")
    metrics_isolated = compute_train_survey_metrics(extracted_train)
    assert metrics_isolated == metrics_initial


def test_real_manifest_hash_matches_raw_dataset_and_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_p = repo_root / "data" / "manifest.json"
    config_p = repo_root / "configs" / "experiment.json"
    if manifest_p.is_file() and config_p.is_file():
        with manifest_p.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        with config_p.open("r", encoding="utf-8") as f:
            config = json.load(f)

        assert manifest["selected_file"]["sha256"] == config["data_source"]["sha256"]
        assert (
            manifest["selected_file"]["filename"]
            == Path(config["data_source"]["primary_file"]).name
        )


@pytest.mark.parametrize("manifest_error", ["missing_file", "missing_hash", "wrong_hash"])
def test_survey_rejects_invalid_manifest(
    tmp_path: Path, manifest_error: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, _ = setup_mini_valid_dataset(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/experiment.json").write_text(json.dumps(cfg), encoding="utf-8")
    manifest_path = tmp_path / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_error == "missing_file":
        manifest_path.unlink()
    else:
        if manifest_error == "missing_hash":
            manifest["selected_file"].pop("sha256")
        else:
            manifest["selected_file"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError), match="[Mm]anifest"):
        survey.build_survey_evidence(tmp_path)
    monkeypatch.setattr(survey, "__file__", str(tmp_path / "scripts/survey_protocol_evidence.py"))
    assert survey.main() == 1
    assert not (tmp_path / "reports/protocol_survey.json").exists()


def test_survey_records_config_and_code_provenance(tmp_path: Path) -> None:
    cfg, _ = setup_mini_valid_dataset(tmp_path)
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs/experiment.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    evidence = survey.build_survey_evidence(tmp_path)
    provenance = evidence["provenance"]
    assert provenance["config_path"] == "configs/experiment.json"
    assert provenance["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert provenance["config_snapshot"] == cfg
    assert len(provenance["git_commit_hash"]) == 40
    assert provenance["git_dirty_flag"] is False
    assert provenance["reproduce_command"] == "uv run python -m scripts.survey_protocol_evidence"
    assert evidence["train_scope"]["years"] == [2022, 2023]
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    updated = survey.build_survey_evidence(tmp_path)
    assert updated["provenance"]["git_dirty_flag"] is True
    assert updated["provenance"]["config_sha256"] != provenance["config_sha256"]
    assert updated["train_scope"] == evidence["train_scope"]


def test_survey_separates_sessions_in_lap_and_weather_groups() -> None:
    frame = pd.DataFrame(
        {
            "Year": [2022, 2022],
            "EventName": ["Example GP", "Example GP"],
            "Session": ["R", "FP1"],
            "Driver": ["VER", "VER"],
            "LapNumber": [2.0, 2.0],
            "TyreLife": [2.0, 2.0],
            "Compound": ["SOFT", "SOFT"],
            "TrackTemp": [30.0, 40.0],
        }
    )
    result = compute_train_survey_metrics(frame)
    assert result["weather_intra_lap_observation"]["total_lap_groups"] == 2
    assert result["weather_intra_lap_observation"]["laps_with_multiple_tracktemp_values"] == 0
    assert result["lags_train_survey"]["first_observed_per_driver_event"] == 2
