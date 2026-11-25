"""Experiment protocol configuration validation and helpers for F1 Lap Time Regression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_TOP_LEVEL_SECTIONS = [
    "problem_definition",
    "data_source",
    "scope_and_filters",
    "split_protocol",
    "features",
    "preprocessing_and_pipeline",
    "evaluation",
]

SUPPORTED_PRESEASON_CONDITIONS = {"starts_with_any", "isin"}
SUPPORTED_SC_CONDITIONS = {"any_equals_1"}
SUPPORTED_TARGET_CONDITIONS = {"numeric_positive_finite"}
SUPPORTED_DERIVED_FORMULA_TYPES = {"subtract_constant"}


def get_default_config_path() -> Path:
    """Return default path to configs/experiment.json relative to repository root."""
    return Path(__file__).resolve().parents[2] / "configs" / "experiment.json"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _check_str(d: dict[str, Any], key: str, path: str, errors: list[str]) -> str | None:
    if key not in d:
        errors.append(f"Missing required field '{path}.{key}'")
        return None
    val = d[key]
    if not isinstance(val, str) or val.strip() == "":
        errors.append(f"Field '{path}.{key}' must be a non-empty string, got: {type(val).__name__}")
        return None
    return val


def _check_int(
    d: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    min_val: int | None = None,
) -> int | None:
    if key not in d:
        errors.append(f"Missing required field '{path}.{key}'")
        return None
    val = d[key]
    if type(val) is not int:
        errors.append(f"Field '{path}.{key}' must be an integer, got: {type(val).__name__}")
        return None
    if min_val is not None and val < min_val:
        errors.append(f"Field '{path}.{key}' must be >= {min_val}, got: {val}")
        return None
    return val


def _check_float_or_int(
    d: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    min_val: float | None = None,
) -> float | None:
    if key not in d:
        errors.append(f"Missing required field '{path}.{key}'")
        return None
    val = d[key]
    if type(val) not in (int, float):
        errors.append(
            f"Field '{path}.{key}' must be a number (float or int), got: {type(val).__name__}"
        )
        return None
    if min_val is not None and val < min_val:
        errors.append(f"Field '{path}.{key}' must be >= {min_val}, got: {val}")
        return None
    return float(val)


def _check_str_list(
    d: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    allow_empty: bool = False,
) -> list[str] | None:
    if key not in d:
        errors.append(f"Missing required field '{path}.{key}'")
        return None
    val = d[key]
    if not isinstance(val, list):
        errors.append(f"Field '{path}.{key}' must be a list, got: {type(val).__name__}")
        return None
    if not allow_empty and len(val) == 0:
        errors.append(f"Field '{path}.{key}' must not be empty")
        return None
    for idx, item in enumerate(val):
        if not isinstance(item, str) or item.strip() == "":
            errors.append(
                f"Element {idx} in '{path}.{key}' must be a non-empty string, got: {type(item).__name__}"
            )
            return None
    if len(val) != len(set(val)):
        errors.append(f"Field '{path}.{key}' contains duplicate elements")
        return None
    return val


def _check_int_list(
    d: dict[str, Any],
    key: str,
    path: str,
    errors: list[str],
    allow_empty: bool = False,
) -> list[int] | None:
    if key not in d:
        errors.append(f"Missing required field '{path}.{key}'")
        return None
    val = d[key]
    if not isinstance(val, list):
        errors.append(f"Field '{path}.{key}' must be a list, got: {type(val).__name__}")
        return None
    if not allow_empty and len(val) == 0:
        errors.append(f"Field '{path}.{key}' must not be empty")
        return None
    for idx, item in enumerate(val):
        if type(item) is not int:
            errors.append(
                f"Element {idx} in '{path}.{key}' must be an integer, got: {type(item).__name__}"
            )
            return None
    if len(val) != len(set(val)):
        errors.append(f"Field '{path}.{key}' contains duplicate integer elements")
        return None
    return val


def _check_temporal_split_order(split_years: dict[str, set[int]], errors: list[str]) -> None:
    """Require every year in each split to precede the following split."""
    for earlier, later in [("train", "validation"), ("validation", "final_test")]:
        if max(split_years[earlier]) >= min(split_years[later]):
            errors.append(
                f"Temporal split order requires all {earlier} years to precede all {later} years"
            )


def load_experiment_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load and parse experiment JSON configuration.

    Args:
        config_path: Path to configuration JSON file. If None, uses default.

    Returns:
        Dictionary containing parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If JSON is invalid or missing required top-level sections.
    """
    path = Path(config_path) if config_path is not None else get_default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid JSON in {path}: {err}") from err

    errors = validate_config_structure(config)
    if errors:
        raise ValueError(
            "Config structure validation failed with errors:\n - " + "\n - ".join(errors)
        )

    return config


def validate_config_structure(config: Any) -> list[str]:
    """Validate structure and type integrity of experiment configuration.

    Returns:
        List of error strings (empty if completely valid).
    """
    errors: list[str] = []

    if not isinstance(config, dict):
        return ["Configuration root must be a JSON object (dict)"]

    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in config:
            errors.append(f"Missing required top-level section: '{section}'")
        elif not isinstance(config[section], dict):
            errors.append(
                f"Section '{section}' must be a JSON object (dict), got: {type(config[section]).__name__}"
            )

    if errors:
        return errors

    # 1. problem_definition
    prob = config["problem_definition"]
    _check_str(prob, "task_type", "problem_definition", errors)
    if prob.get("task_type") != "regression":
        errors.append(
            f"problem_definition.task_type must be 'regression', got '{prob.get('task_type')}'"
        )
    _check_str(prob, "target_column", "problem_definition", errors)
    _check_str(prob, "target_unit", "problem_definition", errors)
    _check_str(prob, "primary_metric", "problem_definition", errors)
    _check_str_list(prob, "reported_metrics", "problem_definition", errors)
    if "metric_units" not in prob or not isinstance(prob["metric_units"], dict):
        errors.append("problem_definition.metric_units must be a dict mapping metric to unit")

    # 2. data_source
    ds = config["data_source"]
    _check_str(ds, "primary_file", "data_source", errors)
    _check_str(ds, "source_doi", "data_source", errors)
    _check_int(ds, "version", "data_source", errors, min_val=1)
    _check_str(ds, "sha256", "data_source", errors)
    _check_int(ds, "raw_rows", "data_source", errors, min_val=1)
    _check_int(ds, "raw_columns", "data_source", errors, min_val=1)
    _check_str_list(ds, "candidate_key", "data_source", errors)

    # 3. scope_and_filters
    sf = config["scope_and_filters"]
    _check_str(sf, "session_scope", "scope_and_filters", errors)
    _check_int(sf, "expected_clean_rows", "scope_and_filters", errors, min_val=1)

    if "preseason_filter" not in sf or not isinstance(sf["preseason_filter"], dict):
        errors.append("scope_and_filters.preseason_filter must be a dict")
    else:
        ps = sf["preseason_filter"]
        action = _check_str(ps, "action", "scope_and_filters.preseason_filter", errors)
        if action and action != "exclude":
            errors.append("scope_and_filters.preseason_filter.action must be 'exclude'")
        cond = _check_str(ps, "condition", "scope_and_filters.preseason_filter", errors)
        if cond and cond not in SUPPORTED_PRESEASON_CONDITIONS:
            errors.append(
                f"Unsupported preseason condition: '{cond}'. Supported: {sorted(SUPPORTED_PRESEASON_CONDITIONS)}"
            )
        _check_str_list(ps, "matched_events", "scope_and_filters.preseason_filter", errors)
        _check_int(ps, "rows_excluded", "scope_and_filters.preseason_filter", errors, min_val=0)

    if "safety_car_filter" not in sf or not isinstance(sf["safety_car_filter"], dict):
        errors.append("scope_and_filters.safety_car_filter must be a dict")
    else:
        sc = sf["safety_car_filter"]
        action = _check_str(sc, "action", "scope_and_filters.safety_car_filter", errors)
        if action and action != "exclude":
            errors.append("scope_and_filters.safety_car_filter.action must be 'exclude'")
        cond_sc = _check_str(sc, "condition", "scope_and_filters.safety_car_filter", errors)
        if cond_sc and cond_sc not in SUPPORTED_SC_CONDITIONS:
            errors.append(
                f"Unsupported safety car condition: '{cond_sc}'. Supported: {sorted(SUPPORTED_SC_CONDITIONS)}"
            )
        _check_str_list(sc, "columns_checked", "scope_and_filters.safety_car_filter", errors)
        _check_int(
            sc,
            "rows_excluded_outside_preseason",
            "scope_and_filters.safety_car_filter",
            errors,
            min_val=0,
        )

    if "pit_laps_policy" not in sf or not isinstance(sf["pit_laps_policy"], dict):
        errors.append("scope_and_filters.pit_laps_policy must be a dict")
    else:
        pit = sf["pit_laps_policy"]
        _check_str(pit, "action", "scope_and_filters.pit_laps_policy", errors)
        _check_str(pit, "explanation", "scope_and_filters.pit_laps_policy", errors)

    if "target_validity" not in sf or not isinstance(sf["target_validity"], dict):
        errors.append("scope_and_filters.target_validity must be a dict")
    else:
        tv = sf["target_validity"]
        validity_target = _check_str(
            tv, "target_column", "scope_and_filters.target_validity", errors
        )
        if validity_target and validity_target != prob.get("target_column"):
            errors.append(
                "Target mismatch between scope_and_filters.target_validity.target_column "
                "and problem_definition.target_column"
            )
        cond_tv = _check_str(tv, "condition", "scope_and_filters.target_validity", errors)
        if cond_tv and cond_tv not in SUPPORTED_TARGET_CONDITIONS:
            errors.append(
                f"Unsupported target validity condition: '{cond_tv}'. Supported: {sorted(SUPPORTED_TARGET_CONDITIONS)}"
            )

    if "duplicate_handling" not in sf or not isinstance(sf["duplicate_handling"], dict):
        errors.append("scope_and_filters.duplicate_handling must be a dict")
    else:
        dh = sf["duplicate_handling"]
        _check_str_list(dh, "key", "scope_and_filters.duplicate_handling", errors)
        _check_str(dh, "action", "scope_and_filters.duplicate_handling", errors)

    # 4. split_protocol
    split = config["split_protocol"]
    strategy = _check_str(split, "strategy", "split_protocol", errors)
    if strategy and strategy != "temporal_by_year":
        errors.append("split_protocol.strategy must be 'temporal_by_year'")
    valid_years = _check_int_list(split, "valid_years", "split_protocol", errors)

    split_years_seen: dict[str, set[int]] = {}
    for split_key in ["train", "validation", "final_test"]:
        if split_key not in split or not isinstance(split[split_key], dict):
            errors.append(f"split_protocol.{split_key} must be a dict")
            continue
        sub_split = split[split_key]
        years = _check_int_list(sub_split, "years", f"split_protocol.{split_key}", errors)
        _check_int(sub_split, "expected_rows", f"split_protocol.{split_key}", errors, min_val=1)
        _check_int(
            sub_split, "distinct_event_names", f"split_protocol.{split_key}", errors, min_val=1
        )
        _check_int(
            sub_split, "event_session_groups", f"split_protocol.{split_key}", errors, min_val=1
        )
        _check_str(sub_split, "purpose", f"split_protocol.{split_key}", errors)
        if years:
            split_years_seen[split_key] = set(years)

    # Check pairwise year overlap
    if len(split_years_seen) == 3:
        _check_temporal_split_order(split_years_seen, errors)
        train_y = split_years_seen["train"]
        val_y = split_years_seen["validation"]
        test_y = split_years_seen["final_test"]
        if train_y & val_y:
            errors.append(f"Year overlap between train and validation: {sorted(train_y & val_y)}")
        if train_y & test_y:
            errors.append(f"Year overlap between train and final_test: {sorted(train_y & test_y)}")
        if val_y & test_y:
            errors.append(
                f"Year overlap between validation and final_test: {sorted(val_y & test_y)}"
            )

        all_splits_combined = train_y | val_y | test_y
        if valid_years and all_splits_combined != set(valid_years):
            errors.append(
                f"Combined split years {sorted(all_splits_combined)} do not equal valid_years {sorted(valid_years)}"
            )

    # 5. features
    feats = config["features"]
    target = _check_str(feats, "target", "features", errors)
    prob_target = prob.get("target_column")
    if target and prob_target and target != prob_target:
        errors.append(
            f"Target mismatch between problem_definition ('{prob_target}') and features ('{target}')"
        )

    allowed_list = _check_str_list(feats, "allowed_candidate_features", "features", errors)
    forbidden_list = _check_str_list(feats, "forbidden_features", "features", errors)
    tracking_list = _check_str_list(feats, "tracking_and_split_columns", "features", errors)
    pending_list = _check_str_list(
        feats, "pending_verification_features", "features", errors, allow_empty=True
    )

    allowed = set(allowed_list) if allowed_list is not None else set()
    forbidden = set(forbidden_list) if forbidden_list is not None else set()
    tracking = set(tracking_list) if tracking_list is not None else set()
    pending = set(pending_list) if pending_list is not None else set()

    if target:
        if target in allowed:
            errors.append(f"Target '{target}' must not be in allowed_candidate_features")
        if target in forbidden:
            errors.append(f"Target '{target}' must not be in forbidden_features")
        if target in tracking:
            errors.append(f"Target '{target}' must not be in tracking_and_split_columns")
        if target in pending:
            errors.append(f"Target '{target}' must not be in pending_verification_features")

    # Mutual exclusivity checks across all 4 disjoint feature subsets
    pairs = [
        ("allowed", allowed, "forbidden", forbidden),
        ("allowed", allowed, "tracking", tracking),
        ("allowed", allowed, "pending", pending),
        ("forbidden", forbidden, "tracking", tracking),
        ("forbidden", forbidden, "pending", pending),
        ("tracking", tracking, "pending", pending),
    ]
    for name_a, set_a, name_b, set_b in pairs:
        inter = set_a & set_b
        if inter:
            errors.append(f"Feature set overlap between {name_a} and {name_b}: {sorted(inter)}")

    # Derived features specification
    derived = feats.get("allowed_derived_features_in_pipeline", [])
    if not isinstance(derived, list):
        errors.append("features.allowed_derived_features_in_pipeline must be a list")
    else:
        for idx, item in enumerate(derived):
            if not isinstance(item, dict):
                errors.append(
                    f"Derived feature {idx} must be a dictionary, got: {type(item).__name__}"
                )
                continue
            name = _check_str(item, "name", f"derived[{idx}]", errors)
            f_type = _check_str(item, "formula_type", f"derived[{idx}]", errors)
            if f_type and f_type not in SUPPORTED_DERIVED_FORMULA_TYPES:
                errors.append(
                    f"Unsupported formula_type '{f_type}' in derived feature '{name}'. Supported: {sorted(SUPPORTED_DERIVED_FORMULA_TYPES)}"
                )
            inp = _check_str(item, "input_feature", f"derived[{idx}]", errors)
            if inp and inp not in allowed:
                errors.append(
                    f"Input feature '{inp}' for derived feature '{name}' must be in allowed_candidate_features"
                )
            _check_str(item, "condition", f"derived[{idx}]", errors)
            _check_str(item, "missing_handling", f"derived[{idx}]", errors)
            _check_str(item, "meaning", f"derived[{idx}]", errors)

    # 6. preprocessing_and_pipeline
    prep = config["preprocessing_and_pipeline"]
    _check_str_list(prep, "pseudo_missing_strings", "preprocessing_and_pipeline", errors)
    if "missing_policy" not in prep or not isinstance(prep["missing_policy"], dict):
        errors.append("preprocessing_and_pipeline.missing_policy must be a dict")
    else:
        mp = prep["missing_policy"]
        if "numeric_columns" not in mp or not isinstance(mp["numeric_columns"], dict):
            errors.append("missing_policy.numeric_columns must be a dict")
        if "categorical_columns" not in mp or not isinstance(mp["categorical_columns"], dict):
            errors.append("missing_policy.categorical_columns must be a dict")
        _check_str(mp, "all_missing_on_train_rule", "missing_policy", errors)

    if "unseen_category_policy" not in prep or not isinstance(prep["unseen_category_policy"], dict):
        errors.append("preprocessing_and_pipeline.unseen_category_policy must be a dict")
    else:
        ucp = prep["unseen_category_policy"]
        _check_str(ucp, "encoder", "unseen_category_policy", errors)
        _check_str(ucp, "explanation", "unseen_category_policy", errors)

    _check_str(prep, "team_naming_policy", "preprocessing_and_pipeline", errors)
    _check_str(prep, "lagged_features_policy", "preprocessing_and_pipeline", errors)
    _check_str(prep, "leakage_boundary", "preprocessing_and_pipeline", errors)

    # 7. evaluation
    eval_sec = config["evaluation"]
    if "model_selection" not in eval_sec or not isinstance(eval_sec["model_selection"], dict):
        errors.append("Missing or invalid 'model_selection' in evaluation")
    else:
        ms = eval_sec["model_selection"]
        _check_str(ms, "primary_criterion", "evaluation.model_selection", errors)
        _check_float_or_int(
            ms, "tie_breaking_threshold_seconds", "evaluation.model_selection", errors, min_val=0.0
        )
        _check_str_list(ms, "tie_breaking_rules", "evaluation.model_selection", errors)

    _check_str_list(eval_sec, "provenance_metadata", "evaluation", errors)
    _check_str_list(eval_sec, "interpretation_rules", "evaluation", errors)

    return errors


def apply_protocol_filters(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Filter raw dataset according to protocol rules (Session, Pre-season, SC/VSC).

    Args:
        df: Raw pandas DataFrame.
        config: Validated experiment configuration dictionary.

    Returns:
        Clean DataFrame meeting all protocol filtering criteria.

    Raises:
        ValueError: If filtering conditions or required columns are invalid.
    """
    clean, _ = filter_with_counts(df, config)
    return clean


def filter_with_counts(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the protocol once and count sequential exclusions without double-counting."""
    sf = config["scope_and_filters"]
    for filter_name in ["preseason_filter", "safety_car_filter"]:
        if sf[filter_name].get("action") != "exclude":
            raise ValueError(f"scope_and_filters.{filter_name}.action must be 'exclude'")

    # 1. Session scope
    sess_val = sf["session_scope"]
    if "Session" not in df.columns:
        raise ValueError("Missing 'Session' column in DataFrame for session filter")
    sess_mask = df["Session"] == sess_val

    # 2. Pre-season filter
    ps_cfg = sf["preseason_filter"]
    ps_cond = ps_cfg["condition"]
    if "EventName" not in df.columns:
        raise ValueError("Missing 'EventName' column in DataFrame for preseason filter")

    if ps_cond == "starts_with_any":
        prefixes = tuple(ps_cfg.get("match_prefixes", ["Pre-Season"]))
        preseason_mask = df["EventName"].str.startswith(prefixes)
    elif ps_cond == "isin":
        preseason_mask = df["EventName"].isin(ps_cfg["matched_events"])
    else:
        raise ValueError(f"Unsupported preseason condition in config: '{ps_cond}'")

    # 3. Safety Car / VSC filter
    sc_cfg = sf["safety_car_filter"]
    sc_cond = sc_cfg["condition"]
    cols = sc_cfg["columns_checked"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing safety car column '{c}' in DataFrame")

    if sc_cond == "any_equals_1":
        sc_mask = (df[cols] == 1).any(axis=1)
    else:
        raise ValueError(f"Unsupported safety car condition in config: '{sc_cond}'")

    clean_mask = sess_mask & (~preseason_mask) & (~sc_mask)
    return df[clean_mask].copy(), {
        "excluded_session": int((~sess_mask).sum()),
        "excluded_preseason": int((sess_mask & preseason_mask).sum()),
        "excluded_sc_vsc": int((sess_mask & ~preseason_mask & sc_mask).sum()),
    }


def extract_split_subset(
    clean_df: pd.DataFrame,
    config: dict[str, Any],
    split_name: str,
) -> pd.DataFrame:
    """Extract rows belonging to a specific temporal split ('train', 'validation', 'final_test').

    Args:
        clean_df: Clean DataFrame after protocol filters.
        config: Validated experiment configuration dictionary.
        split_name: Name of split ('train', 'validation', or 'final_test').

    Returns:
        Subset DataFrame for the given split.

    Raises:
        ValueError: If the split name, strategy, temporal order, or 'Year' column is invalid.
    """
    splits = config["split_protocol"]
    if split_name not in splits:
        raise ValueError(
            f"Unknown split name '{split_name}'. Available: {[k for k in splits if isinstance(splits[k], dict)]}"
        )
    if "Year" not in clean_df.columns:
        raise ValueError("Missing 'Year' column in DataFrame for split extraction")

    if splits.get("strategy") != "temporal_by_year":
        raise ValueError("split_protocol.strategy must be 'temporal_by_year'")
    split_years: dict[str, set[int]] = {}
    errors: list[str] = []
    for name in ["train", "validation", "final_test"]:
        years = _check_int_list(splits[name], "years", f"split_protocol.{name}", errors)
        if years:
            split_years[name] = set(years)
    if len(split_years) == 3:
        _check_temporal_split_order(split_years, errors)
    if errors:
        raise ValueError("Invalid split protocol:\n - " + "\n - ".join(errors))

    years = splits[split_name]["years"]
    return clean_df[clean_df["Year"].isin(years)].copy()


def read_verified_source(config: dict[str, Any], repo_root: Path) -> pd.DataFrame:
    """Read the Parquet after verifying its identity, manifest, and dimensions."""
    primary_rel = config["data_source"]["primary_file"]
    raw_path = repo_root / primary_rel
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw primary dataset file not found: {raw_path}")

    # 1. Verify file hash against config
    actual_hash = sha256_file(raw_path)
    expected_hash = config["data_source"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"Raw dataset SHA-256 mismatch: got {actual_hash}, expected {expected_hash}"
        )

    # 2. Check manifest (required in repo environment)
    manifest_path = repo_root / "data" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "selected_file" not in manifest or not isinstance(manifest["selected_file"], dict):
        raise ValueError("Missing 'selected_file' section in manifest.json")

    sel_record = manifest["selected_file"]
    target_name = Path(primary_rel).name
    if sel_record.get("filename") != target_name:
        raise ValueError(
            f"Manifest selected_file filename mismatch: '{sel_record.get('filename')}' != '{target_name}'"
        )
    if sel_record.get("sha256") != expected_hash:
        raise ValueError(
            f"Manifest selected_file SHA-256 mismatch: '{sel_record.get('sha256')}' != '{expected_hash}'"
        )
    if sel_record.get("size_bytes") != raw_path.stat().st_size:
        raise ValueError(
            f"Manifest selected_file size mismatch: {sel_record.get('size_bytes')} != {raw_path.stat().st_size}"
        )

    # Check source DOI and version consistency
    manifest_source = manifest.get("source", {})
    cfg_doi = config["data_source"].get("source_doi")
    if manifest_source.get("doi") != cfg_doi:
        raise ValueError(
            f"Source DOI mismatch with manifest: '{manifest_source.get('doi')}' != '{cfg_doi}'"
        )
    cfg_ver = config["data_source"].get("version")
    if manifest_source.get("version") != cfg_ver:
        raise ValueError(
            f"Source version mismatch with manifest: {manifest_source.get('version')} != {cfg_ver}"
        )

    df = pd.read_parquet(raw_path)
    if df.shape[0] != config["data_source"]["raw_rows"]:
        raise ValueError(f"Raw rows mismatch: {df.shape[0]} != {config['data_source']['raw_rows']}")
    if df.shape[1] != config["data_source"]["raw_columns"]:
        raise ValueError(
            f"Raw columns mismatch: {df.shape[1]} != {config['data_source']['raw_columns']}"
        )

    return df


def verify_config_against_raw_data(
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Verify configuration rules against raw dataset.

    Raises:
        FileNotFoundError: If raw dataset file or manifest file is missing.
        ValueError: If validation checks fail.
    """
    df = read_verified_source(config, repo_root)
    raw_cols = set(df.columns)
    target = config["scope_and_filters"]["target_validity"].get("target_column")
    if (
        not isinstance(target, str)
        or not target.strip()
        or target != config["features"]["target"]
        or target != config["problem_definition"]["target_column"]
    ):
        raise ValueError(
            "Target mismatch: scope_and_filters.target_validity.target_column, "
            "features.target, and problem_definition.target_column must agree"
        )
    allowed = config["features"]["allowed_candidate_features"]
    forbidden = config["features"]["forbidden_features"]
    tracking = config["features"]["tracking_and_split_columns"]
    pending = config["features"].get("pending_verification_features", [])

    all_categorized = set([target] + allowed + forbidden + tracking + pending)
    if all_categorized != raw_cols:
        missing_in_categorized = raw_cols - all_categorized
        extra_in_categorized = all_categorized - raw_cols
        raise ValueError(
            f"Columns categorization incomplete: missing in config: {sorted(missing_in_categorized)}, extra: {sorted(extra_in_categorized)}"
        )

    # 3. Check candidate key
    key = config["data_source"]["candidate_key"]
    if df[key].isna().any().any():
        raise ValueError("Candidate key contains null values in raw data")
    if df.duplicated(subset=key).any():
        raise ValueError("Candidate key contains duplicate rows in raw data")

    # 4. Check individual filter exclusion counts
    ps_cfg = config["scope_and_filters"]["preseason_filter"]
    ps_cond = ps_cfg["condition"]
    if ps_cond == "starts_with_any":
        prefixes = tuple(ps_cfg.get("match_prefixes", ["Pre-Season"]))
        preseason_mask = df["EventName"].str.startswith(prefixes)
    elif ps_cond == "isin":
        preseason_mask = df["EventName"].isin(ps_cfg["matched_events"])
    else:
        raise ValueError(f"Unsupported preseason condition in config: '{ps_cond}'")

    n_pre = int(preseason_mask.sum())
    if n_pre != ps_cfg["rows_excluded"]:
        raise ValueError(
            f"Pre-Season excluded count mismatch: got {n_pre}, expected {ps_cfg['rows_excluded']}"
        )

    sc_cfg = config["scope_and_filters"]["safety_car_filter"]
    cols = sc_cfg["columns_checked"]
    sc_mask = (df[cols] == 1).any(axis=1)

    n_sc_outside_pre = int((sc_mask & (~preseason_mask)).sum())
    if n_sc_outside_pre != sc_cfg["rows_excluded_outside_preseason"]:
        raise ValueError(
            f"SC/VSC excluded mismatch: got {n_sc_outside_pre}, expected {sc_cfg['rows_excluded_outside_preseason']}"
        )

    # Use helper for clean DataFrame
    clean_df = apply_protocol_filters(df, config)

    # Target validity: numeric, finite, positive, non-null
    if not pd.api.types.is_numeric_dtype(clean_df[target]):
        raise ValueError(f"Target '{target}' column is not numeric: {clean_df[target].dtype}")
    if clean_df[target].isna().any():
        raise ValueError(f"Target '{target}' contains NaN values in clean dataset")
    if np.isinf(clean_df[target]).any():
        raise ValueError(
            f"Target '{target}' contains infinite values (+inf or -inf) in clean dataset"
        )
    if (clean_df[target] <= 0).any():
        raise ValueError(f"Target '{target}' contains non-positive values (<= 0) in clean dataset")

    exp_clean = config["scope_and_filters"]["expected_clean_rows"]
    if len(clean_df) != exp_clean:
        raise ValueError(f"Clean rows mismatch: got {len(clean_df)}, expected {exp_clean}")

    # 5. Check splits rigorously: completeness, mutual exclusivity, no missing/unassigned rows
    splits = config["split_protocol"]
    valid_years = set(splits["valid_years"])
    clean_years = set(clean_df["Year"].unique())
    if not clean_years.issubset(valid_years):
        unexpected_years = clean_years - valid_years
        raise ValueError(
            f"Clean dataset contains unexpected years outside valid_years: {sorted(unexpected_years)}"
        )

    train_years = splits["train"]["years"]
    val_years = splits["validation"]["years"]
    test_years = splits["final_test"]["years"]

    train_mask = clean_df["Year"].isin(train_years)
    val_mask = clean_df["Year"].isin(val_years)
    test_mask = clean_df["Year"].isin(test_years)

    split_assignments = train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)
    if (split_assignments == 0).any():
        unassigned_count = int((split_assignments == 0).sum())
        raise ValueError(f"{unassigned_count} clean rows were not assigned to any split")
    if (split_assignments > 1).any():
        multi_assigned_count = int((split_assignments > 1).sum())
        raise ValueError(f"{multi_assigned_count} clean rows were assigned to multiple splits")

    results: dict[str, int] = {}
    for split_key in ["train", "validation", "final_test"]:
        sub = extract_split_subset(clean_df, config, split_key)
        if len(sub) == 0:
            raise ValueError(f"Split '{split_key}' is empty (0 rows)")

        exp_rows = splits[split_key]["expected_rows"]
        if len(sub) != exp_rows:
            raise ValueError(f"{split_key} rows mismatch: got {len(sub)}, expected {exp_rows}")

        exp_events = splits[split_key].get("distinct_event_names")
        if exp_events and sub["EventName"].nunique() != exp_events:
            raise ValueError(
                f"{split_key} distinct event names mismatch: got {sub['EventName'].nunique()}, expected {exp_events}"
            )

        exp_groups = splits[split_key].get("event_session_groups")
        if exp_groups:
            act_groups = sub.groupby(["Year", "EventName", "Session"]).ngroups
            if act_groups != exp_groups:
                raise ValueError(
                    f"{split_key} event session groups mismatch: got {act_groups}, expected {exp_groups}"
                )

        results[split_key] = len(sub)

    return {
        "raw_rows": len(df),
        "clean_rows": len(clean_df),
        "splits": results,
    }
