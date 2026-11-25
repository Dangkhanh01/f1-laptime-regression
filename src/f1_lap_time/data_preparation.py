"""Data preparation logic and utilities for F1 Lap Time Regression (F1-05).

Provides deterministic scope filtering, duplicate checking, pseudo-missing value
normalization, stable row_id generation, temporal split assignment, and loading
utilities for downstream modeling and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_lap_time.protocol import (
    filter_with_counts,
    sha256_file,
)

KEY_COLUMNS: list[str] = ["Year", "EventName", "Session", "Driver", "LapNumber"]
TARGET_COLUMN: str = "LapTimeSec"

ALLOWED_CANDIDATE_FEATURES: list[str] = [
    "Driver",
    "Team",
    "EventName",
    "LapNumber",
    "Compound",
    "FreshTyre",
    "Stint",
    "TyreLife",
]

TRACKING_AND_SPLIT_COLUMNS: list[str] = [
    "Year",
    "Session",
    "EventKey",
    "IsAccurate",
]

BASE_COLUMNS: list[str] = [
    "row_id",
    "Year",
    "EventName",
    "Session",
    "Driver",
    "LapNumber",
    "Team",
    "Compound",
    "FreshTyre",
    "Stint",
    "TyreLife",
    "EventKey",
    "IsAccurate",
    "LapTimeSec",
]

BASE_DTYPES: dict[str, str] = {
    "row_id": "string",
    "Year": "int64",
    "EventName": "string",
    "Session": "string",
    "Driver": "string",
    "LapNumber": "float64",
    "Team": "string",
    "Compound": "object",
    "FreshTyre": "bool",
    "Stint": "float64",
    "TyreLife": "float64",
    "EventKey": "string",
    "IsAccurate": "bool",
    "LapTimeSec": "float64",
}

SPLIT_COLUMNS: list[str] = ["row_id", "split"]

SPLIT_DTYPES: dict[str, str] = {
    "row_id": "string",
    "split": "string",
}

PSEUDO_MISSING_STRINGS: tuple[str, ...] = ("None", "nan")


def normalize_pseudo_missing(
    df: pd.DataFrame,
    column: str = "Compound",
    pseudo_strings: tuple[str, ...] | list[str] = PSEUDO_MISSING_STRINGS,
) -> pd.DataFrame:
    """Replace string representations of missing values (e.g. 'None', 'nan') with NaN.

    Strips whitespace before comparison to catch entries like ' None ' or 'nan '.

    Args:
        df: Input DataFrame.
        column: Name of the column to normalize.
        pseudo_strings: Collection of strings representing pseudo-missing values.

    Returns:
        DataFrame with normalized missing values in the given column.
    """
    result = df.copy()
    if column in result.columns:
        series_clean = result[column].astype(str).str.strip()
        mask = series_clean.isin(pseudo_strings) | (series_clean == "") | result[column].isna()
        result.loc[mask, column] = np.nan
    return result


def validate_target_column(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> None:
    """Validate that target column exists, is numeric, finite, positive, and non-null.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.

    Raises:
        ValueError: If target column violates any protocol validity rule.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in DataFrame")

    series = df[target_col]
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Target column '{target_col}' must be numeric, got: {series.dtype}")
    if series.isna().any():
        raise ValueError(
            f"Target column '{target_col}' contains NaN values ({series.isna().sum()})"
        )
    if np.isinf(series).any():
        raise ValueError(f"Target column '{target_col}' contains infinite values")
    if (series <= 0).any():
        raise ValueError(f"Target column '{target_col}' contains non-positive values (<= 0)")


def validate_tyrelife_column(df: pd.DataFrame, col: str = "TyreLife") -> None:
    """Validate that TyreLife values (when present) are finite numbers >= 1.0.

    Per Protocol Section 5.2, non-numeric, infinite, or values < 1.0 must raise an error.
    NaN values are kept as missing.

    Args:
        df: Input DataFrame.
        col: TyreLife column name.

    Raises:
        ValueError: If TyreLife contains non-numeric, infinite, or values < 1.0.
    """
    if col not in df.columns:
        return
    valid_series = df[col].dropna()
    if valid_series.empty:
        return
    if not pd.api.types.is_numeric_dtype(valid_series):
        raise ValueError(f"Column '{col}' must be numeric, got: {valid_series.dtype}")
    if np.isinf(valid_series).any():
        raise ValueError(f"Column '{col}' contains infinite values (+inf or -inf)")
    if (valid_series < 1.0).any():
        invalid_sample = valid_series[valid_series < 1.0].head(3).tolist()
        raise ValueError(
            f"Column '{col}' contains values < 1.0 (sample: {invalid_sample}). "
            "Per protocol Section 5.2, TyreLife must be >= 1.0."
        )


def check_candidate_key_duplicates(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    """Check for duplicate rows on candidate key.

    If exact duplicate rows exist, drop duplicates and return count.
    If rows conflict on non-key columns, raise ValueError.

    Args:
        df: Input DataFrame.
        key_columns: List of columns defining candidate key. Defaults to KEY_COLUMNS.

    Returns:
        Tuple of (deduplicated DataFrame, number of duplicate rows dropped).

    Raises:
        ValueError: If conflicting values exist for identical candidate keys.
    """
    key = key_columns or KEY_COLUMNS
    missing_key_cols = [c for c in key if c not in df.columns]
    if missing_key_cols:
        raise ValueError(f"Missing candidate key columns in DataFrame: {missing_key_cols}")

    if df[key].isna().any().any():
        raise ValueError("Candidate key contains null values")

    dup_key_mask = df.duplicated(subset=key, keep=False)
    if not dup_key_mask.any():
        return df.copy(), 0

    distinct = df.drop_duplicates()
    conflicts = distinct.duplicated(subset=key, keep=False)
    if conflicts.any():
        sample = distinct.loc[conflicts, key].head(3).to_dict(orient="records")
        raise ValueError(
            f"Candidate key conflicts found in {int(conflicts.sum())} distinct rows: {sample}"
        )
    return distinct.copy(), len(df) - len(distinct)


def generate_row_id(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> pd.Series:
    """Generate stable, unique, deterministic row identifiers from candidate key columns.

    Format: {Year}_{EventName_slug}_{Session}_{Driver}_{LapNumber}
    e.g. '2022_Bahrain_Grand_Prix_R_LEC_2'

    Each row's lap number is formatted row-invariantly (int string if integer, else float string).

    Args:
        df: Input DataFrame.
        key_columns: List of candidate key columns.

    Returns:
        Series of row_id strings matching df.index.

    Raises:
        ValueError: If key columns are missing, contain nulls, or generated row_ids are not unique.
    """
    key = key_columns or KEY_COLUMNS
    for col in key:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' for row_id generation")

    if df[key].isna().any().any():
        raise ValueError("Cannot generate row_id with null key components")

    year_str = df["Year"].astype(str)
    event_str = df["EventName"].astype(str).str.replace(" ", "_")
    session_str = df["Session"].astype(str)
    driver_str = df["Driver"].astype(str)

    # LapNumber formatted row-invariantly
    lap_numeric = pd.to_numeric(df["LapNumber"], errors="raise")
    lap_str = lap_numeric.apply(lambda x: str(int(x)) if float(x).is_integer() else str(x))

    row_ids = year_str + "_" + event_str + "_" + session_str + "_" + driver_str + "_" + lap_str

    if row_ids.isna().any():
        raise ValueError("Generated row_ids contain NaN")
    if row_ids.duplicated().any():
        n_dups = row_ids.duplicated().sum()
        sample = row_ids[row_ids.duplicated()].head(3).tolist()
        raise ValueError(
            f"Generated row_ids are not unique: {n_dups} duplicates found. Sample: {sample}"
        )

    return row_ids


def assign_splits(
    clean_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    """Assign temporal split labels ('train', 'validation', 'test') to each row by Year.

    Args:
        clean_df: Filtered clean DataFrame with 'Year' column.
        config: Validated experiment configuration.

    Returns:
        Series of split labels aligned with clean_df index.

    Raises:
        ValueError: If Year column is missing, unassigned rows exist, or overlap occurs.
    """
    if "Year" not in clean_df.columns:
        raise ValueError("Missing 'Year' column for split assignment")

    split_cfg = config.get("split_protocol", {})
    if split_cfg.get("strategy") != "temporal_by_year":
        raise ValueError(f"Unsupported split strategy: {split_cfg.get('strategy')}")

    train_years = set(split_cfg.get("train", {}).get("years", []))
    val_years = set(split_cfg.get("validation", {}).get("years", []))
    test_cfg = split_cfg.get("final_test") or split_cfg.get("test", {})
    test_years = set(test_cfg.get("years", []))

    if not train_years or not val_years or not test_years:
        raise ValueError("Split years configuration incomplete in split_protocol")

    if train_years & val_years:
        raise ValueError(f"Year overlap between train and validation: {train_years & val_years}")
    if train_years & test_years:
        raise ValueError(f"Year overlap between train and test: {train_years & test_years}")
    if val_years & test_years:
        raise ValueError(f"Year overlap between validation and test: {val_years & test_years}")

    split_series = pd.Series(index=clean_df.index, dtype="object")
    split_series[clean_df["Year"].isin(train_years)] = "train"
    split_series[clean_df["Year"].isin(val_years)] = "validation"
    split_series[clean_df["Year"].isin(test_years)] = "test"

    unassigned_mask = split_series.isna()
    if unassigned_mask.any():
        n_unassigned = int(unassigned_mask.sum())
        unexpected_years = sorted(clean_df.loc[unassigned_mask, "Year"].unique())
        raise ValueError(
            f"{n_unassigned} rows could not be assigned to any split. Unexpected years: {unexpected_years}"
        )

    return split_series


def prepare_base_and_splits(
    raw_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Clean raw data, assign row_id and splits, and output base and split DataFrames.

    Args:
        raw_df: Raw DataFrame (89,980 rows from primary parquet).
        config: Experiment configuration dictionary.

    Returns:
        Tuple of:
          - base_df: Clean DataFrame containing row_id, keys, target, and allowed candidate features.
          - splits_df: DataFrame containing row_id and split ('train', 'validation', 'test').
          - stats: Dictionary recording step counts, missing counts (including Stint), and validation checks.

    Raises:
        ValueError: If filtering, schemas, duplicates, or target validity fail.
    """
    raw_rows = len(raw_df)

    # 1. Apply protocol scope and filters
    clean_df, exclusions = filter_with_counts(raw_df, config)
    clean_rows_after_filter = len(clean_df)

    # 2. Check and handle duplicates
    clean_df, dups_dropped = check_candidate_key_duplicates(clean_df, KEY_COLUMNS)
    clean_df = clean_df.reset_index(drop=True)

    # 3. Normalize pseudo-missing strings in Compound ('None', 'nan' -> NaN)
    pseudo_strings = config.get("preprocessing_and_pipeline", {}).get(
        "pseudo_missing_strings", list(PSEUDO_MISSING_STRINGS)
    )
    clean_df = normalize_pseudo_missing(clean_df, column="Compound", pseudo_strings=pseudo_strings)

    # Validate source values before conversion can hide invalid types.
    validate_target_column(clean_df, TARGET_COLUMN)
    validate_tyrelife_column(clean_df, "TyreLife")
    for col in ("FreshTyre", "IsAccurate"):
        if not pd.api.types.is_bool_dtype(clean_df[col]) or clean_df[col].isna().any():
            raise ValueError(f"Column '{col}' must be boolean and non-null")

    for col in ("LapNumber", "Stint"):
        values = clean_df[col].dropna()
        if not values.empty and (
            not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values).all()
        ):
            raise ValueError(f"Column '{col}' must contain finite numbers")

    # 4. Standardize dtypes
    clean_df["FreshTyre"] = clean_df["FreshTyre"].astype(bool)
    clean_df["IsAccurate"] = clean_df["IsAccurate"].astype(bool)
    clean_df["LapNumber"] = clean_df["LapNumber"].astype(float)
    clean_df["Stint"] = clean_df["Stint"].astype(float)
    clean_df["TyreLife"] = clean_df["TyreLife"].astype(float)
    clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(float)

    # 5. Generate row_id
    row_ids = generate_row_id(clean_df, KEY_COLUMNS)
    clean_df.insert(0, "row_id", row_ids)

    # 6. Assign splits
    split_series = assign_splits(clean_df, config)

    # 7. Construct base_df (only row_id, candidate keys, target, allowed candidates, tracking columns)
    features = config["features"]
    permitted = {
        "row_id",
        TARGET_COLUMN,
        *features["allowed_candidate_features"],
        *features["tracking_and_split_columns"],
    }
    if set(BASE_COLUMNS) != permitted or permitted.intersection(
        features["forbidden_features"] + features["pending_verification_features"]
    ):
        raise ValueError("Base schema conflicts with the feature roles in the protocol")
    base_df = clean_df[BASE_COLUMNS].copy()

    # 8. Construct splits_df
    splits_df = pd.DataFrame({"row_id": clean_df["row_id"].values, "split": split_series.values})

    # Alignment check
    if not (base_df["row_id"].values == splits_df["row_id"].values).all():
        raise ValueError("base_df and splits_df row_id alignment mismatch")

    # 9. Record detailed statistics
    train_mask = splits_df["split"] == "train"
    val_mask = splits_df["split"] == "validation"
    test_mask = splits_df["split"] == "test"

    train_base = base_df[train_mask]
    val_base = base_df[val_mask]
    test_base = base_df[test_mask]

    austrian_2022_mask = (base_df["Year"] == 2022) & (base_df["EventName"] == "Austrian Grand Prix")
    austrian_base = base_df[austrian_2022_mask]

    stats: dict[str, Any] = {
        "raw_rows": int(raw_rows),
        **exclusions,
        "rows_after_scope_filters": int(clean_rows_after_filter),
        "duplicate_rows_dropped": int(dups_dropped),
        "clean_rows": int(len(base_df)),
        "base_columns_count": int(base_df.shape[1]),
        "base_column_names": list(base_df.columns),
        "split_counts": {
            "train": int(train_mask.sum()),
            "validation": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "split_event_groups": {
            "train_distinct_events": int(train_base["EventName"].nunique()),
            "train_session_groups": int(
                train_base.groupby(["Year", "EventName", "Session"]).ngroups
            ),
            "val_distinct_events": int(val_base["EventName"].nunique()),
            "val_session_groups": int(val_base.groupby(["Year", "EventName", "Session"]).ngroups),
            "test_distinct_events": int(test_base["EventName"].nunique()),
            "test_session_groups": int(test_base.groupby(["Year", "EventName", "Session"]).ngroups),
        },
        "missing_counts": {
            "train_missing_compound": int(train_base["Compound"].isna().sum()),
            "train_missing_tyrelife": int(train_base["TyreLife"].isna().sum()),
            "train_missing_stint": int(train_base["Stint"].isna().sum()),
            "austrian_2022_total_rows": int(len(austrian_base)),
            "austrian_2022_missing_compound": int(austrian_base["Compound"].isna().sum()),
            "austrian_2022_missing_tyrelife": int(austrian_base["TyreLife"].isna().sum()),
            "austrian_2022_missing_stint": int(austrian_base["Stint"].isna().sum()),
        },
    }

    return base_df, splits_df, stats


def load_prepared_data(
    repo_root: Path | str | None = None,
    split: str | None = None,
    verify_hashes: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load prepared base dataset and split mappings, joined by row_id.

    Applies explicit dtypes, validates 1:1 row join, and resets index to 0..N-1
    when returning split subsets to prevent downstream pandas index misalignment.

    Args:
        repo_root: Root directory of repository. If None, detected from module location.
        split: Optional split filter: 'train', 'validation', 'test', or 'final_test'.
               ('final_test' is automatically mapped to 'test').
        verify_hashes: Whether to verify base.csv and splits.csv checksums against metadata.json.

    Returns:
        Tuple of (merged DataFrame with 'split' column and 0-based index, metadata dictionary).

    Raises:
        FileNotFoundError: If prepared data files or metadata do not exist.
        ValueError: If split name is unrecognized or checksum verification fails.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    processed_dir = root / "data" / "processed"
    base_file = processed_dir / "base.csv"
    splits_file = processed_dir / "splits.csv"
    meta_file = processed_dir / "metadata.json"

    for f in [base_file, splits_file, meta_file]:
        if not f.is_file():
            raise FileNotFoundError(
                f"Prepared file not found: {f}. Run 'uv run python -m scripts.prepare_data' first."
            )

    with meta_file.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)

    if verify_hashes:
        base_hash = sha256_file(base_file)
        splits_hash = sha256_file(splits_file)
        expected_base_hash = meta.get("processed_outputs", {}).get("base_csv", {}).get("sha256")
        expected_splits_hash = meta.get("processed_outputs", {}).get("splits_csv", {}).get("sha256")
        if (
            not expected_base_hash
            or not isinstance(expected_base_hash, str)
            or not expected_base_hash.strip()
        ):
            raise ValueError(f"Missing required 'base_csv' SHA-256 hash in {meta_file.name}")
        if (
            not expected_splits_hash
            or not isinstance(expected_splits_hash, str)
            or not expected_splits_hash.strip()
        ):
            raise ValueError(f"Missing required 'splits_csv' SHA-256 hash in {meta_file.name}")
        if base_hash != expected_base_hash:
            raise ValueError(
                f"base.csv SHA-256 mismatch: got {base_hash}, expected {expected_base_hash}"
            )
        if splits_hash != expected_splits_hash:
            raise ValueError(
                f"splits.csv SHA-256 mismatch: got {splits_hash}, expected {expected_splits_hash}"
            )

    base_df = pd.read_csv(base_file, dtype=BASE_DTYPES, keep_default_na=False, na_values=[""])
    splits_df = pd.read_csv(splits_file, dtype=SPLIT_DTYPES, keep_default_na=False, na_values=[""])

    if base_df["row_id"].isna().any() or splits_df["row_id"].isna().any():
        raise ValueError("Null row_id in base.csv or splits.csv")
    merged = base_df.merge(splits_df, on="row_id", how="inner", validate="1:1")
    if len(merged) != len(base_df) or len(merged) != len(splits_df):
        raise ValueError("row_id mismatch between base.csv and splits.csv")
    if not merged["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Unknown or missing split label in splits.csv")

    if split is not None:
        target_split = "test" if split == "final_test" else split
        valid_splits = {"train", "validation", "test"}
        if target_split not in valid_splits:
            raise ValueError(
                f"Unknown split '{split}'. Available: {sorted(valid_splits)} (or 'final_test')"
            )
        merged = merged[merged["split"] == target_split].reset_index(drop=True)

    return merged, meta
