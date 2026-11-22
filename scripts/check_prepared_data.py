"""Verify prepared base dataset and split mappings against protocol acceptance criteria (F1-05).

Saves structured verification report to reports/data-preparation/checks.json.
Exits 0 if all checks pass, 1 if any check fails.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from f1_lap_time.data_preparation import (
    BASE_COLUMNS,
    KEY_COLUMNS,
    PSEUDO_MISSING_STRINGS,
    SPLIT_COLUMNS,
    TARGET_COLUMN,
)
from f1_lap_time.protocol import load_experiment_config, sha256_file
from scripts.download_data import verify_file

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "data-preparation" / "checks.json"


def to_serializable(val: Any) -> Any:
    """Convert numpy and non-serializable objects to native Python types."""
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (list, tuple, set)):
        return [to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    return val


def run_checks(repo_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Execute all acceptance checks on prepared data outputs."""
    cfg_path = config_path or (repo_root / "configs" / "experiment.json")
    config = load_experiment_config(cfg_path)

    checks: list[dict[str, Any]] = []

    def record_check(
        category: str,
        name: str,
        passed: bool,
        details: str,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        checks.append(
            {
                "category": category,
                "name": name,
                "passed": bool(passed),
                "details": str(details),
                "expected": to_serializable(expected),
                "actual": to_serializable(actual),
            }
        )

    # 1. Source and reproducibility
    raw_rel = config["data_source"]["primary_file"]
    raw_path = repo_root / raw_rel
    manifest_path = repo_root / "data" / "manifest.json"

    raw_exists = raw_path.is_file()
    record_check("Nguồn và khả năng tái tạo", "Raw file exists", raw_exists, str(raw_rel))

    if raw_exists and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        try:
            verify_file(raw_path, manifest["selected_file"])
            raw_hash_ok = True
        except ValueError:
            raw_hash_ok = False
        actual_raw_hash = sha256_file(raw_path)
        expected_raw_hash = config["data_source"]["sha256"]
        record_check(
            "Nguồn và khả năng tái tạo",
            "Raw file SHA-256 matches manifest and config",
            actual_raw_hash == expected_raw_hash and raw_hash_ok,
            f"Hash: {actual_raw_hash}",
            expected=expected_raw_hash,
            actual=actual_raw_hash,
        )

    # Check processed files exist
    processed_dir = repo_root / "data" / "processed"
    base_file = processed_dir / "base.csv"
    splits_file = processed_dir / "splits.csv"
    meta_file = processed_dir / "metadata.json"

    base_exists = base_file.is_file()
    splits_exists = splits_file.is_file()
    meta_exists = meta_file.is_file()

    record_check(
        "Nguồn và khả năng tái tạo",
        "base.csv exists",
        base_exists,
        str(base_file.relative_to(repo_root)),
    )
    record_check(
        "Nguồn và khả năng tái tạo",
        "splits.csv exists",
        splits_exists,
        str(splits_file.relative_to(repo_root)),
    )
    record_check(
        "Nguồn và khả năng tái tạo",
        "metadata.json exists",
        meta_exists,
        str(meta_file.relative_to(repo_root)),
    )

    if not (base_exists and splits_exists and meta_exists):
        return {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "all_passed": False,
            "total_checks": len(checks),
            "passed_checks": sum(1 for c in checks if c["passed"]),
            "failed_checks": sum(1 for c in checks if not c["passed"]),
            "checks": checks,
        }

    with meta_file.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Verify hashes against metadata
    actual_base_hash = sha256_file(base_file)
    actual_splits_hash = sha256_file(splits_file)
    expected_base_hash = meta.get("processed_outputs", {}).get("base_csv", {}).get("sha256")
    expected_splits_hash = meta.get("processed_outputs", {}).get("splits_csv", {}).get("sha256")

    record_check(
        "Nguồn và khả năng tái tạo",
        "base.csv SHA-256 matches metadata",
        actual_base_hash == expected_base_hash,
        f"actual: {actual_base_hash}",
        expected=expected_base_hash,
        actual=actual_base_hash,
    )
    record_check(
        "Nguồn và khả năng tái tạo",
        "splits.csv SHA-256 matches metadata",
        actual_splits_hash == expected_splits_hash,
        f"actual: {actual_splits_hash}",
        expected=expected_splits_hash,
        actual=actual_splits_hash,
    )

    # Verify config SHA-256 in metadata against current experiment.json
    meta_cfg_hash = meta.get("provenance", {}).get("config_sha256")
    current_cfg_hash = sha256_file(cfg_path)
    record_check(
        "Nguồn và khả năng tái tạo",
        "metadata config_sha256 matches current experiment.json",
        meta_cfg_hash == current_cfg_hash and bool(meta_cfg_hash),
        f"metadata: {meta_cfg_hash}, current: {current_cfg_hash}",
        expected=current_cfg_hash,
        actual=meta_cfg_hash,
    )

    # Read datasets
    base_df = pd.read_csv(base_file)
    splits_df = pd.read_csv(splits_file)

    # 2. Schema and columns
    expected_base_cols = BASE_COLUMNS
    actual_base_cols = list(base_df.columns)
    record_check(
        "Schema và ý nghĩa cột",
        "base.csv columns match exact allowed schema",
        actual_base_cols == expected_base_cols,
        f"Total columns: {len(actual_base_cols)}",
        expected=expected_base_cols,
        actual=actual_base_cols,
    )

    actual_splits_cols = list(splits_df.columns)
    record_check(
        "Schema và ý nghĩa cột",
        "splits.csv columns match schema",
        actual_splits_cols == SPLIT_COLUMNS,
        f"Columns: {actual_splits_cols}",
        expected=SPLIT_COLUMNS,
        actual=actual_splits_cols,
    )

    # Column dtypes check
    fresh_tyre_bool = base_df["FreshTyre"].isin([True, False, 1, 0, "True", "False"]).all()
    is_accurate_bool = base_df["IsAccurate"].isin([True, False, 1, 0, "True", "False"]).all()
    numeric_lap = pd.to_numeric(base_df["LapNumber"], errors="coerce").notna().all()
    numeric_stint = pd.to_numeric(base_df["Stint"].dropna(), errors="coerce").notna().all()
    numeric_tyrelife = pd.to_numeric(base_df["TyreLife"].dropna(), errors="coerce").notna().all()
    numeric_target = pd.to_numeric(base_df[TARGET_COLUMN], errors="coerce").notna().all()

    record_check(
        "Schema và ý nghĩa cột",
        "FreshTyre is boolean",
        fresh_tyre_bool,
        f"dtype: {base_df['FreshTyre'].dtype}",
    )
    record_check(
        "Schema và ý nghĩa cột",
        "IsAccurate is boolean",
        is_accurate_bool,
        f"dtype: {base_df['IsAccurate'].dtype}",
    )
    record_check(
        "Schema và ý nghĩa cột",
        "LapNumber is numeric",
        numeric_lap,
        f"dtype: {base_df['LapNumber'].dtype}",
    )
    record_check(
        "Schema và ý nghĩa cột",
        "Stint is numeric",
        numeric_stint,
        f"dtype: {base_df['Stint'].dtype}",
    )
    tyrelife_series = base_df["TyreLife"].dropna()
    numeric_tyrelife = (
        pd.to_numeric(tyrelife_series, errors="coerce").notna().all()
        and not np.isinf(tyrelife_series).any()
        and (tyrelife_series >= 1.0).all()
    )
    record_check(
        "Schema và ý nghĩa cột",
        "TyreLife (when present) is numeric, finite and >= 1.0 per protocol",
        numeric_tyrelife,
        f"dtype: {base_df['TyreLife'].dtype}, min: {float(tyrelife_series.min()) if not tyrelife_series.empty else 'N/A'}",
        expected=">= 1.0 and finite",
        actual=float(tyrelife_series.min()) if not tyrelife_series.empty else None,
    )
    record_check(
        "Schema và ý nghĩa cột",
        "LapTimeSec is numeric",
        numeric_target,
        f"dtype: {base_df[TARGET_COLUMN].dtype}",
    )

    # 3. Target and value validity
    target = base_df[TARGET_COLUMN]
    has_no_target_null = not target.isna().any()
    has_no_target_inf = not np.isinf(target).any()
    target_positive = (target > 0).all()
    target_min = float(target.min())
    target_max = float(target.max())

    record_check(
        "Target và giá trị lỗi",
        "Target contains no NaN values",
        has_no_target_null,
        f"Missing: {target.isna().sum()}",
        expected=0,
        actual=int(target.isna().sum()),
    )
    record_check(
        "Target và giá trị lỗi",
        "Target contains no infinite values",
        has_no_target_inf,
        "Finite check",
    )
    record_check(
        "Target và giá trị lỗi",
        "Target is strictly positive",
        target_positive,
        f"Min target: {target_min:.3f}s",
        expected="> 0",
        actual=target_min,
    )
    record_check(
        "Target và giá trị lỗi",
        "Target range aligns with protocol observation (67.012s - 149.321s)",
        target_min == 67.012 and target_max == 149.321,
        f"Observed: [{target_min}, {target_max}]",
        expected=[67.012, 149.321],
        actual=[target_min, target_max],
    )

    # 4. Duplicates and cleaning
    expected_clean_rows = config["scope_and_filters"]["expected_clean_rows"]
    actual_rows = len(base_df)
    record_check(
        "Dòng trùng và làm sạch",
        "Clean rows count matches protocol expected_clean_rows",
        actual_rows == expected_clean_rows,
        f"{actual_rows:,} rows",
        expected=expected_clean_rows,
        actual=actual_rows,
    )

    row_ids = base_df["row_id"]
    row_ids_no_null = not row_ids.isna().any()
    row_ids_unique = not row_ids.duplicated().any()
    record_check(
        "Dòng trùng và làm sạch",
        "row_id has no nulls",
        row_ids_no_null,
        f"Nulls: {row_ids.isna().sum()}",
        expected=0,
        actual=int(row_ids.isna().sum()),
    )
    record_check(
        "Dòng trùng và làm sạch",
        "row_id is 100% unique across all clean rows",
        row_ids_unique,
        f"Duplicates: {row_ids.duplicated().sum()}",
        expected=0,
        actual=int(row_ids.duplicated().sum()),
    )

    key_unique = not base_df.duplicated(subset=KEY_COLUMNS).any()
    record_check(
        "Dòng trùng và làm sạch",
        "Candidate key (Year, EventName, Session, Driver, LapNumber) is unique",
        key_unique,
        f"Duplicates: {base_df.duplicated(subset=KEY_COLUMNS).sum()}",
        expected=0,
        actual=int(base_df.duplicated(subset=KEY_COLUMNS).sum()),
    )

    # 5. Split validity and alignment
    splits_len = len(splits_df)
    record_check(
        "Split",
        "splits.csv has exact same row count as base.csv",
        splits_len == actual_rows,
        f"splits: {splits_len:,}, base: {actual_rows:,}",
        expected=actual_rows,
        actual=splits_len,
    )

    alignment = (base_df["row_id"].values == splits_df["row_id"].values).all()
    record_check(
        "Split",
        "base.csv and splits.csv row_id order is identical 1-to-1",
        alignment,
        "Order and values match",
    )

    split_values = set(splits_df["split"].unique())
    record_check(
        "Split",
        "Split labels are strictly 'train', 'validation', 'test'",
        split_values == {"train", "validation", "test"},
        f"Observed splits: {sorted(split_values)}",
        expected=["test", "train", "validation"],
        actual=sorted(split_values),
    )

    train_count = int((splits_df["split"] == "train").sum())
    val_count = int((splits_df["split"] == "validation").sum())
    test_count = int((splits_df["split"] == "test").sum())
    exp_train = config["split_protocol"]["train"]["expected_rows"]
    exp_val = config["split_protocol"]["validation"]["expected_rows"]
    exp_test = config["split_protocol"]["final_test"]["expected_rows"]

    record_check(
        "Split",
        "Train rows count matches protocol",
        train_count == exp_train,
        f"Train: {train_count:,}",
        expected=exp_train,
        actual=train_count,
    )
    record_check(
        "Split",
        "Validation rows count matches protocol",
        val_count == exp_val,
        f"Validation: {val_count:,}",
        expected=exp_val,
        actual=val_count,
    )
    record_check(
        "Split",
        "Test rows count matches protocol",
        test_count == exp_test,
        f"Test: {test_count:,}",
        expected=exp_test,
        actual=test_count,
    )

    # Temporal exclusivity check
    train_years = set(base_df.loc[splits_df["split"] == "train", "Year"].unique())
    val_years = set(base_df.loc[splits_df["split"] == "validation", "Year"].unique())
    test_years = set(base_df.loc[splits_df["split"] == "test", "Year"].unique())
    record_check(
        "Split",
        "Train years match protocol (2022, 2023)",
        train_years == {2022, 2023},
        f"Observed: {sorted(train_years)}",
        expected=[2022, 2023],
        actual=sorted(train_years),
    )
    record_check(
        "Split",
        "Validation years match protocol (2024)",
        val_years == {2024},
        f"Observed: {sorted(val_years)}",
        expected=[2024],
        actual=sorted(val_years),
    )
    record_check(
        "Split",
        "Test years match protocol (2025)",
        test_years == {2025},
        f"Observed: {sorted(test_years)}",
        expected=[2025],
        actual=sorted(test_years),
    )

    # 6. Leakage prevention
    forbidden_features = config.get("features", {}).get("forbidden_features", [])
    pending_weather = config.get("features", {}).get("pending_verification_features", [])
    leaked_forbidden = [c for c in forbidden_features if c in base_df.columns]
    leaked_weather = [c for c in pending_weather if c in base_df.columns]

    record_check(
        "Leakage",
        "No forbidden features present in base.csv",
        len(leaked_forbidden) == 0,
        f"Leaked: {leaked_forbidden}",
        expected=0,
        actual=len(leaked_forbidden),
    )
    record_check(
        "Leakage",
        "No pending weather features present in base.csv",
        len(leaked_weather) == 0,
        f"Leaked: {leaked_weather}",
        expected=0,
        actual=len(leaked_weather),
    )

    # 7. Missing and category handling
    pseudo_in_compound = base_df["Compound"].isin(list(PSEUDO_MISSING_STRINGS)).sum()
    record_check(
        "Missing và category",
        "Compound contains zero pseudo-missing strings ('None', 'nan')",
        pseudo_in_compound == 0,
        f"Found: {pseudo_in_compound}",
        expected=0,
        actual=int(pseudo_in_compound),
    )

    train_mask = splits_df["split"] == "train"
    train_missing_compound = int(base_df.loc[train_mask, "Compound"].isna().sum())
    train_missing_tyrelife = int(base_df.loc[train_mask, "TyreLife"].isna().sum())

    record_check(
        "Missing và category",
        "Train missing Compound count matches protocol (327)",
        train_missing_compound == 327,
        f"Missing: {train_missing_compound}",
        expected=327,
        actual=train_missing_compound,
    )
    record_check(
        "Missing và category",
        "Train missing TyreLife count matches protocol (428)",
        train_missing_tyrelife == 428,
        f"Missing: {train_missing_tyrelife}",
        expected=428,
        actual=train_missing_tyrelife,
    )

    austrian_2022_mask = (base_df["Year"] == 2022) & (base_df["EventName"] == "Austrian Grand Prix")
    aus_compound_missing = int(base_df.loc[austrian_2022_mask, "Compound"].isna().sum())
    aus_tyrelife_missing = int(base_df.loc[austrian_2022_mask, "TyreLife"].isna().sum())
    aus_total_rows = int(austrian_2022_mask.sum())

    record_check(
        "Missing và category",
        "Austrian GP 2022 total rows match protocol (1,176)",
        aus_total_rows == 1176,
        f"Rows: {aus_total_rows}",
        expected=1176,
        actual=aus_total_rows,
    )
    record_check(
        "Missing và category",
        "Austrian GP 2022 missing Compound matches protocol (259)",
        aus_compound_missing == 259,
        f"Missing: {aus_compound_missing}",
        expected=259,
        actual=aus_compound_missing,
    )
    record_check(
        "Missing và category",
        "Austrian GP 2022 missing TyreLife matches protocol (307)",
        aus_tyrelife_missing == 307,
        f"Missing: {aus_tyrelife_missing}",
        expected=307,
        actual=aus_tyrelife_missing,
    )

    # Stint missing checks across splits (Crucial for F1-06 Imputer planning)
    val_mask = splits_df["split"] == "validation"
    test_mask = splits_df["split"] == "test"
    train_stint_missing = int(base_df.loc[train_mask, "Stint"].isna().sum())
    val_stint_missing = int(base_df.loc[val_mask, "Stint"].isna().sum())
    test_stint_missing = int(base_df.loc[test_mask, "Stint"].isna().sum())

    record_check(
        "Missing và category",
        "Train missing Stint count is strictly 0",
        train_stint_missing == 0,
        f"Missing: {train_stint_missing}",
        expected=0,
        actual=train_stint_missing,
    )
    record_check(
        "Missing và category",
        "Validation missing Stint count is strictly 0",
        val_stint_missing == 0,
        f"Missing: {val_stint_missing}",
        expected=0,
        actual=val_stint_missing,
    )
    record_check(
        "Missing và category",
        "Test missing Stint count matches expected raw observations (382)",
        test_stint_missing == 382,
        f"Missing: {test_stint_missing}",
        expected=382,
        actual=test_stint_missing,
    )

    # 8. Index alignment and 0-based continuity via load_prepared_data
    from f1_lap_time.data_preparation import load_prepared_data

    loaded_val, _ = load_prepared_data(repo_root, split="validation")
    loaded_test, _ = load_prepared_data(repo_root, split="test")

    val_index_zero_based = (loaded_val.index[0] == 0) and (
        loaded_val.index[-1] == len(loaded_val) - 1
    )
    test_index_zero_based = (loaded_test.index[0] == 0) and (
        loaded_test.index[-1] == len(loaded_test) - 1
    )

    record_check(
        "Split",
        "load_prepared_data(split='validation') resets index to contiguous 0..N-1",
        val_index_zero_based,
        f"Range: [{loaded_val.index[0]}..{loaded_val.index[-1]}], len: {len(loaded_val)}",
        expected=[0, len(loaded_val) - 1],
        actual=[loaded_val.index[0], loaded_val.index[-1]],
    )
    record_check(
        "Split",
        "load_prepared_data(split='test') resets index to contiguous 0..N-1",
        test_index_zero_based,
        f"Range: [{loaded_test.index[0]}..{loaded_test.index[-1]}], len: {len(loaded_test)}",
        expected=[0, len(loaded_test) - 1],
        actual=[loaded_test.index[0], loaded_test.index[-1]],
    )

    all_passed = all(c["passed"] for c in checks)
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": "F1-05 — Shared Data Preparation Verification",
        "all_passed": all_passed,
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c["passed"]),
        "failed_checks": sum(1 for c in checks if not c["passed"]),
        "hashes": {
            "base_csv_sha256": actual_base_hash,
            "splits_csv_sha256": actual_splits_hash,
            "metadata_json_sha256": sha256_file(meta_file),
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check prepared data outputs against protocol acceptance criteria"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to experiment.json")
    args = parser.parse_args()

    results = run_checks(ROOT, args.config)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print("\n==================================================")
    print("F1-05 Acceptance Checks Summary")
    print("==================================================")
    print(f"Total checks:  {results['total_checks']}")
    print(f"Passed:        {results['passed_checks']}")
    print(f"Failed:        {results['failed_checks']}")
    print(f"Report saved:  {REPORT_PATH.relative_to(ROOT)}")
    print("==================================================")

    if not results["all_passed"]:
        print("\nFAILED CHECKS:")
        for c in results["checks"]:
            if not c["passed"]:
                print(f" [FAIL] [{c['category']}] {c['name']}: {c['details']}")
        return 1

    print("\nALL ACCEPTANCE CHECKS PASSED (Ready for F1-06 handover)!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
