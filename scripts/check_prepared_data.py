"""Verify saved F1-05 data against its source, protocol, metadata and optional release hashes."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from f1_lap_time.data_preparation import (
    BASE_COLUMNS,
    KEY_COLUMNS,
    SPLIT_COLUMNS,
    assign_splits,
    load_prepared_data,
    prepare_base_and_splits,
    validate_target_column,
    validate_tyrelife_column,
)
from f1_lap_time.protocol import load_experiment_config, read_verified_source, sha256_file
from scripts.prepare_data import CODE_FILES

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path("reports/data-preparation/checks.json")


# Observations locked in F1-03 section 6.1, before F1-05 preparation.
TRAIN_MISSING_COUNTS = {
    "train_missing_compound": 327,
    "train_missing_tyrelife": 428,
    "austrian_2022_total_rows": 1176,
    "austrian_2022_missing_compound": 259,
    "austrian_2022_missing_tyrelife": 307,
}


def run_checks(
    repo_root: Path, config_path: Path | None = None, reference: Path | None = None
) -> dict[str, Any]:
    """Record completed checks and return a failing report if an input cannot be verified."""
    checks = []
    hashes = {}
    summary = {}

    def compare(name, actual, expected):
        checks.append(
            {"name": name, "passed": actual == expected, "actual": actual, "expected": expected}
        )

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    try:
        root = repo_root.resolve()
        proc = root / "data/processed"
        paths = {
            "base_csv": proc / "base.csv",
            "splits_csv": proc / "splits.csv",
            "metadata_json": proc / "metadata.json",
        }
        for name, path in paths.items():
            hashes[f"{name}_sha256"] = sha256_file(path)
        cfg_path = (root / (config_path or "configs/experiment.json")).resolve()
        config = load_experiment_config(cfg_path)
        raw = read_verified_source(config, root)
        compare(
            "Verified raw dimensions",
            list(raw.shape),
            [config["data_source"]["raw_rows"], config["data_source"]["raw_columns"]],
        )
        if reference is not None:
            baseline = json.loads(
                (root / reference).read_text("utf-8"), parse_constant=invalid_constant
            )
            compare("Reference report passed", baseline["all_passed"], True)
            compare(
                "Release source SHA-256", config["data_source"]["sha256"], baseline["source_sha256"]
            )
            compare("Release config SHA-256", sha256_file(cfg_path), baseline["config_sha256"])
            for name in ("base_csv_sha256", "splits_csv_sha256"):
                compare(f"Release {name}", hashes[name], baseline["hashes"][name])

        meta = json.loads(
            paths["metadata_json"].read_text("utf-8"), parse_constant=invalid_constant
        )
        source = config["data_source"]
        compare(
            "Source identity",
            meta["data_source"],
            {
                "path": source["primary_file"],
                "sha256": source["sha256"],
                "version": source["version"],
                "source_doi": source["source_doi"],
                "raw_rows": len(raw),
                "raw_columns": len(raw.columns),
            },
        )
        provenance = meta["provenance"]
        compare("Config SHA-256", provenance["config_sha256"], sha256_file(cfg_path))
        compare("Config snapshot matches", provenance["config_snapshot"] == config, True)
        compare(
            "Config path matches", (root / provenance["config_path"]).resolve() == cfg_path, True
        )
        compare(
            "Preparation code SHA-256",
            provenance["code_files_sha256"],
            {name: sha256_file(root / name) for name in CODE_FILES},
        )
        for name, columns in (("base_csv", BASE_COLUMNS), ("splits_csv", SPLIT_COLUMNS)):
            info = meta["processed_outputs"][name]
            output = paths[name]
            compare(f"{name} header", pd.read_csv(output, nrows=0).columns.tolist(), columns)
            compare(
                f"{name} metadata",
                info,
                {
                    "path": output.relative_to(root).as_posix(),
                    "sha256": hashes[f"{name}_sha256"],
                    "rows": config["scope_and_filters"]["expected_clean_rows"],
                    "columns": len(columns),
                    "column_names": columns,
                },
            )

        # The loader checks both hashes and the bijection of the saved row identifiers.
        saved, _ = load_prepared_data(root)
        compare("Saved row count", len(saved), config["scope_and_filters"]["expected_clean_rows"])
        compare("Duplicate candidate keys", int(saved.duplicated(KEY_COLUMNS).sum()), 0)
        validate_target_column(saved)
        validate_tyrelife_column(saved)
        for col in ("Stint", "LapNumber"):
            compare(
                f"Non-finite {col} values among observed laps",
                int((~np.isfinite(saved[col].dropna())).sum()),
                0,
            )
        compare(
            "Invalid split assignments",
            int((saved["split"] != assign_splits(saved, config)).sum()),
            0,
        )
        compare(
            "Event groups assigned to multiple splits",
            int((saved.groupby(["Year", "EventName", "Session"])["split"].nunique() > 1).sum()),
            0,
        )
        pseudo = config["preprocessing_and_pipeline"]["pseudo_missing_strings"]
        compare(
            "Unnormalized Compound values", int(saved["Compound"].str.strip().isin(pseudo).sum()), 0
        )

        expected, _, stats = prepare_base_and_splits(raw, config)
        # Compare by key so a reorder cannot hide an X/y shift or create a false failure.
        actual = saved.set_index("row_id")
        expected = expected.set_index("row_id")
        different_ids = actual.index.symmetric_difference(expected.index)
        compare("Row IDs differing from verified source", len(different_ids), 0)
        if different_ids.empty:
            actual = actual.reindex(expected.index)
            for col in expected:
                same = actual[col].eq(expected[col]).fillna(False) | (
                    actual[col].isna() & expected[col].isna()
                )
                compare(f"{col} values differing from verified source", int((~same).sum()), 0)

        steps = {
            key: stats[key]
            for key in (
                "raw_rows",
                "excluded_session",
                "excluded_preseason",
                "excluded_sc_vsc",
                "rows_after_scope_filters",
                "duplicate_rows_dropped",
                "clean_rows",
            )
        }
        compare("Filtering and deduplication counts", meta["step_counts"], steps)
        scope = config["scope_and_filters"]
        compare(
            "Pre-Season exclusions",
            steps["excluded_preseason"],
            scope["preseason_filter"]["rows_excluded"],
        )
        compare(
            "SC/VSC exclusions after Pre-Season",
            steps["excluded_sc_vsc"],
            scope["safety_car_filter"]["rows_excluded_outside_preseason"],
        )
        compare("Metadata split counts", meta["split_counts"], stats["split_counts"])
        compare("Metadata event counts", meta["split_event_groups"], stats["split_event_groups"])
        for split, key in (
            ("train", "train"),
            ("validation", "validation"),
            ("test", "final_test"),
        ):
            subset = saved.loc[saved["split"] == split]
            rule = config["split_protocol"][key]
            compare(f"{split} rows", len(subset), rule["expected_rows"])
            compare(
                f"{split} years", sorted(subset["Year"].unique().tolist()), sorted(rule["years"])
            )
            compare(
                f"{split} event names", subset["EventName"].nunique(), rule["distinct_event_names"]
            )
            compare(
                f"{split} event groups",
                subset.groupby(["Year", "EventName", "Session"]).ngroups,
                rule["event_session_groups"],
            )

        missing = stats["missing_counts"]
        compare("Train missing counts in metadata", meta["missing_counts"], missing)
        for name, count in TRAIN_MISSING_COUNTS.items():
            compare(name, missing[name], count)
        summary = {
            "step_counts": steps,
            "split_counts": stats["split_counts"],
            "source_sha256": source["sha256"],
            "config_sha256": sha256_file(cfg_path),
            "preparation_commit": provenance["git_commit_hash"],
            "preparation_dirty": provenance["git_dirty_flag"],
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        OverflowError,
    ) as exc:
        checks.append({"name": "Read and validate inputs", "passed": False, "details": str(exc)})
    failures = sum(not check["passed"] for check in checks)
    return {
        "task": "F1-05",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "all_passed": failures == 0,
        "total_checks": len(checks),
        "passed_checks": len(checks) - failures,
        "failed_checks": failures,
        "hashes": hashes,
        **summary,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--reference", type=Path, help="Compare CSV hashes with a prior passing report"
    )
    parser.add_argument(
        "--report", type=Path, default=REPORT_PATH, help="Where to write the result"
    )
    args = parser.parse_args()
    output = ROOT / args.report
    if args.reference and (ROOT / args.reference).resolve() == output.resolve():
        parser.error("--report and --reference must refer to different files")
    result = run_checks(ROOT, args.config, args.reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        tmp = Path(directory) / "checks.json"
        tmp.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        tmp.replace(output)
    print(f"{result['passed_checks']}/{result['total_checks']} checks passed. Report: {output}")
    for check in result["checks"]:
        if not check["passed"]:
            print(f"FAIL {check['name']}: {check.get('details') or check.get('actual')}")
    return int(not result["all_passed"])


if __name__ == "__main__":
    sys.exit(main())
