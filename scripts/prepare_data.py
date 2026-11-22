"""Prepare clean base dataset and split mappings for F1 Lap Time Regression (F1-05).

Executes protocol filters, duplicate checking, pseudo-missing normalization,
stable row_id generation, and outputs:
- data/processed/base.csv
- data/processed/splits.csv
- data/processed/metadata.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from f1_lap_time.data_preparation import prepare_base_and_splits
from f1_lap_time.protocol import load_experiment_config, sha256_file
from scripts.download_data import verify_file

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "uv run python -m scripts.prepare_data"
CODE_FILES = (
    "scripts/prepare_data.py",
    "src/f1_lap_time/data_preparation.py",
    "src/f1_lap_time/protocol.py",
)


def get_git_info(repo_root: Path) -> tuple[str | None, bool | None]:
    """Retrieve current Git commit hash and dirty status."""
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
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def run_data_preparation(repo_root: Path, config_path: Path | None = None) -> dict:
    """Run data preparation pipeline and write outputs to data/processed/."""
    cfg_path = config_path or (repo_root / "configs" / "experiment.json")
    config = load_experiment_config(cfg_path)

    # 1. Verify raw data source against manifest and config
    raw_rel = config["data_source"]["primary_file"]
    raw_path = repo_root / raw_rel
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    manifest_path = repo_root / "data" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found at: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    selected = manifest.get("selected_file", {})
    verify_file(raw_path, selected)

    raw_hash = sha256_file(raw_path)
    if raw_hash != config["data_source"]["sha256"]:
        raise ValueError(f"Raw data hash mismatch: {raw_hash} != {config['data_source']['sha256']}")

    print(f"Reading raw dataset from: {raw_rel} ...")
    raw_df = pd.read_parquet(raw_path)
    print(f"Loaded raw dataset: {len(raw_df):,} rows, {raw_df.shape[1]} columns")

    # 2. Run core preparation
    print("Executing protocol filters, duplicate checking, and normalization ...")
    base_df, splits_df, stats = prepare_base_and_splits(raw_df, config)

    # 3. Output paths
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    base_path = processed_dir / "base.csv"
    splits_path = processed_dir / "splits.csv"
    meta_path = processed_dir / "metadata.json"

    # 4. Write CSV files
    print(
        f"Writing base dataset ({len(base_df):,} rows, {base_df.shape[1]} columns) to: {base_path.name} ..."
    )
    base_df.to_csv(base_path, index=False, lineterminator="\n")

    print(f"Writing split mappings ({len(splits_df):,} rows) to: {splits_path.name} ...")
    splits_df.to_csv(splits_path, index=False, lineterminator="\n")

    # 5. Compute output checksums
    base_hash = sha256_file(base_path)
    splits_hash = sha256_file(splits_path)

    # 6. Gather provenance metadata
    commit, dirty = get_git_info(repo_root)
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    code_hashes = {
        rel: sha256_file(repo_root / rel) for rel in CODE_FILES if (repo_root / rel).is_file()
    }

    metadata = {
        "task": "F1-05 — Shared Data Preparation",
        "timestamp_utc": timestamp_utc,
        "command": COMMAND,
        "python_version": sys.version,
        "data_source": {
            "path": raw_rel,
            "sha256": raw_hash,
            "source_doi": config["data_source"]["source_doi"],
            "version": config["data_source"]["version"],
            "raw_rows": stats["raw_rows"],
            "raw_columns": config["data_source"]["raw_columns"],
        },
        "provenance": {
            "config_path": cfg_path.relative_to(repo_root).as_posix(),
            "config_sha256": sha256_file(cfg_path),
            "git_commit_hash": commit,
            "git_dirty_flag": dirty,
            "code_files_sha256": code_hashes,
        },
        "processed_outputs": {
            "base_csv": {
                "path": base_path.relative_to(repo_root).as_posix(),
                "sha256": base_hash,
                "rows": int(len(base_df)),
                "columns": int(base_df.shape[1]),
                "column_names": list(base_df.columns),
            },
            "splits_csv": {
                "path": splits_path.relative_to(repo_root).as_posix(),
                "sha256": splits_hash,
                "rows": int(len(splits_df)),
                "columns": int(splits_df.shape[1]),
                "column_names": list(splits_df.columns),
            },
        },
        "step_counts": {
            "raw_rows": stats["raw_rows"],
            "rows_after_scope_filters": stats["rows_after_scope_filters"],
            "duplicate_rows_dropped": stats["duplicate_rows_dropped"],
            "clean_rows": stats["clean_rows"],
        },
        "split_counts": stats["split_counts"],
        "split_event_groups": stats["split_event_groups"],
        "missing_counts": stats["missing_counts"],
        "target_stats": stats["target_stats"],
    }

    print(f"Writing metadata to: {meta_path.name} ...")
    with meta_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare clean base data and split mappings (F1-05)"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to experiment.json")
    args = parser.parse_args()

    try:
        meta = run_data_preparation(ROOT, args.config)
        print("\n==================================================")
        print("F1-05 Data Preparation Completed Successfully")
        print("==================================================")
        print(f"Raw rows:        {meta['step_counts']['raw_rows']:,}")
        print(f"Clean rows:      {meta['step_counts']['clean_rows']:,}")
        print(f"Train rows:      {meta['split_counts']['train']:,} (2022-2023)")
        print(f"Validation rows: {meta['split_counts']['validation']:,} (2024)")
        print(f"Test rows:       {meta['split_counts']['test']:,} (2025)")
        print(f"Train missing Compound: {meta['missing_counts']['train_missing_compound']}")
        print(f"Train missing TyreLife: {meta['missing_counts']['train_missing_tyrelife']}")
        print(
            f"Train missing Stint:    {meta['missing_counts']['train_missing_stint']} "
            f"(Test missing Stint: {meta['missing_counts']['test_missing_stint']} - Imputer required for F1-06!)"
        )
        print(f"Base CSV SHA-256:   {meta['processed_outputs']['base_csv']['sha256']}")
        print(f"Splits CSV SHA-256: {meta['processed_outputs']['splits_csv']['sha256']}")
        print("==================================================")
        return 0
    except Exception as exc:
        print(f"\nERROR during data preparation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
