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
from tempfile import TemporaryDirectory

from f1_lap_time.data_preparation import prepare_base_and_splits
from f1_lap_time.protocol import load_experiment_config, read_verified_source, sha256_file

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
    repo_root = repo_root.resolve()
    cfg_path = (repo_root / (config_path or "configs/experiment.json")).resolve()
    cfg_name = (
        cfg_path.relative_to(repo_root).as_posix()
        if cfg_path.is_relative_to(repo_root)
        else cfg_path.as_posix()
    )
    config = load_experiment_config(cfg_path)
    commit, dirty = get_git_info(repo_root)
    raw_rel = config["data_source"]["primary_file"]
    raw_df = read_verified_source(config, repo_root)
    print(f"Loaded raw dataset: {len(raw_df):,} rows, {raw_df.shape[1]} columns")

    # 2. Run core preparation
    print("Executing protocol filters, duplicate checking, and normalization ...")
    base_df, splits_df, stats = prepare_base_and_splits(raw_df, config)

    # 3. Output paths
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "config_path": cfg_name,
        "config_sha256": sha256_file(cfg_path),
        "config_snapshot": config,
        "git_commit_hash": commit,
        "git_dirty_flag": dirty,
        "code_files_sha256": {rel: sha256_file(repo_root / rel) for rel in CODE_FILES},
    }
    metadata = {
        "task": "F1-05",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "command": COMMAND if config_path is None else f'{COMMAND} --config "{cfg_name}"',
        "python_version": sys.version,
        "data_source": {
            "path": raw_rel,
            "sha256": config["data_source"]["sha256"],
            "source_doi": config["data_source"]["source_doi"],
            "version": config["data_source"]["version"],
            "raw_rows": len(raw_df),
            "raw_columns": len(raw_df.columns),
        },
        "provenance": provenance,
        "processed_outputs": {},
        "step_counts": {
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
        },
        "split_counts": stats["split_counts"],
        "split_event_groups": stats["split_event_groups"],
        "missing_counts": stats["missing_counts"],
    }

    # Finish serialization and hashing before publishing. Publish the verifying metadata last.
    with TemporaryDirectory(dir=processed_dir) as directory:
        staged = Path(directory)
        for name, frame in (("base", base_df), ("splits", splits_df)):
            filename = f"{name}.csv"
            frame.to_csv(staged / filename, index=False, lineterminator="\n")
            metadata["processed_outputs"][f"{name}_csv"] = {
                "path": f"data/processed/{filename}",
                "sha256": sha256_file(staged / filename),
                "rows": len(frame),
                "columns": len(frame.columns),
                "column_names": list(frame.columns),
            }
        (staged / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for filename in ("base.csv", "splits.csv", "metadata.json"):
            (staged / filename).replace(processed_dir / filename)

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
        print(f"Base CSV SHA-256:   {meta['processed_outputs']['base_csv']['sha256']}")
        print(f"Splits CSV SHA-256: {meta['processed_outputs']['splits_csv']['sha256']}")
        print("==================================================")
        return 0
    except Exception as exc:
        print(f"\nERROR during data preparation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
