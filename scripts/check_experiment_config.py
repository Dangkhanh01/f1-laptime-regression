"""CLI script to validate configs/experiment.json and verify consistency against raw dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from f1_lap_time.protocol import (
    load_experiment_config,
    verify_config_against_raw_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate configs/experiment.json structure and consistency against raw dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment JSON configuration file (default: configs/experiment.json).",
    )
    parser.add_argument(
        "--check-structure-only",
        action="store_true",
        help="Only validate JSON schema/structure without verifying against raw dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        args.config if args.config is not None else repo_root / "configs" / "experiment.json"
    )

    print(f"Loading experiment configuration from: {config_path}")
    try:
        config = load_experiment_config(config_path)
    except Exception as exc:
        print(f"ERROR: Configuration validation failed: {exc}", file=sys.stderr)
        return 1

    print("OK: Config JSON structure, keys, and logical consistency are valid.")

    if args.check_structure_only:
        print(
            "WARNING: --check-structure-only flag set. Skipping verification against raw dataset."
        )
        print(
            "NOTICE: Dataset-level constraints (row counts, columns, filters) HAVE NOT been verified."
        )
        return 0

    print("Verifying configuration rules against raw dataset...")
    try:
        results = verify_config_against_raw_data(config, repo_root)
    except FileNotFoundError as fnf:
        print(
            f"ERROR: Dataset verification failed because raw file was not found: {fnf}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"ERROR: Dataset verification failed: {exc}", file=sys.stderr)
        return 1

    print("OK: Raw dataset verification passed completely:")
    print(f"  - Raw rows: {results['raw_rows']:,}")
    print(f"  - Clean rows: {results['clean_rows']:,}")
    print(f"  - Train (2022-2023): {results['splits']['train']:,} rows")
    print(f"  - Validation (2024): {results['splits']['validation']:,} rows")
    print(f"  - Final Test (2025): {results['splits']['final_test']:,} rows")
    print(
        "SUCCESS: Experiment protocol configuration is fully valid and verified against raw data."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
