"""Generate reproducible F1-04 tables and charts from the protocol train subset."""

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pyarrow.parquet import ParquetFile

from f1_lap_time.eda import (
    EVENT_KEYS,
    IQR_MULTIPLIER,
    MIN_GROUP_ROWS,
    REFERENCE_COMPOUND,
    compare_features,
    prepare_train,
    summarize_groups,
    summarize_target,
)
from f1_lap_time.protocol import (
    apply_protocol_filters,
    extract_split_subset,
    load_experiment_config,
    sha256_file,
)
from scripts.download_data import verify_file

ROOT = Path(__file__).resolve().parents[1]
COMMAND = "uv run --locked python -m scripts.run_eda"
CODE_FILES = (
    "scripts/run_eda.py",
    "scripts/download_data.py",
    "src/f1_lap_time/eda.py",
    "src/f1_lap_time/protocol.py",
)
COLORS = ("#156072", "#b84527")


def load_train(repo_root: Path, config: dict, manifest: dict) -> tuple[pd.DataFrame, dict]:
    """Verify the pinned inputs and read only the years and columns used by F1-04."""
    source = config["data_source"]
    raw_path = repo_root / source["primary_file"]
    selected = manifest["selected_file"]
    source_fields = {
        "filename": raw_path.name,
        "path": source["primary_file"],
        "sha256": source["sha256"],
    }
    if any(selected.get(key) != value for key, value in source_fields.items()):
        raise ValueError("Manifest selected file differs from the experiment config")
    if (
        manifest["source"]["doi"] != source["source_doi"]
        or manifest["source"]["version"] != source["version"]
    ):
        raise ValueError("Manifest DOI or version differs from the experiment config")
    verify_file(raw_path, selected)
    with ParquetFile(raw_path) as parquet:
        shape = parquet.metadata.num_rows, len(parquet.schema_arrow.names)
    if shape != (source["raw_rows"], source["raw_columns"]):
        raise ValueError(f"Raw Parquet shape differs from config: {shape}")

    target = config["problem_definition"]["target_column"]
    scope = config["scope_and_filters"]
    years = config["split_protocol"]["train"]["years"]
    candidates = ["LapNumber", "Compound", "TyreLife"]
    if not set(candidates).issubset(config["features"]["allowed_candidate_features"]):
        raise ValueError("F1-04 requires LapNumber, Compound and TyreLife as allowed candidates")

    columns = list(
        dict.fromkeys(
            [
                *EVENT_KEYS,
                *source["candidate_key"],
                *scope["safety_car_filter"]["columns_checked"],
                *candidates,
                target,
            ]
        )
    )
    raw_train = pd.read_parquet(
        raw_path, engine="pyarrow", columns=columns, filters=[("Year", "in", years)]
    )
    train = extract_split_subset(apply_protocol_filters(raw_train, config), config, "train")
    key = source["candidate_key"]
    if train[key].isna().any().any() or train.duplicated(subset=key).any():
        raise ValueError("Train contains missing or duplicate lap keys after scope filters")

    expected_rows = config["split_protocol"]["train"]["expected_rows"]
    if len(train) != expected_rows:
        raise ValueError(
            f"Filtered train size differs from config: {len(train)} != {expected_rows}"
        )

    counts = {
        "source_total_rows": shape[0],
        "source_total_columns": shape[1],
        "partition": "train",
        "years": years,
        "columns_read": columns,
        "rows_before_scope_filter": len(raw_train),
        "rows_excluded_by_scope_filter": len(raw_train) - len(train),
        "rows_analyzed": len(train),
        "event_sessions": train.groupby(EVENT_KEYS).ngroups,
        "rows_by_year": [
            {
                "year": year,
                "before_scope_filter": int(raw_train["Year"].eq(year).sum()),
                "after_scope_filter": int(train["Year"].eq(year).sum()),
            }
            for year in years
        ],
    }
    return train, counts


def plot_target(train: pd.DataFrame, target: str, output: Path) -> None:
    figure = Figure(figsize=(8.4, 4.3), layout="constrained")
    axis = figure.subplots()
    bins = np.histogram_bin_edges(train[target], bins=62)
    for index, (year, rows) in enumerate(train.groupby("Year")):
        weights = np.full(len(rows), 100 / len(rows))
        axis.hist(
            rows[target],
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=1.8,
            color=COLORS[index % len(COLORS)],
            label=f"{year} (n={len(rows):,})",
        )
    axis.set(
        title="Phân phối thời gian vòng trên tập train",
        xlabel=f"{target} (giây)",
        ylabel="Tỷ lệ vòng trong mỗi mùa (%)",
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=150)


def plot_events(groups: pd.DataFrame, output: Path) -> None:
    events = groups.loc[groups["group"].eq("event")]
    ordered_events = events.groupby("EventName")["median_s"].mean().sort_values()
    positions = {event: position for position, event in enumerate(ordered_events.index)}
    figure = Figure(figsize=(8.4, 8.5), layout="constrained")
    axis = figure.subplots()
    seasons = events.groupby("Year")
    offsets = np.linspace(-0.15, 0.15, seasons.ngroups) if seasons.ngroups > 1 else [0]
    for index, (year, rows) in enumerate(seasons):
        axis.errorbar(
            rows["median_s"],
            rows["EventName"].map(positions) + offsets[index],
            xerr=[rows["median_s"] - rows["q1_s"], rows["q3_s"] - rows["median_s"]],
            fmt="o",
            markersize=4,
            linewidth=1.3,
            color=COLORS[index % len(COLORS)],
            label=str(year),
        )
    axis.set_yticks(
        list(positions.values()), [event.removesuffix(" Grand Prix") for event in positions]
    )
    axis.invert_yaxis()
    axis.set(
        title="Thời gian vòng theo chặng trên tập train",
        xlabel="Trung vị và khoảng Q1–Q3 (giây)",
    )
    axis.grid(axis="x", alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=150)


def plot_features(comparisons: pd.DataFrame, output: Path) -> None:
    figure = Figure(figsize=(10.2, 4.8), layout="constrained")
    compound_ax, correlation_ax = figure.subplots(1, 2)
    compound = comparisons.loc[comparisons["feature"].eq("Compound")]
    tyre_life = comparisons.loc[comparisons["feature"].eq("TyreLife_start")]
    lap_number = comparisons.loc[comparisons["feature"].eq("LapNumber"), "value"]
    panels = [
        (
            compound_ax,
            list(compound.groupby("compound")["value"]),
            "Loại lốp so với MEDIUM",
            "Chênh lệch trung vị trong chặng (giây)",
        ),
        (
            correlation_ax,
            [("Số thứ tự vòng", lap_number), *tyre_life.groupby("compound")["value"]],
            "Vòng đua và tuổi lốp",
            "Tương quan Pearson với thời gian vòng",
        ),
    ]
    for axis, groups, title, xlabel in panels:
        nonempty = [(label, values) for label, values in groups if len(values)]
        if nonempty:
            axis.boxplot(
                [values for _, values in nonempty],
                tick_labels=[f"{label} ({len(values)})" for label, values in nonempty],
                orientation="horizontal",
                medianprops={"color": COLORS[0], "linewidth": 2},
            )
            axis.invert_yaxis()
        else:
            axis.text(0.5, 0.5, "Không đủ nhóm để so sánh", ha="center", transform=axis.transAxes)
        axis.axvline(0, color="#87949a", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel=xlabel)
        axis.grid(axis="x", alpha=0.22)
    correlation_ax.set_xlim(-1, 1)
    figure.suptitle("Quan hệ trong từng chặng • số trong ngoặc là số nhóm")
    figure.savefig(output, dpi=150)


def git_provenance(repo_root: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True)
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True)
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit_hash": None, "git_dirty_flag": None}
    return {"git_commit_hash": commit.strip(), "git_dirty_flag": bool(status.strip())}


def run_eda(repo_root: Path) -> dict:
    config_path = repo_root / "configs" / "experiment.json"
    manifest_path = repo_root / "data" / "manifest.json"
    config = load_experiment_config(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filtered_train, scope = load_train(repo_root, config, manifest)
    target = config["problem_definition"]["target_column"]
    placeholders = config["preprocessing_and_pipeline"]["pseudo_missing_strings"]
    train = prepare_train(filtered_train, target, placeholders)
    groups = summarize_groups(train, target)
    comparisons = compare_features(train, target)
    metadata = {
        "source": {
            "path": config["data_source"]["primary_file"],
            "doi": config["data_source"]["source_doi"],
            "version": config["data_source"]["version"],
            "sha256": config["data_source"]["sha256"],
        },
        "provenance": {
            "command": COMMAND,
            "manifest_path": "data/manifest.json",
            "manifest_sha256": sha256_file(manifest_path),
            "config_path": "configs/experiment.json",
            "config_sha256": sha256_file(config_path),
            "code_sha256": {file: sha256_file(repo_root / file) for file in CODE_FILES},
            **git_provenance(repo_root),
            "python": sys.version.split()[0],
            "package_versions": {
                package: version(package)
                for package in ("pandas", "numpy", "pyarrow", "matplotlib")
            },
        },
        "scope": scope,
        "methods": {
            "target": target,
            "unit": config["problem_definition"]["target_unit"],
            "event_keys": EVENT_KEYS,
            "tail_diagnostic": {"iqr_multiplier": IQR_MULTIPLIER, "excluded_rows": 0},
            "min_rows_per_feature_group": MIN_GROUP_ROWS,
            "compound_reference": REFERENCE_COMPOUND,
            "compound_placeholder_strings": placeholders,
            "tyre_life_start": "TyreLife - 1.0",
        },
    }

    report = repo_root / "reports" / "eda"
    report.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".f1-04-", dir=report.parent) as staging:
        temporary = Path(staging)
        figures = temporary / "figures"
        figures.mkdir()
        for table, name in [
            (summarize_target(train, target), "target-summary.csv"),
            (groups, "group-summary.csv"),
            (comparisons, "feature-summary.csv"),
        ]:
            table.to_csv(temporary / name, index=False, float_format="%.8f", lineterminator="\n")
        plot_target(train, target, figures / "target-distribution.png")
        plot_events(groups, figures / "event-lap-times.png")
        plot_features(comparisons, figures / "feature-relationships.png")
        outputs = sorted(path for path in temporary.rglob("*") if path.is_file())
        metadata["output_sha256"] = {
            path.relative_to(temporary).as_posix(): sha256_file(path) for path in outputs
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for path in [*outputs, temporary / "metadata.json"]:
            destination = report / path.relative_to(temporary)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
    return metadata


def main() -> int:
    try:
        metadata = run_eda(ROOT)
    except (OSError, ValueError, KeyError) as exc:
        print(f"EDA error: {exc}", file=sys.stderr)
        return 1
    print(f"Train rows analyzed: {metadata['scope']['rows_analyzed']:,}")
    print("EDA outputs: reports/eda/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
