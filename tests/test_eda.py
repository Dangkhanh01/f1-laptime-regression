"""Regression tests for train-only reporting and descriptive statistics."""

import numpy as np
import pandas as pd
import pytest

from f1_lap_time.eda import compare_features, prepare_train, summarize_groups, summarize_target
from f1_lap_time.protocol import load_experiment_config, sha256_file
from scripts.run_eda import load_train

TARGET = "LapTimeSec"


@pytest.fixture
def eda_source(tmp_path):
    config = load_experiment_config()
    source = config["data_source"]
    raw_path = tmp_path / source["primary_file"]
    raw_path.parent.mkdir(parents=True)
    laps = pd.DataFrame(
        {
            "Year": [2022, 2023, 2022, 2023, 2023, 2022, 2024, 2025],
            "EventName": ["Race", "Race", "Pre-Season Test", *["Race"] * 5],
            "Session": ["R", "R", "R", "R", "R", "Q", "R", "R"],
            "Driver": ["ABC"] * 8,
            "LapNumber": range(2, 10),
            "IsSafetyCar": [0, 0, 0, 1, 0, 0, 0, 0],
            "IsVSC": [0, 0, 0, 0, 1, 0, 0, 0],
            "Compound": ["MEDIUM"] * 8,
            "TyreLife": [2.0] * 8,
            TARGET: [81.0, 92.0, 83.0, 105.0, 106.0, 80.0, 555.0, 999.0],
        }
    )
    laps.to_parquet(raw_path, index=False)
    source.update(raw_rows=len(laps), raw_columns=len(laps.columns), sha256=sha256_file(raw_path))
    config["split_protocol"]["train"]["expected_rows"] = 2
    manifest = {
        "selected_file": {
            "filename": raw_path.name,
            "path": source["primary_file"],
            "sha256": source["sha256"],
            "size_bytes": raw_path.stat().st_size,
        },
        "source": {"doi": source["source_doi"], "version": source["version"]},
    }
    return raw_path, config, manifest


def test_eda_uses_only_filtered_train_rows(tmp_path, eda_source):
    raw_path, config, manifest = eda_source
    train, scope = load_train(tmp_path, config, manifest)
    statistics = summarize_target(prepare_train(train, TARGET, ["None", "nan"]), TARGET)

    assert scope["source_total_rows"] == 8
    assert scope["rows_before_scope_filter"] == 6
    assert scope["rows_excluded_by_scope_filter"] == 4
    assert scope["rows_analyzed"] == 2
    assert train["Year"].tolist() == [2022, 2023]
    assert train[TARGET].tolist() == [81.0, 92.0]
    assert statistics.iloc[0]["median_s"] == 86.5
    assert sha256_file(raw_path) == config["data_source"]["sha256"]


def test_eda_checks_raw_hash(tmp_path, eda_source):
    raw_path, config, manifest = eda_source
    original = raw_path.read_bytes()
    raw_path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))

    with pytest.raises(ValueError, match="Wrong SHA-256"):
        load_train(tmp_path, config, manifest)


def test_event_fences_and_missing_values_are_reported_without_dropping_rows():
    train = pd.DataFrame(
        {
            "Year": [2022] * 10,
            "EventName": ["A"] * 5 + ["B"] * 5,
            "Session": ["R"] * 10,
            "Compound": ["None", "nan", "MEDIUM", "MEDIUM", "MEDIUM"] * 2,
            "TyreLife": [1.0, 2.0, 3.0, 4.0, np.nan] * 2,
            TARGET: [80.0, 81.0, 82.0, 83.0, 94.0, 110.0, 111.0, 112.0, 113.0, 114.0],
        }
    )
    analysis = prepare_train(train, TARGET, ["None", "nan"])
    groups = summarize_groups(analysis, TARGET)
    events = groups.loc[groups["group"].eq("event")]
    missing_compound = groups.loc[groups["group"].eq("event_compound") & groups["Compound"].isna()]

    assert len(analysis) == len(train)
    assert train["Compound"].tolist()[:2] == ["None", "nan"]
    assert analysis["TyreLife_start"].iloc[0] == 0
    assert events["above_event_fence_n"].tolist() == [1, 0]
    assert events["missing_tyre_life_n"].tolist() == [1, 1]
    assert missing_compound["n"].tolist() == [2, 2]


def test_feature_comparisons_use_same_event_reference_and_group_minimum():
    compound = ["MEDIUM"] * 30 + ["HARD"] * 30 + ["SOFT"] * 29 + ["HARD"] * 30
    times = [*np.arange(80, 110), *np.arange(81, 111), *np.arange(83, 112), *np.arange(95, 125)]
    train = pd.DataFrame(
        {
            "Year": [2022] * len(compound),
            "EventName": ["A"] * 89 + ["B"] * 30,
            "Session": ["R"] * len(compound),
            "LapNumber": np.arange(len(compound)) + 2,
            "Compound": compound,
            "TyreLife": np.arange(len(compound)) + 1.0,
            TARGET: times,
        }
    )
    train.loc[30, "TyreLife"] = np.nan
    results = compare_features(prepare_train(train, TARGET, []), TARGET)

    contrasts = results.loc[results["feature"].eq("Compound")]
    assert contrasts["EventName"].tolist() == ["A"]
    assert contrasts["compound"].tolist() == ["HARD"]
    assert contrasts["reference_compound"].tolist() == ["MEDIUM"]
    assert contrasts[["n", "reference_n", "value"]].iloc[0].tolist() == [30, 30, 1.0]

    tyre_age = results.loc[results["feature"].eq("TyreLife_start")]
    assert list(zip(tyre_age["EventName"], tyre_age["compound"], strict=True)) == [
        ("A", "MEDIUM"),
        ("B", "HARD"),
    ]
    assert tyre_age["value"].tolist() == pytest.approx([1.0, 1.0])
