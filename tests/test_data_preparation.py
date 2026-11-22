"""Unit tests for F1-05 data preparation logic, leakage boundaries, and split assignments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from f1_lap_time.data_preparation import (
    BASE_COLUMNS,
    KEY_COLUMNS,
    assign_splits,
    check_candidate_key_duplicates,
    generate_row_id,
    load_prepared_data,
    normalize_pseudo_missing,
    prepare_base_and_splits,
    validate_target_column,
    validate_tyrelife_column,
)
from f1_lap_time.protocol import sha256_file


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Minimal valid config dictionary for testing data preparation."""
    return {
        "problem_definition": {
            "task_type": "regression",
            "target_column": "LapTimeSec",
        },
        "data_source": {
            "primary_file": "data/raw/f1_research_dataset_with_risk_safetycar.parquet",
            "source_doi": "10.17632/djr8rnjtjp.1",
            "version": 1,
            "raw_rows": 10,
            "raw_columns": 35,
            "sha256": "dummy",
            "candidate_key": ["Year", "EventName", "Session", "Driver", "LapNumber"],
        },
        "scope_and_filters": {
            "session_scope": "R",
            "preseason_filter": {
                "action": "exclude",
                "condition": "starts_with_any",
                "match_prefixes": ["Pre-Season"],
                "rows_excluded": 1,
            },
            "safety_car_filter": {
                "action": "exclude",
                "condition": "any_equals_1",
                "columns_checked": ["IsSafetyCar", "IsVSC"],
                "rows_excluded_outside_preseason": 1,
            },
            "expected_clean_rows": 4,
            "duplicate_handling": {
                "key": ["Year", "EventName", "Session", "Driver", "LapNumber"],
                "action": "raise_on_conflict_else_drop",
            },
            "target_validity": {
                "target_column": "LapTimeSec",
                "condition": "numeric_positive_finite",
            },
        },
        "split_protocol": {
            "strategy": "temporal_by_year",
            "valid_years": [2022, 2023, 2024, 2025],
            "train": {"years": [2022, 2023], "expected_rows": 2},
            "validation": {"years": [2024], "expected_rows": 1},
            "final_test": {"years": [2025], "expected_rows": 1},
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
            "tracking_and_split_columns": ["Year", "Session", "EventKey", "IsAccurate"],
            "forbidden_features": ["Position", "TrackStatus", "Time", "Sector1Time"],
            "pending_verification_features": ["AirTemp", "TrackTemp"],
        },
        "preprocessing_and_pipeline": {
            "pseudo_missing_strings": ["None", "nan"],
        },
    }


def test_normalize_pseudo_missing() -> None:
    df = pd.DataFrame({"Compound": ["SOFT", "None", "MEDIUM", "nan", "HARD"]})
    clean = normalize_pseudo_missing(df, "Compound", ("None", "nan"))
    assert clean["Compound"].isna().sum() == 2
    assert list(clean["Compound"].dropna()) == ["SOFT", "MEDIUM", "HARD"]


def test_normalize_pseudo_missing_with_whitespace() -> None:
    df = pd.DataFrame({"Compound": ["SOFT", " None ", "nan ", "", "HARD"]})
    clean = normalize_pseudo_missing(df, "Compound", ("None", "nan"))
    assert clean["Compound"].isna().sum() == 3
    assert list(clean["Compound"].dropna()) == ["SOFT", "HARD"]


def test_validate_target_column_success() -> None:
    df = pd.DataFrame({"LapTimeSec": [80.5, 91.2, 75.0]})
    validate_target_column(df, "LapTimeSec")  # Should not raise


@pytest.mark.parametrize(
    "invalid_values, error_substr",
    [
        ([80.0, np.nan, 90.0], "NaN"),
        ([80.0, np.inf, 90.0], "infinite"),
        ([80.0, -np.inf, 90.0], "infinite"),
        ([80.0, 0.0, 90.0], "non-positive"),
        ([80.0, -5.0, 90.0], "non-positive"),
    ],
)
def test_validate_target_column_invalid(invalid_values: list[float], error_substr: str) -> None:
    df = pd.DataFrame({"LapTimeSec": invalid_values})
    with pytest.raises(ValueError, match=error_substr):
        validate_target_column(df, "LapTimeSec")


def test_validate_target_column_missing_column() -> None:
    df = pd.DataFrame({"OtherColumn": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing target column"):
        validate_target_column(df, "LapTimeSec")


def test_validate_tyrelife_column_valid() -> None:
    df = pd.DataFrame({"TyreLife": [1.0, 2.5, 30.0, np.nan]})
    validate_tyrelife_column(df, "TyreLife")  # Should not raise


@pytest.mark.parametrize(
    "invalid_values, error_substr",
    [
        ([1.0, 0.0, 5.0], "< 1.0"),
        ([1.0, -2.0, 5.0], "< 1.0"),
        ([1.0, 0.999, 5.0], "< 1.0"),
        ([1.0, np.inf, 5.0], "infinite"),
        ([1.0, -np.inf, 5.0], "infinite"),
    ],
)
def test_validate_tyrelife_column_invalid(invalid_values: list[float], error_substr: str) -> None:
    df = pd.DataFrame({"TyreLife": invalid_values})
    with pytest.raises(ValueError, match=error_substr):
        validate_tyrelife_column(df, "TyreLife")


def test_check_candidate_key_duplicates_exact() -> None:
    df = pd.DataFrame(
        {
            "Year": [2022, 2022, 2023],
            "EventName": ["Austrian GP", "Austrian GP", "Austrian GP"],
            "Session": ["R", "R", "R"],
            "Driver": ["VER", "VER", "VER"],
            "LapNumber": [2.0, 2.0, 3.0],
            "LapTimeSec": [85.0, 85.0, 86.0],
            "Team": ["Red Bull", "Red Bull", "Red Bull"],
        }
    )
    dedup, dropped = check_candidate_key_duplicates(df, KEY_COLUMNS)
    assert len(dedup) == 2
    assert dropped == 1


def test_check_candidate_key_duplicates_conflict_raises() -> None:
    df = pd.DataFrame(
        {
            "Year": [2022, 2022],
            "EventName": ["Austrian GP", "Austrian GP"],
            "Session": ["R", "R"],
            "Driver": ["VER", "VER"],
            "LapNumber": [2.0, 2.0],
            "LapTimeSec": [85.0, 95.0],  # Conflicting target!
            "Team": ["Red Bull", "Red Bull"],
        }
    )
    with pytest.raises(ValueError, match="Candidate key conflicts found"):
        check_candidate_key_duplicates(df, KEY_COLUMNS)


def test_check_candidate_key_duplicates_null_raises() -> None:
    df = pd.DataFrame(
        {
            "Year": [2022, np.nan],
            "EventName": ["GP", "GP"],
            "Session": ["R", "R"],
            "Driver": ["VER", "VER"],
            "LapNumber": [2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="Candidate key contains null values"):
        check_candidate_key_duplicates(df, KEY_COLUMNS)


def test_check_candidate_key_duplicates_missing_col_raises() -> None:
    df = pd.DataFrame({"Year": [2022], "Session": ["R"]})
    with pytest.raises(ValueError, match="Missing candidate key columns"):
        check_candidate_key_duplicates(df, KEY_COLUMNS)


def test_generate_row_id() -> None:
    df = pd.DataFrame(
        {
            "Year": [2022, 2023],
            "EventName": ["Bahrain Grand Prix", "São Paulo Grand Prix"],
            "Session": ["R", "R"],
            "Driver": ["LEC", "HAM"],
            "LapNumber": [2.0, 15.0],
        }
    )
    row_ids = generate_row_id(df, KEY_COLUMNS)
    assert list(row_ids) == [
        "2022_Bahrain_Grand_Prix_R_LEC_2",
        "2023_São_Paulo_Grand_Prix_R_HAM_15",
    ]
    assert not row_ids.isna().any()
    assert row_ids.nunique() == 2


def test_generate_row_id_invariance_with_mixed_laps() -> None:
    # Lap 2.0 must format to '2' regardless of whether a non-integer lap (3.5) exists in the batch
    df = pd.DataFrame(
        {
            "Year": [2022, 2022],
            "EventName": ["GP", "GP"],
            "Session": ["R", "R"],
            "Driver": ["VER", "HAM"],
            "LapNumber": [2.0, 3.5],
        }
    )
    row_ids = generate_row_id(df, KEY_COLUMNS)
    assert row_ids.iloc[0] == "2022_GP_R_VER_2"
    assert row_ids.iloc[1] == "2022_GP_R_HAM_3.5"


def test_generate_row_id_null_raises() -> None:
    df = pd.DataFrame(
        {
            "Year": [2022],
            "EventName": [np.nan],
            "Session": ["R"],
            "Driver": ["VER"],
            "LapNumber": [2.0],
        }
    )
    with pytest.raises(ValueError, match="null key components"):
        generate_row_id(df, KEY_COLUMNS)


def test_generate_row_id_duplicate_raises() -> None:
    df = pd.DataFrame(
        {
            "Year": [2022, 2022],
            "EventName": ["GP", "GP"],
            "Session": ["R", "R"],
            "Driver": ["VER", "VER"],
            "LapNumber": [2.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="not unique"):
        generate_row_id(df, KEY_COLUMNS)


def test_assign_splits(sample_config: dict[str, Any]) -> None:
    df = pd.DataFrame({"Year": [2022, 2023, 2024, 2025]})
    splits = assign_splits(df, sample_config)
    assert list(splits) == ["train", "train", "validation", "test"]


def test_assign_splits_overlap_raises(sample_config: dict[str, Any]) -> None:
    sample_config["split_protocol"]["train"]["years"] = [2022, 2024]  # Overlap with val!
    df = pd.DataFrame({"Year": [2022, 2024]})
    with pytest.raises(ValueError, match="Year overlap"):
        assign_splits(df, sample_config)


def test_assign_splits_unassigned_raises(sample_config: dict[str, Any]) -> None:
    df = pd.DataFrame({"Year": [2022, 2026]})  # 2026 is unexpected
    with pytest.raises(ValueError, match="could not be assigned"):
        assign_splits(df, sample_config)


def test_prepare_base_and_splits_pipeline(sample_config: dict[str, Any]) -> None:
    raw_df = pd.DataFrame(
        {
            "Year": [2022, 2022, 2023, 2024, 2025, 2022, 2023],
            "EventName": [
                "Pre-Season Test",  # Excluded by pre-season filter
                "Austrian GP",
                "Austrian GP",
                "Austrian GP",
                "Austrian GP",
                "Austrian GP",  # Excluded by SC filter
                "Austrian GP",  # Excluded by session filter
            ],
            "Session": ["R", "R", "R", "R", "R", "R", "Q"],
            "Driver": ["VER", "VER", "HAM", "NOR", "LEC", "SAI", "ALO"],
            "LapNumber": [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "IsSafetyCar": [0, 0, 0, 0, 0, 1, 0],
            "IsVSC": [0, 0, 0, 0, 0, 0, 0],
            "Team": ["RB", "RB", "MER", "MCL", "FER", "FER", "AM"],
            "Compound": ["MEDIUM", "None", "SOFT", "HARD", "nan", "SOFT", "SOFT"],
            "FreshTyre": [True, False, True, True, False, True, True],
            "Stint": [1.0, 1.0, 2.0, 2.0, 3.0, 1.0, 1.0],
            "TyreLife": [1.0, np.nan, 5.0, 10.0, 2.0, 3.0, 4.0],
            "EventKey": ["E1", "E2", "E3", "E4", "E5", "E6", "E7"],
            "IsAccurate": [True] * 7,
            "LapTimeSec": [80.0, 85.0, 86.5, 87.0, 88.0, 120.0, 75.0],
            # Forbidden / pending columns that must NOT leak into base.csv
            "Position": [1, 2, 3, 4, 5, 6, 7],
            "TrackStatus": ["1", "1", "1", "1", "1", "4", "1"],
            "Time": [100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0],
            "Sector1Time": [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 24.0],
            "AirTemp": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
            "TrackTemp": [30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        }
    )

    base_df, splits_df, stats = prepare_base_and_splits(raw_df, sample_config)

    # Clean rows: 4 rows remain (VER 2022, HAM 2023, NOR 2024, LEC 2025)
    assert len(base_df) == 4
    assert len(splits_df) == 4
    assert stats["clean_rows"] == 4

    # Check row_id alignment
    assert (base_df["row_id"].values == splits_df["row_id"].values).all()

    # Check split assignment: 2 train, 1 validation, 1 test
    assert stats["split_counts"]["train"] == 2
    assert stats["split_counts"]["validation"] == 1
    assert stats["split_counts"]["test"] == 1

    # Check pseudo-missing normalization: "None" in VER 2022 and "nan" in LEC 2025 become NaN
    assert base_df["Compound"].isna().sum() == 2

    # Check leakage: NO forbidden or weather columns in base_df
    forbidden = ["Position", "TrackStatus", "Time", "Sector1Time"]
    weather = ["AirTemp", "TrackTemp"]
    for col in forbidden + weather:
        assert col not in base_df.columns, f"Column '{col}' leaked into base_df!"

    # Check expected schema
    assert list(base_df.columns) == BASE_COLUMNS


def test_load_prepared_data_mock(tmp_path: Path) -> None:
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)

    base_df = pd.DataFrame(
        {
            "row_id": ["id_1", "id_2", "id_3"],
            "Year": [2022, 2024, 2025],
            "EventName": ["GP1", "GP2", "GP3"],
            "Session": ["R", "R", "R"],
            "Driver": ["VER", "HAM", "NOR"],
            "LapNumber": [2.0, 3.0, 4.0],
            "Team": ["RB", "MER", "MCL"],
            "Compound": ["SOFT", "MEDIUM", "HARD"],
            "FreshTyre": [True, False, True],
            "Stint": [1.0, 1.0, 2.0],
            "TyreLife": [2.0, 5.0, 8.0],
            "EventKey": ["E1", "E2", "E3"],
            "IsAccurate": [True, True, True],
            "LapTimeSec": [80.0, 81.0, 82.0],
        }
    )
    splits_df = pd.DataFrame(
        {"row_id": ["id_1", "id_2", "id_3"], "split": ["train", "validation", "test"]}
    )

    base_file = proc_dir / "base.csv"
    splits_file = proc_dir / "splits.csv"
    meta_file = proc_dir / "metadata.json"

    base_df.to_csv(base_file, index=False)
    splits_df.to_csv(splits_file, index=False)

    metadata = {
        "processed_outputs": {
            "base_csv": {"sha256": sha256_file(base_file)},
            "splits_csv": {"sha256": sha256_file(splits_file)},
        }
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # 1. Load full
    loaded, meta = load_prepared_data(tmp_path, verify_hashes=True)
    assert len(loaded) == 3
    assert "split" in loaded.columns

    # 2. Load train only
    train_only, _ = load_prepared_data(tmp_path, split="train")
    assert len(train_only) == 1
    assert train_only["split"].iloc[0] == "train"

    # 3. Load test via final_test alias
    test_only, _ = load_prepared_data(tmp_path, split="final_test")
    assert len(test_only) == 1
    assert test_only["split"].iloc[0] == "test"

    # 4. Check index reset on split filter (0-based contiguous)
    val_only, _ = load_prepared_data(tmp_path, split="validation")
    assert val_only.index[0] == 0
    assert len(val_only) == 1

    # 5. Unknown split raises ValueError
    with pytest.raises(ValueError, match="Unknown split"):
        load_prepared_data(tmp_path, split="invalid_split_name")

    # 6. Hash mismatch check
    with base_file.open("a", encoding="utf-8") as f:
        f.write("\ncorrupted_line")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_prepared_data(tmp_path, verify_hashes=True)


def test_load_prepared_data_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Prepared file not found"):
        load_prepared_data(tmp_path)


def test_load_prepared_data_sklearn_pipeline_compatibility(tmp_path: Path) -> None:
    """Ensure loaded prepared data with missing values works seamlessly with scikit-learn transformers."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)

    base_df = pd.DataFrame(
        {
            "row_id": ["id_1", "id_2", "id_3"],
            "Year": [2022, 2022, 2023],
            "EventName": ["GP1", "GP2", "GP3"],
            "Session": ["R", "R", "R"],
            "Driver": ["VER", "HAM", "NOR"],
            "LapNumber": [2.0, 3.0, 4.0],
            "Team": ["RB", "MER", "MCL"],
            "Compound": ["SOFT", np.nan, "HARD"],
            "FreshTyre": [True, False, True],
            "Stint": [1.0, np.nan, 2.0],
            "TyreLife": [2.0, 5.0, np.nan],
            "EventKey": ["E1", "E2", "E3"],
            "IsAccurate": [True, True, True],
            "LapTimeSec": [80.0, 81.0, 82.0],
        }
    )
    splits_df = pd.DataFrame(
        {"row_id": ["id_1", "id_2", "id_3"], "split": ["train", "train", "train"]}
    )

    base_file = proc_dir / "base.csv"
    splits_file = proc_dir / "splits.csv"
    meta_file = proc_dir / "metadata.json"

    base_df.to_csv(base_file, index=False)
    splits_df.to_csv(splits_file, index=False)

    metadata = {
        "processed_outputs": {
            "base_csv": {"sha256": sha256_file(base_file)},
            "splits_csv": {"sha256": sha256_file(splits_file)},
        }
    }
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f)

    train_data, _ = load_prepared_data(tmp_path, split="train")

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("cat", cat_pipeline, ["Compound"]),
            ("num", num_pipeline, ["Stint", "TyreLife"]),
        ]
    )

    transformed = preprocessor.fit_transform(train_data)
    assert transformed.shape == (3, 4)
    assert not np.isnan(transformed).any()


def test_load_prepared_data_missing_hash_in_metadata_raises(tmp_path: Path) -> None:
    proc_dir = tmp_path / "data" / "processed"
    proc_dir.mkdir(parents=True)
    base_file = proc_dir / "base.csv"
    splits_file = proc_dir / "splits.csv"
    meta_file = proc_dir / "metadata.json"

    pd.DataFrame({"row_id": ["id_1"], "LapTimeSec": [80.0]}).to_csv(base_file, index=False)
    pd.DataFrame({"row_id": ["id_1"], "split": ["train"]}).to_csv(splits_file, index=False)
    # Empty or missing hashes in metadata
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump({"processed_outputs": {}}, f)

    with pytest.raises(ValueError, match="Missing required 'base_csv' SHA-256 hash"):
        load_prepared_data(tmp_path, verify_hashes=True)


@pytest.mark.parametrize(
    "column, invalid_value, error_substr",
    [
        ("TyreLife", 0.0, "contains values < 1.0"),
        ("TyreLife", "2.0", "must be numeric"),
        ("FreshTyre", "False", "FreshTyre.*boolean"),
        ("IsAccurate", "False", "IsAccurate.*boolean"),
        ("LapTimeSec", "80.0", "must be numeric"),
    ],
)
def test_prepare_base_and_splits_rejects_invalid_source_values(
    sample_config: dict[str, Any], column: str, invalid_value: Any, error_substr: str
) -> None:
    raw_df = pd.DataFrame(
        {
            "Year": [2022],
            "EventName": ["Austrian GP"],
            "Session": ["R"],
            "Driver": ["VER"],
            "LapNumber": [2.0],
            "IsSafetyCar": [0],
            "IsVSC": [0],
            "Team": ["RB"],
            "Compound": ["MEDIUM"],
            "FreshTyre": [True],
            "Stint": [1.0],
            "TyreLife": [1.0],
            "EventKey": ["E1"],
            "IsAccurate": [True],
            "LapTimeSec": [80.0],
        }
    )
    raw_df[column] = invalid_value
    with pytest.raises(ValueError, match=error_substr):
        prepare_base_and_splits(raw_df, sample_config)
