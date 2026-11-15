"""Create reproducible schema and quality reports for the selected lap dataset."""

import json

import pandas as pd

from scripts.download_data import MANIFEST, ROOT, verify_extracted, verify_file

REPORT = ROOT / "reports" / "data-inventory"
KEY = ["Year", "EventName", "Session", "Driver", "LapNumber"]
REQUIRED = set(KEY + ["LapTime", "LapTimeSec", "Sector1Time", "Sector2Time", "Sector3Time"])
FASTF1_LAPS = "https://github.com/theOehrly/Fast-F1/blob/main/docs/api_reference/timing_data.rst"
FASTF1_WEATHER = "https://github.com/theOehrly/Fast-F1/blob/main/fastf1/_api.py"
MENDELEY = "https://data.mendeley.com/datasets/djr8rnjtjp/1"
STRING_PLACEHOLDERS = {"None", "nan"}
MEANINGS = {
    "Driver": ("Mã tay đua", "", "tên cột"),
    "Team": ("Tên đội đua", "", FASTF1_LAPS),
    "LapNumber": ("Số vòng của tay đua trong sự kiện", "vòng", "tên cột"),
    "LapTime": ("Thời gian vòng; đối chiếu với LapTimeSec", "giây", "đối chiếu dữ liệu"),
    "LapTimeSec": ("Thời gian vòng", "giây", "tên cột và đối chiếu dữ liệu"),
    "Sector1Time": ("Thời gian sector 1", "giây", "tổng sector đối chiếu LapTimeSec"),
    "Sector2Time": ("Thời gian sector 2", "giây", "tổng sector đối chiếu LapTimeSec"),
    "Sector3Time": ("Thời gian sector 3", "giây", "tổng sector đối chiếu LapTimeSec"),
    "Compound": ("Loại lốp trong sự kiện", "", FASTF1_LAPS),
    "TyreLife": ("Số vòng đã chạy trên bộ lốp, kể cả phiên khác", "vòng", FASTF1_LAPS),
    "FreshTyre": ("Lốp mới khi bắt đầu stint", "đúng/sai", FASTF1_LAPS),
    "Stint": ("Số thứ tự stint", "stint", FASTF1_LAPS),
    "IsAccurate": (
        "Cờ kiểm tra đồng bộ đầu/cuối vòng; không bảo đảm mọi dữ liệu chính xác",
        "đúng/sai",
        FASTF1_LAPS,
    ),
    "TrackStatus": ("Chuỗi mã các trạng thái đường đua xảy ra trong vòng", "mã", FASTF1_LAPS),
    "PitInTime": (
        "Thời điểm xe vào pit tính từ đầu phiên; đơn vị sau chuyển đổi chưa xác minh",
        "",
        FASTF1_LAPS,
    ),
    "PitOutTime": (
        "Thời điểm xe ra pit tính từ đầu phiên; đơn vị sau chuyển đổi chưa xác minh",
        "",
        FASTF1_LAPS,
    ),
    "Position": ("Vị trí tay đua khi kết thúc vòng", "vị trí", FASTF1_LAPS),
    "Time": (
        "Thời điểm kết thúc vòng tính từ đầu phiên; đơn vị giây là suy luận",
        "giây (suy luận)",
        FASTF1_LAPS,
    ),
    "Year": ("Mùa giải", "năm", "tên cột"),
    "EventName": ("Tên sự kiện", "", "tên cột"),
    "Session": ("Mã phiên", "", "tên cột"),
    "AirTemp": ("Nhiệt độ không khí", "°C", FASTF1_WEATHER),
    "TrackTemp": ("Nhiệt độ mặt đường đua", "°C", FASTF1_WEATHER),
    "Humidity": ("Độ ẩm tương đối", "%", FASTF1_WEATHER),
    "Pressure": ("Áp suất không khí", "mbar", FASTF1_WEATHER),
    "Rainfall": ("Cờ có mưa tại thời điểm đo thời tiết", "đúng/sai", FASTF1_WEATHER),
    "WindSpeed": ("Tốc độ gió", "m/s", FASTF1_WEATHER),
    "WindDirection": ("Hướng gió", "độ (0–359)", FASTF1_WEATHER),
    "EventKey": ("Khóa sự kiện do tác giả tạo", "", "đối chiếu dữ liệu"),
    "IsSafetyCar": ("Nhãn Safety Car từ race control", "0/1", "mô tả nguồn"),
    "IsVSC": ("Nhãn Virtual Safety Car từ race control", "0/1", "mô tả nguồn"),
    "Risk_SafetyCar": ("Đầu ra mô hình rủi ro Safety Car của tác giả", "", "dataset_card.json"),
    "RaceProgress": ("Tiến độ cuộc đua; công thức chưa xác minh", "tỷ lệ 0–1 (quan sát)", MENDELEY),
    "HasPitIn": (
        "Cờ có pit-in theo tên cột; công thức chưa xác minh",
        "0/1 (quan sát)",
        "suy luận từ tên cột; chưa có định nghĩa nguồn",
    ),
    "HasPitOut": (
        "Cờ có pit-out theo tên cột; công thức chưa xác minh",
        "0/1 (quan sát)",
        "suy luận từ tên cột; chưa có định nghĩa nguồn",
    ),
}


def schema_rows(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for column in frame:
        meaning, unit, evidence = MEANINGS.get(column, ("", "", "chưa xác minh"))
        missing = int(frame[column].isna().sum())
        placeholders = int(frame[column].isin(STRING_PLACEHOLDERS).sum())
        rows.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "missing_count": missing,
                "missing_pct": round(missing * 100 / len(frame), 4) if len(frame) else 0,
                "placeholder_count": placeholders,
                "meaning": meaning,
                "unit": unit,
                "evidence": evidence,
            }
        )
    return rows


def inspect_frame(frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise ValueError("Dataset is empty")
    missing_columns = REQUIRED - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    years = frame.groupby("Year", dropna=False)
    target = frame["LapTimeSec"]
    sector_sum = frame[["Sector1Time", "Sector2Time", "Sector3Time"]].sum(axis=1, min_count=3)
    result = {
        "observation_unit": "one driver lap in one event/session",
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "candidate_key": KEY,
        "candidate_key_missing_rows": int(frame[KEY].isna().any(axis=1).sum()),
        "candidate_key_duplicate_rows": int(frame.duplicated(KEY).sum()),
        "coverage": {
            "year_rows": {str(year): len(group) for year, group in years},
            "events_by_year": {
                str(year): int(group["EventName"].nunique()) for year, group in years
            },
            "event_names_by_year": {
                str(year): sorted(group["EventName"].dropna().unique().tolist())
                for year, group in years
            },
            "event_keys": int(frame["EventKey"].nunique()) if "EventKey" in frame else None,
            "session_rows": {str(k): int(v) for k, v in frame["Session"].value_counts().items()},
            "preseason_rows": int(
                frame["EventName"].str.contains("Pre-Season", case=False, na=False).sum()
            ),
        },
        "invalid_or_suspect": {
            "lap_time_sec_nonpositive": int(target.le(0).sum()),
            "lap_number_nonpositive_or_noninteger": int(
                (frame["LapNumber"].le(0) | frame["LapNumber"].mod(1).ne(0)).sum()
            ),
            "year_outside_2022_2025": int(frame["Year"].isin(range(2022, 2026)).eq(False).sum()),
            "sector_sum_vs_lap_time_mismatch": int(sector_sum.sub(target).abs().gt(0.001).sum()),
            "lap_time_vs_lap_time_sec_mismatch": int(
                frame["LapTime"].sub(target).abs().gt(0.000001).sum()
            ),
        },
        "target_candidate": {
            "column": "LapTimeSec",
            "unit_candidate": "seconds",
            "min": float(target.min()),
            "max": float(target.max()),
            "missing": int(target.isna().sum()),
        },
    }
    for column, invalid in {
        "Humidity": lambda values: values.lt(0) | values.gt(100),
        "RaceProgress": lambda values: values.lt(0) | values.gt(1),
        "Position": lambda values: values.le(0),
        "TyreLife": lambda values: values.lt(0),
    }.items():
        if column in frame:
            result["invalid_or_suspect"][f"{column}_out_of_range"] = int(
                invalid(frame[column]).sum()
            )
    for column in ("IsAccurate", "HasPitIn", "HasPitOut", "IsSafetyCar", "IsVSC"):
        if column in frame:
            result[column] = {str(k): int(v) for k, v in frame[column].value_counts().items()}
    for column in ("PitInTime", "PitOutTime"):
        if column in frame:
            result[f"{column}_missing"] = int(frame[column].isna().sum())
    if "Compound" in frame:
        result["invalid_or_suspect"]["Compound_placeholder_count"] = int(
            frame["Compound"].isin(STRING_PLACEHOLDERS).sum()
        )
    if "TrackStatus" in frame:
        result["TrackStatus"] = {
            str(k): int(v) for k, v in frame["TrackStatus"].value_counts().items()
        }
    return result


def compare_event_data(events: pd.DataFrame, selected: pd.DataFrame) -> dict:
    """Measure source coverage; compare keys without claiming a processing recipe."""
    for label, frame in (("event", events), ("selected", selected)):
        if frame[KEY].isna().any(axis=None) or frame.duplicated(KEY).any():
            raise ValueError(f"Cannot compare {label} data with missing or duplicate keys")
    event_keys = pd.MultiIndex.from_frame(events[KEY])
    selected_keys = pd.MultiIndex.from_frame(selected[KEY])
    accurate = events["IsAccurate"].eq(True).fillna(False)
    accurate_keys = event_keys[accurate.to_numpy(dtype=bool)]
    excluded = events.loc[~event_keys.isin(selected_keys)]
    return {
        "rows": len(events),
        "columns": len(events.columns),
        "candidate_key": KEY,
        "IsAccurate": {
            str(k): int(v) for k, v in events["IsAccurate"].value_counts(dropna=False).items()
        },
        "matched_keys": int(event_keys.isin(selected_keys).sum()),
        "event_only_keys": len(excluded),
        "selected_only_keys": int((~selected_keys.isin(event_keys)).sum()),
        "selected_keys_equal_accurate_event_keys": bool(
            len(accurate_keys) == len(selected_keys) and accurate_keys.isin(selected_keys).all()
        ),
        "event_only_IsAccurate": {
            str(k): int(v) for k, v in excluded["IsAccurate"].value_counts(dropna=False).items()
        },
        "selected_only_columns": sorted(set(selected.columns) - set(events.columns)),
        "pit_times": {
            column: {
                "nonmissing_rows": int(events[column].notna().sum()),
                "nonmissing_accurate_rows": int((events[column].notna() & accurate).sum()),
                "nonmissing_event_only_rows": int(excluded[column].notna().sum()),
            }
            for column in ("PitInTime", "PitOutTime")
        },
    }


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = manifest["selected_file"]
    archive_record = manifest["archive"]
    path = (ROOT / record["path"]).resolve()
    if not path.is_relative_to((ROOT / "data" / "raw").resolve()):
        raise ValueError("Selected file must stay inside data/raw")
    verify_file(path, record)
    verify_extracted(ROOT / "data" / "raw", record, manifest["extracted"])
    frame = pd.read_parquet(path)
    quality = inspect_frame(frame)
    quality["source"] = {
        "doi": manifest["source"]["doi"],
        "version": manifest["source"]["version"],
        "path": record["path"],
        "sha256": record["sha256"],
        "archive_sha256": archive_record["sha256"],
    }
    quality["event_parquet_files"] = manifest["extracted"]["event_parquet_files"]
    event_paths = sorted((ROOT / "data/raw/f1_safetycar/datasets").rglob("event=*.parquet"))
    events = pd.concat([pd.read_parquet(path) for path in event_paths], ignore_index=True)
    quality["event_data_comparison"] = compare_event_data(events, frame)
    REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(schema_rows(frame)).to_csv(
        REPORT / "schema.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    (REPORT / "quality.json").write_bytes(
        (json.dumps(quality, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    )
    print(f"Inspected {len(frame)} rows, {len(frame.columns)} columns from {record['path']}")


if __name__ == "__main__":
    main()
