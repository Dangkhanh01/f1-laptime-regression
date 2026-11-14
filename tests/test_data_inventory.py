"""Small checks for source integrity and inventory counts."""

import hashlib
import zipfile

import pandas as pd
import pytest

from scripts.download_data import extract_package, extract_selected, verify_extracted, verify_file
from scripts.inspect_data import inspect_frame, schema_rows


def test_extract_selected_checks_content_before_writing(tmp_path):
    archive = tmp_path / "source.zip"
    content = b"PAR1testPAR1"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("nested/laps.parquet", content)
    record = {
        "archive_member": "nested/laps.parquet",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    destination = tmp_path / "raw" / "laps.parquet"
    extract_selected(archive, destination, record)
    assert destination.read_bytes() == content

    record["sha256"] = "0" * 64
    destination.unlink()
    with pytest.raises(ValueError, match="SHA-256"):
        extract_selected(archive, destination, record)
    assert not destination.exists()


def test_verify_file_rejects_wrong_size(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="Wrong size"):
        verify_file(path, {"size_bytes": 4, "sha256": "0" * 64})


def test_extract_package_keeps_dataset_and_outputs_but_skips_cache(tmp_path):
    archive = tmp_path / "source.zip"
    member = "source/datasets/laps.parquet"
    content = b"PAR1testPAR1"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(member, content)
        zip_file.writestr("source/paper_outputs/dataset_card.json", b"{}")
        zip_file.writestr("source/fastf1_cache/cache.ff1pkl", b"cache")
    record = {"archive_member": member, "sha256": hashlib.sha256(content).hexdigest()}
    raw = tmp_path / "raw"

    assert extract_package(archive, raw, record) == 2
    assert (raw / "f1_safetycar/datasets/laps.parquet").read_bytes() == content
    assert (raw / "f1_safetycar/paper_outputs/dataset_card.json").is_file()
    assert not (raw / "f1_safetycar/fastf1_cache").exists()
    assert extract_package(archive, raw, record) == 2


def test_extract_package_rejects_traversal_before_writing(tmp_path):
    archive = tmp_path / "source.zip"
    member = "source/datasets/laps.parquet"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr(member, b"ok")
        zip_file.writestr("source/datasets/../../outside.txt", b"bad")
    record = {"archive_member": member, "sha256": hashlib.sha256(b"ok").hexdigest()}
    raw = tmp_path / "raw"

    with pytest.raises(ValueError, match="Unsafe"):
        extract_package(archive, raw, record)
    assert not (raw / "f1_safetycar/datasets/laps.parquet").exists()


def test_verify_extracted_works_without_zip_and_rejects_missing_files(tmp_path):
    raw = tmp_path / "raw"
    datasets = raw / "f1_safetycar" / "datasets"
    datasets.mkdir(parents=True)
    content = b"PAR1testPAR1"
    (datasets / "laps.parquet").write_bytes(content)
    (datasets / "event=one_session=R.parquet").write_bytes(b"event")
    (raw / "laps.parquet").write_bytes(content)
    record = {
        "filename": "laps.parquet",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    expected = {"file_count": 2, "event_parquet_files": 1}
    assert verify_extracted(raw, record, expected) == 2
    (datasets / "event=one_session=R.parquet").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        verify_extracted(raw, record, expected)


def test_inventory_counts_duplicate_key_and_bad_lap():
    frame = pd.DataFrame(
        {
            "Year": [2022, 2022],
            "EventName": ["Pre-Season Test", "Pre-Season Test"],
            "Session": ["R", "R"],
            "Driver": ["ABC", "ABC"],
            "LapNumber": [1.0, 1.0],
            "LapTime": [90.0, 90.0],
            "LapTimeSec": [90.0, -2.0],
            "Sector1Time": [30.0, 30.0],
            "Sector2Time": [30.0, 30.0],
            "Sector3Time": [30.0, 30.0],
            "Humidity": [50.0, -1.0],
            "Compound": ["None", "SOFT"],
        }
    )
    report = inspect_frame(frame)
    assert report["candidate_key_duplicate_rows"] == 1
    assert report["invalid_or_suspect"]["lap_time_sec_nonpositive"] == 1
    assert report["invalid_or_suspect"]["lap_time_vs_lap_time_sec_mismatch"] == 1
    assert report["coverage"]["preseason_rows"] == 2
    assert report["coverage"]["event_names_by_year"]["2022"] == ["Pre-Season Test"]
    assert report["invalid_or_suspect"]["Humidity_out_of_range"] == 1
    assert report["invalid_or_suspect"]["Compound_placeholder_count"] == 1
    compound = next(row for row in schema_rows(frame) if row["column"] == "Compound")
    assert compound["missing_count"] == 0
    assert compound["placeholder_count"] == 1
