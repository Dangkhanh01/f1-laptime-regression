"""Verify extracted data or extract it from the pinned Mendeley ZIP."""

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_file(path: Path, record: dict) -> None:
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"Wrong size: {path}")
    if sha256(path) != record["sha256"]:
        raise ValueError(f"Wrong SHA-256: {path}")


def store_file(source, destination: Path, record: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(source, temporary)
        verify_file(temporary_path, record)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def extract_selected(archive: Path, destination: Path, record: dict) -> None:
    if destination.exists():
        verify_file(destination, record)
        return
    with zipfile.ZipFile(archive) as zip_file:
        member = zip_file.getinfo(record["archive_member"])
        if member.is_dir() or member.file_size != record["size_bytes"]:
            raise ValueError("Selected ZIP member has wrong size or type")
        with zip_file.open(member) as source:
            store_file(source, destination, record)


def extract_package(archive: Path, raw_root: Path, selected_record: dict) -> int:
    member_name = selected_record["archive_member"]
    if "/datasets/" not in member_name:
        raise ValueError("Selected file is not inside datasets/")
    prefix = member_name.split("/datasets/", 1)[0]
    prefix_parts = prefix.split("/")
    allowed = (f"{prefix}/datasets/", f"{prefix}/paper_outputs/")
    raw_root = raw_root.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        members = [
            info
            for info in zip_file.infolist()
            if not info.is_dir() and info.filename.startswith(allowed)
        ]
        if member_name not in {info.filename for info in members}:
            raise ValueError("Selected file is missing from ZIP")
        destinations = []
        seen = set()
        for info in members:
            parts = info.filename.split("/")
            if (
                any(part in ("", ".", "..") for part in parts)
                or "\\" in info.filename
                or ":" in info.filename
                or stat.S_ISLNK(info.external_attr >> 16)
                or info.filename.casefold() in seen
            ):
                raise ValueError(f"Unsafe or duplicate ZIP member: {info.filename}")
            seen.add(info.filename.casefold())
            destination = raw_root.joinpath("f1_safetycar", *parts[len(prefix_parts) :])
            if not destination.resolve().is_relative_to(raw_root) or destination.is_symlink():
                raise ValueError(f"ZIP member escapes data/raw: {info.filename}")
            destinations.append(destination)
        for info, destination in zip(members, destinations, strict=True):
            with zip_file.open(info) as source:
                record = {
                    "size_bytes": info.file_size,
                    "sha256": hashlib.file_digest(source, "sha256").hexdigest(),
                }
            if info.filename == member_name and record["sha256"] != selected_record["sha256"]:
                raise ValueError("Selected file hash differs from manifest")
            if destination.exists():
                verify_file(destination, record)
            else:
                with zip_file.open(info) as source:
                    store_file(source, destination, record)
    return len(members)


def verify_extracted(raw_root: Path, selected_record: dict, expected: dict) -> int:
    package = raw_root / "f1_safetycar"
    files = [path for path in package.rglob("*") if path.is_file()]
    events = list((package / "datasets").rglob("event=*_session=*.parquet"))
    if len(files) != expected["file_count"] or len(events) != expected["event_parquet_files"]:
        raise ValueError("Extracted files are incomplete; restore them from Git or --archive")
    verify_file(package / "datasets" / selected_record["filename"], selected_record)
    verify_file(raw_root / selected_record["filename"], selected_record)
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing ZIP, for example from Downloads")
    parser.add_argument("--url", help="Direct HTTPS URL for the ZIP, if available")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archive_record = manifest["archive"]
    selected_record = manifest["selected_file"]
    raw_root = ROOT / "data" / "raw"
    selected = ROOT / selected_record["path"]
    if args.archive and args.url:
        parser.error("Use --archive or --url, not both")
    if args.archive:
        verify_file(args.archive, archive_record)
        extract_package(args.archive, raw_root, selected_record)
        extract_selected(args.archive, selected, selected_record)
    elif args.url:
        if not args.url.startswith("https://"):
            raise ValueError("Download URL must use HTTPS")
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / archive_record["filename"]
            with urllib.request.urlopen(args.url, timeout=60) as response:
                store_file(response, archive, archive_record)
            extract_package(archive, raw_root, selected_record)
            extract_selected(archive, selected, selected_record)
    count = verify_extracted(raw_root, selected_record, manifest["extracted"])
    print(f"Verified {count} source files and {selected.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
