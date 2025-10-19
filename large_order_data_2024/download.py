"""Download and extract the large Parquet dataset from Google Drive."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Optional

import gdown

FILE_ID = "154rC3fn-JsoxKvCNTHAEIgv8HXBVB3xm"
ZIP_PATH = Path(__file__).resolve().parent / "large_order_data_10M_2024.zip"
EXTRACT_DIR = ZIP_PATH.parent
EXPECTED_SHA256: Optional[str] = None  # ใส่ค่า checksum ถ้าต้องการตรวจสอบไฟล์


def download_and_extract(force: bool = False, cleanup_zip: bool = True) -> Path:
    """
    Download the ZIP file from Google Drive and extract its contents.

    Parameters
    ----------
    force : bool
        True = ดาวน์โหลดใหม่เสมอแม้ไฟล์ ZIP จะมีอยู่แล้ว
    cleanup_zip : bool
        True = ลบไฟล์ ZIP หลังแตกไฟล์เสร็จ

    Returns
    -------
    Path
        โฟลเดอร์ที่มีไฟล์ที่ถูกแตกออกมา
    """
    if ZIP_PATH.exists() and not force:
        print(f"[skip] {ZIP_PATH} already exists (use force=True to re-download).")
    else:
        _download_zip()

    if EXPECTED_SHA256:
        checksum = _sha256_of(ZIP_PATH)
        if checksum != EXPECTED_SHA256:
            ZIP_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch: expected {EXPECTED_SHA256}, got {checksum}"
            )
        print(f"[verify] SHA-256 OK: {checksum}")

    extracted = _extract_zip(ZIP_PATH, EXTRACT_DIR)

    if cleanup_zip:
        ZIP_PATH.unlink(missing_ok=True)
        print(f"[cleanup] Removed {ZIP_PATH}")

    return extracted


def _download_zip() -> None:
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching {url} -> {ZIP_PATH}")
    gdown.download(url, str(ZIP_PATH), quiet=False)


def _extract_zip(zip_path: Path, target_dir: Path) -> Path:
    print(f"[extract] Extracting {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)

    # ถ้ารู้ว่าข้างในเป็นไฟล์เดียว ให้คืน path นั้น
    members = archive.namelist()
    if len(members) == 1:
        extracted_path = target_dir / members[0]
        print(f"[extract] Extracted {extracted_path}")
        return extracted_path

    print(f"[extract] Extracted {len(members)} items into {target_dir}")
    return target_dir


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    download_and_extract()
