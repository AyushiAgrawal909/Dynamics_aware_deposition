#!/usr/bin/env python3
"""Build an optional deterministic file inventory and SHA-256 checksum list."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
    "_matplotlib_cache",
    "numba_cache",
    "reproduced",
    ".venv",
}
EXCLUDED_TOP_LEVEL = {
    Path("FILE_INVENTORY.csv"),
    Path("SHA256SUMS.txt"),
}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and relative not in EXCLUDED_TOP_LEVEL
        and path.suffix != ".pyc"
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = sorted((path for path in ROOT.rglob("*") if included(path)), key=lambda p: p.as_posix())
    rows = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    with (ROOT / "FILE_INVENTORY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    final_paths = paths + [ROOT / "FILE_INVENTORY.csv"]
    final_paths.sort(key=lambda path: path.as_posix())
    with (ROOT / "SHA256SUMS.txt").open("w", encoding="utf-8") as handle:
        for path in final_paths:
            handle.write(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")

    print(
        f"Recorded {len(final_paths)} files and "
        f"{sum(path.stat().st_size for path in final_paths)} bytes"
    )


if __name__ == "__main__":
    main()
