#!/usr/bin/env python3
"""Run the  examples and reproduce the archived reference results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
REPRO = ROOT / "examples" / "paper_results"
EXACT_WIDTH = (
    ROOT
    / "campaigns"
    / "E2_q050_x000_exact_width_vb080_v1_job5233486"
)
EXACT_WIDTH_CONFIG_SHA256 = (
    "3b763155abe28c4b105a1824969ff411a6d0b1b6a5f3d9319b23ff3c39ec4800"
)
EXPECTED_DIRS = (
    "examples/paper_results",
    "pic_kde_2d",
    "pic_kde_2d_landau_cv_stage",
    "campaigns/E2_q050_x000_exact_width_vb080_v1_job5233486",
)
EXPECTED_TEST_COUNT = 87
CORE_HASHES = {
    Path("campaigns/E2_q050_x000_exact_width_vb080_v1_job5233486/raw_remote_output/frozen_config.json"):
        EXACT_WIDTH_CONFIG_SHA256,
    Path("campaigns/E2_q050_x000_exact_width_vb080_v1_job5233486/raw_remote_output/analysis/summary.json"):
        "70a1a6ed7ed392a7bbfc962f6807b18aa9070602e74bf9a05129bd9bb9742b26",
}
IGNORED_RUNTIME_PARTS = {
    ".git",
    ".venv",
    "reproduced",
    "__pycache__",
    ".pytest_cache",
    "_matplotlib_cache",
    "numba_cache",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def new_output(label: str, requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = ROOT / "reproduced" / f"{label}_{stamp}"
    else:
        output = requested if requested.is_absolute() else ROOT / requested
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def verify() -> None:
    missing = [name for name in EXPECTED_DIRS if not (ROOT / name).is_dir()]
    if missing:
        raise FileNotFoundError("missing required package directories: " + ", ".join(missing))

    failures: list[str] = []
    for relative, expected in CORE_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing required reference input: {relative}")
        elif sha256(path) != expected:
            failures.append(f"core reference hash mismatch: {relative}")

    checksum_path = ROOT / "SHA256SUMS.txt"
    checked = 0
    if checksum_path.exists():
        with checksum_path.open(encoding="utf-8") as handle:
            for line in handle:
                expected, relative = line.rstrip("\n").split("  ", 1)
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    failures.append(f"unsafe checksum path: {relative}")
                    continue
                path = ROOT / relative_path
                if not path.is_file():
                    failures.append(f"missing: {relative}")
                elif sha256(path) != expected:
                    failures.append(f"hash mismatch: {relative}")
                checked += 1
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_RUNTIME_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            target = path.resolve(strict=False)
            try:
                target.relative_to(ROOT)
            except ValueError:
                failures.append(f"external symlink: {relative} -> {target}")
    if failures:
        raise RuntimeError("package verification failed:\n" + "\n".join(failures[:50]))

    print("Required directories and core reference hashes verified.")
    if checksum_path.exists():
        print(f"Full repository integrity verified for {checked} files.")
    else:
        print("SHA256SUMS.txt is absent; optional full-file verification was skipped.")
    print(f"Recorded scientific regression target: {EXPECTED_TEST_COUNT} unittest cases.")


def tests() -> None:
    verify()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(REPRO / "tests"),
            "-p",
            "test_*.py",
        ],
        env=env,
    )


def analysis_environment(cache: Path) -> dict[str, str]:
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cache),
            "XDG_CACHE_HOME": str(cache),
        }
    )
    return env


def bounded(output_argument: Path | None, bootstrap: int) -> None:
    output = new_output("bounded", output_argument)
    env = analysis_environment(output.parent / ".mpl-cache-bounded")
    run(
        [
            sys.executable,
            str(REPRO / "analyze_bounded_floor_estimand_reference_20260914.py"),
            "--output",
            str(output),
            "--bootstrap",
            str(bootstrap),
        ],
        env=env,
    )
    print(f"Bounded analysis written to {output}")


def completion(output_argument: Path | None) -> None:
    output = new_output("research_completion", output_argument)
    env = analysis_environment(output.parent / ".mpl-cache-completion")
    run(
        [
            sys.executable,
            str(REPRO / "rebuild_research_completion.py"),
            "--output-root",
            str(output),
        ],
        env=env,
    )
    print(f"Research-completion rebuild written to {output}")


def exact_width(output_argument: Path | None) -> None:
    """Re-run the frozen job-5233486 reducer from its archived histories."""
    output = new_output("exact_width", output_argument)
    env = analysis_environment(output.parent / ".mpl-cache-exact-width")
    run(
        [
            sys.executable,
            str(EXACT_WIDTH / "frozen_campaign_package/code/analyze_E2_q050_exact.py"),
            "--config",
            str(EXACT_WIDTH / "raw_remote_output/frozen_config.json"),
            "--config-sha256",
            EXACT_WIDTH_CONFIG_SHA256,
            "--input",
            str(EXACT_WIDTH / "raw_remote_output/runs"),
            "--output",
            str(output),
        ],
        env=env,
    )
    run(
        [
            sys.executable,
            str(EXACT_WIDTH / "verify_reduction.py"),
            "--candidate",
            str(output),
        ]
    )
    print(f"Exact-width reduction written to {output}")


def reproduce_all(output_argument: Path | None, bootstrap: int) -> None:
    """Run every maintained existing-data reproduction workflow."""
    output = new_output("all_results", output_argument)
    output.mkdir(parents=True)
    tests()
    bounded(output / "bounded", bootstrap)
    completion(output / "completion")
    exact_width(output / "exact_width")
    summary = {
        "status": "COMPLETE",
        "scope": "existing-data scientific reproduction; no simulation submission",
        "regression_tests": EXPECTED_TEST_COUNT,
        "products": {
            "bounded": "bounded",
            "completion": "completion",
            "exact_width": "exact_width",
        },
    }
    (output / "REPRODUCTION_COMPLETE.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"All maintained scientific reproductions written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "verify", help="verify core inputs and, when present, the full checksum list"
    )
    subparsers.add_parser("tests", help="run the archived scientific regressions")

    bounded_parser = subparsers.add_parser("bounded", help="rebuild the latest bounded analysis")
    bounded_parser.add_argument("--output", type=Path)
    bounded_parser.add_argument("--bootstrap", type=int, default=10000)

    completion_parser = subparsers.add_parser(
        "completion", help="rebuild affected research-completion evidence from raw histories"
    )
    completion_parser.add_argument("--output", type=Path)

    exact_width_parser = subparsers.add_parser(
        "exact-width",
        help="rebuild and verify the completed q050 exact-width reduction",
    )
    exact_width_parser.add_argument("--output", type=Path)

    all_parser = subparsers.add_parser(
        "all", help="run tests and all maintained existing-data reproductions"
    )
    all_parser.add_argument("--output", type=Path)
    all_parser.add_argument("--bootstrap", type=int, default=10000)

    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "tests":
        tests()
    elif args.command == "bounded":
        bounded(args.output, args.bootstrap)
    elif args.command == "completion":
        completion(args.output)
    elif args.command == "exact-width":
        exact_width(args.output)
    elif args.command == "all":
        reproduce_all(args.output, args.bootstrap)


if __name__ == "__main__":
    main()
