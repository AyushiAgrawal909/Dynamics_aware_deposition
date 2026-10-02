#!/usr/bin/env python3
"""Run assertion-style test functions without requiring pytest."""

from __future__ import annotations

import argparse
import runpy
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run_file(path: Path) -> tuple[int, int]:
    namespace = runpy.run_path(str(path))
    tests = sorted(
        (name, value)
        for name, value in namespace.items()
        if name.startswith("test_") and callable(value)
    )
    passed = 0
    for name, test in tests:
        label = f"{path.name}::{name}"
        try:
            test()
        except Exception:
            print(f"FAIL {label}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {label}")
    return passed, len(tests)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    total_passed = 0
    total_tests = 0
    for path in args.files:
        passed, count = run_file(path)
        total_passed += passed
        total_tests += count
    print(f"{total_passed}/{total_tests} tests passed")
    if total_passed != total_tests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
