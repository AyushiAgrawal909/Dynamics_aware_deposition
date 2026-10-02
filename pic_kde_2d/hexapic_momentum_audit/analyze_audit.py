"""Parse production HEXAPIC momentum-audit logs into a compact CSV."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


KEY_VALUE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def parse_line(line: str) -> dict[str, str]:
    return dict(KEY_VALUE.findall(line.strip()))


def parse_case(case_dir: Path) -> dict[str, object]:
    log = (case_dir / "run.log").read_text()
    hypre_rows = [parse_line(line) for line in log.splitlines()
                  if line.startswith("HYPRE_AUDIT step=")]
    force_rows = [parse_line(line) for line in log.splitlines()
                  if line.startswith("MOMENTUM_AUDIT step=")]
    if not hypre_rows or not force_rows:
        raise RuntimeError(f"missing audit records in {case_dir / 'run.log'}")
    hypre = hypre_rows[-1]
    force = force_rows[-1]
    parts = case_dir.name.split("__")
    return {
        "case": case_dir.name,
        "shape": parts[0],
        "requested_tolerance": float(parts[1].removeprefix("tol")),
        "initial_guess": parts[2] if len(parts) > 2 else "unspecified",
        "iterations": int(hypre["iterations"]),
        "final_relative_residual": float(hypre["final_relative_residual"]),
        "manual_relative_residual": float(hypre["manual_relative_residual"]),
        "force_x": float(force["force_x"]),
        "force_y": float(force["force_y"]),
        "abs_force_x": float(force["abs_force_x"]),
        "abs_force_y": float(force["abs_force_y"]),
        "relative_force_x": float(force["relative_x"]),
        "relative_force_y": float(force["relative_y"]),
        "momentum_drift_x": float(force["momentum_drift_x"]),
        "momentum_drift_y": float(force["momentum_drift_y"]),
        "integrated_force_dt_x": float(force["integrated_force_dt_x"]),
        "integrated_force_dt_y": float(force["integrated_force_dt_y"]),
        "balance_error_x": float(force["balance_error_x"]),
        "balance_error_y": float(force["balance_error_y"]),
    }


def main(root: Path) -> None:
    rows = [parse_case(path) for path in sorted(root.glob("*__tol*"))]
    output = root / "momentum_audit.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(output)
    for row in rows:
        exact = max(row["relative_force_x"], row["relative_force_y"]) < 1e-12
        print(
            f"{row['case']}: residual={row['manual_relative_residual']:.3e}, "
            f"relative_force=({row['relative_force_x']:.3e},"
            f" {row['relative_force_y']:.3e}), "
            f"drift=({row['momentum_drift_x']:.3e},"
            f" {row['momentum_drift_y']:.3e}), "
            f"{'ROUND-OFF' if exact else 'NONZERO'}"
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
