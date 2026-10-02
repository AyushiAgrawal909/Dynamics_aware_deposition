"""Frozen candidate design for the Task-9 E2 anisotropic search."""

from __future__ import annotations

import math


Q_BUDGET_GRID_TIME = 0.009086217804777913
Q_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
X_ALLOCATIONS = tuple(index / 20.0 for index in range(21))


def make_design(q_budget: float = Q_BUDGET_GRID_TIME) -> list[dict]:
    """Return CIC plus an 84-point interior/boundary feasible design."""
    coefficient_x = (2.0 * math.pi * 12.0 / 128.0) ** 2
    coefficient_y = (2.0 * math.pi / 16.0) ** 2
    rows = [
        {
            "label": "base",
            "q_fraction": 0.0,
            "x_allocation": 0.0,
            "sigma_x_dx": 0.0,
            "sigma_y_dy": 0.0,
            "q_total": 0.0,
        }
    ]
    for radial_fraction in Q_FRACTIONS:
        q_total = radial_fraction * q_budget
        for x_allocation in X_ALLOCATIONS:
            qx = x_allocation * q_total
            qy = (1.0 - x_allocation) * q_total
            rows.append(
                {
                    "label": f"q{int(round(100 * radial_fraction)):03d}_x{int(round(100 * x_allocation)):03d}",
                    "q_fraction": radial_fraction,
                    "x_allocation": x_allocation,
                    "sigma_x_dx": math.sqrt(qx / coefficient_x),
                    "sigma_y_dy": math.sqrt(qy / coefficient_y),
                    "q_total": q_total,
                }
            )
    if len(rows) != 85 or len({row["label"] for row in rows}) != len(rows):
        raise AssertionError("dense anisotropic design construction failed")
    if max(row["q_total"] for row in rows) > q_budget * (1.0 + 1.0e-14):
        raise AssertionError("design exceeds the model-admissible budget")
    return rows

