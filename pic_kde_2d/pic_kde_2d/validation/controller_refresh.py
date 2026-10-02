"""Diagnostics for lagged learned-bandwidth controller refreshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class ControllerRefreshAudit:
    """Accumulate work and staleness diagnostics for one refresh cadence."""

    cadence_steps: int
    steps: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    energy_jumps: List[float] = field(default_factory=list)
    tensor_change_rms: List[float] = field(default_factory=list)
    staleness_error: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cadence_steps <= 0:
            raise ValueError("cadence_steps must be positive")

    def record(
        self,
        *,
        step: int,
        time: float,
        field_energy_before: float,
        field_energy_after: float,
        H_old: np.ndarray,
        H_new: np.ndarray,
        staleness_error: float,
    ) -> None:
        """Record one controller update with no intervening particle push."""
        old = np.asarray(H_old, dtype=np.float64)
        new = np.asarray(H_new, dtype=np.float64)
        values = np.array(
            [time, field_energy_before, field_energy_after, staleness_error],
            dtype=np.float64,
        )
        if old.shape != new.shape or old.shape[-2:] != (2, 2):
            raise ValueError("H_old and H_new must have equal trailing shape (2, 2)")
        if np.any(~np.isfinite(old)) or np.any(~np.isfinite(new)):
            raise ValueError("bandwidth tensors must be finite")
        if np.any(~np.isfinite(values)) or step < 0 or staleness_error < 0.0:
            raise ValueError("refresh diagnostics must be finite and nonnegative where required")

        self.steps.append(int(step))
        self.times.append(float(time))
        self.energy_jumps.append(float(field_energy_after - field_energy_before))
        self.tensor_change_rms.append(float(np.sqrt(np.mean((new - old) ** 2))))
        self.staleness_error.append(float(staleness_error))

    def summary(self) -> Dict[str, float]:
        jumps = np.asarray(self.energy_jumps, dtype=np.float64)
        tensor_change = np.asarray(self.tensor_change_rms, dtype=np.float64)
        staleness = np.asarray(self.staleness_error, dtype=np.float64)
        return {
            "cadence_steps": float(self.cadence_steps),
            "n_refreshes": float(jumps.size),
            "cumulative_signed_controller_work": float(np.sum(jumps)),
            "cumulative_absolute_controller_work": float(np.sum(np.abs(jumps))),
            "mean_absolute_energy_jump": float(np.mean(np.abs(jumps))) if jumps.size else 0.0,
            "max_absolute_energy_jump": float(np.max(np.abs(jumps))) if jumps.size else 0.0,
            "mean_tensor_change_rms": float(np.mean(tensor_change)) if jumps.size else 0.0,
            "max_staleness_error": float(np.max(staleness)) if jumps.size else 0.0,
        }


def cadence_sweep_table(
    audits: Iterable[ControllerRefreshAudit],
) -> List[Dict[str, float]]:
    """Return cadence summaries sorted from most to least frequent refresh."""
    rows = [audit.summary() for audit in audits]
    return sorted(rows, key=lambda row: row["cadence_steps"])
