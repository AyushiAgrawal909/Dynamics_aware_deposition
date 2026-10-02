"""Wu-Qin bandwidth-selection and update semantics.

Wu and Qin minimize least-squares leave-one-out cross validation (LSCV) as a
proxy for MISE.  The loss itself may have no finite minimizer when the sampled
density is close to uniform.  Their time-dependent algorithm therefore gates
re-optimization with a uniformity-test p-value instead of repeatedly widening
the kernel.

This module deliberately separates:

* identifying an interior minimum of a sampled CV curve;
* reporting boundary minima as censored, not finite optima;
* deciding when the Wu-Qin controller is allowed to request a new optimum; and
* relaxing an accepted target into the live width.

The caller supplies the Anderson-Darling p-value.  This keeps the state machine
independent of a particular statistical package and makes its behavior easy to
test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CVMinimum:
    """Identification status for a CV minimum on a finite search grid."""

    status: str
    identified: bool
    bandwidth: Optional[float]
    score: float
    index: int
    search_min: float
    search_max: float
    lower_bound: Optional[float]
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def identify_cv_minimum(
    bandwidths: Sequence[float],
    scores: Sequence[float],
    *,
    relative_tie_tolerance: float = 1.0e-12,
    absolute_tie_tolerance: float = 0.0,
) -> CVMinimum:
    """Identify a finite interior minimum or return an explicit censoring state.

    A minimum at the largest searched width only establishes
    ``h_opt >= h_search_max``.  It must not be returned as a finite optimum.
    A plateau that reaches the upper boundary is treated the same way.
    """
    h = np.asarray(bandwidths, dtype=np.float64)
    loss = np.asarray(scores, dtype=np.float64)
    if h.ndim != 1 or loss.ndim != 1 or h.size != loss.size or h.size < 2:
        raise ValueError("bandwidths and scores must be equal-length 1D arrays")
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(loss)):
        raise ValueError("bandwidths and scores must be finite")
    if np.any(h < 0.0) or np.any(np.diff(h) <= 0.0):
        raise ValueError("bandwidths must be nonnegative and strictly increasing")
    if relative_tie_tolerance < 0.0 or absolute_tie_tolerance < 0.0:
        raise ValueError("tie tolerances must be nonnegative")

    index = int(np.argmin(loss))
    minimum = float(loss[index])
    tolerance = max(
        absolute_tie_tolerance,
        relative_tie_tolerance * max(abs(minimum), 1.0),
    )
    tied = np.flatnonzero(loss <= minimum + tolerance)
    reaches_left = int(tied[0]) == 0
    reaches_right = int(tied[-1]) == h.size - 1

    if reaches_right:
        return CVMinimum(
            status="right_censored",
            identified=False,
            bandwidth=None,
            score=minimum,
            index=index,
            search_min=float(h[0]),
            search_max=float(h[-1]),
            lower_bound=float(h[tied[0]]),
            explanation=(
                "CV does not rise before the upper search boundary; only a "
                "lower bound on the density-optimal bandwidth is identified."
            ),
        )
    if reaches_left:
        return CVMinimum(
            status="left_censored",
            identified=False,
            bandwidth=None,
            score=minimum,
            index=index,
            search_min=float(h[0]),
            search_max=float(h[-1]),
            lower_bound=None,
            explanation=(
                "CV is minimized at the lower search boundary; a finite "
                "interior optimum is not identified."
            ),
        )
    return CVMinimum(
        status="interior",
        identified=True,
        bandwidth=float(h[index]),
        score=minimum,
        index=index,
        search_min=float(h[0]),
        search_max=float(h[-1]),
        lower_bound=None,
        explanation="A finite interior minimum is identified on the search grid.",
    )


@dataclass
class WuQinWidthUpdater:
    """State machine for the width-update strategy in Wu and Qin, Sec. II.D.

    The defaults reproduce the paper's stated controls:

    * reject uniformity at ``p <= 0.01``;
    * re-optimize only after the p-value falls by another factor of two; and
    * relax toward a finite target with rate ``0.05``.

    If CV is right-censored, the unconstrained controller holds its current
    width.  Supplying ``dynamics_cap`` converts a decreasing density objective
    into the constrained solution at that cap.
    """

    current_width: float
    uniformity_threshold: float = 0.01
    adjust_rate: float = 0.05
    target_width: Optional[float] = None
    p_value_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.current_width) or self.current_width < 0.0:
            raise ValueError("current_width must be finite and nonnegative")
        if not 0.0 < self.uniformity_threshold < 1.0:
            raise ValueError("uniformity_threshold must lie in (0, 1)")
        if not 0.0 < self.adjust_rate <= 1.0:
            raise ValueError("adjust_rate must lie in (0, 1]")
        if self.target_width is None:
            self.target_width = float(self.current_width)
        if self.p_value_threshold is None:
            self.p_value_threshold = 2.0 * self.uniformity_threshold

    def step(
        self,
        p_value: float,
        selection: Optional[CVMinimum] = None,
        *,
        dynamics_cap: Optional[float] = None,
    ) -> dict:
        """Advance one update decision and return an auditable event record."""
        p = float(p_value)
        if not np.isfinite(p) or not 0.0 <= p <= 1.0:
            raise ValueError("p_value must lie in [0, 1]")
        if dynamics_cap is not None and (
            not np.isfinite(dynamics_cap) or dynamics_cap < 0.0
        ):
            raise ValueError("dynamics_cap must be finite and nonnegative")

        trigger = False
        accepted = False
        previous = float(self.current_width)
        if p > 2.0 * self.uniformity_threshold:
            self.p_value_threshold = 2.0 * self.uniformity_threshold
            self.target_width = self.current_width
            reason = "uniform_hold"
        elif p <= 0.5 * float(self.p_value_threshold):
            trigger = True
            self.p_value_threshold = p
            if selection is None:
                raise ValueError("a CV selection is required when the AD gate triggers")
            if selection.identified:
                target = float(selection.bandwidth)
                if dynamics_cap is not None:
                    target = min(target, float(dynamics_cap))
                self.target_width = target
                accepted = True
                reason = "finite_cv_target"
            elif selection.status == "right_censored" and dynamics_cap is not None:
                self.target_width = float(dynamics_cap)
                accepted = True
                reason = "right_censored_cv_capped_by_dynamics"
            else:
                self.target_width = self.current_width
                reason = f"{selection.status}_hold"
        else:
            reason = "p_value_change_too_small_hold"

        if accepted:
            self.current_width = (
                self.adjust_rate * float(self.target_width)
                + (1.0 - self.adjust_rate) * self.current_width
            )
        return {
            "p_value": p,
            "triggered_cv": trigger,
            "accepted_target": accepted,
            "reason": reason,
            "previous_width": previous,
            "target_width": float(self.target_width),
            "current_width": float(self.current_width),
            "p_value_threshold": float(self.p_value_threshold),
        }


def silverman_local_width_factors(
    pilot_density_at_particles: Sequence[float],
    sensitivity: float = 0.5,
    density_floor: Optional[float] = None,
) -> np.ndarray:
    """Return Wu-Qin/Silverman adaptive factors ``lambda_i=(f_i/g)^-alpha``."""
    density = np.asarray(pilot_density_at_particles, dtype=np.float64)
    if density.ndim != 1 or density.size == 0 or np.any(~np.isfinite(density)):
        raise ValueError("pilot density must be a nonempty finite 1D array")
    if np.any(density < 0.0):
        raise ValueError("pilot density cannot be negative")
    if not np.isfinite(sensitivity) or sensitivity < 0.0:
        raise ValueError("sensitivity must be finite and nonnegative")
    positive = density[density > 0.0]
    if positive.size == 0:
        raise ValueError("pilot density must contain a positive value")
    floor = (
        max(float(np.min(positive)) * 1.0e-12, np.finfo(np.float64).tiny)
        if density_floor is None
        else float(density_floor)
    )
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("density_floor must be finite and positive")
    safe = np.maximum(density, floor)
    geometric_mean = float(np.exp(np.mean(np.log(safe))))
    return np.power(safe / geometric_mean, -sensitivity)
