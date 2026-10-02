# PFMG Prerequisite Finding for the Momentum Audit

## Status

The full-run momentum comparison is blocked because the current periodic PFMG
path does not converge on the audit case. No force or momentum result from an
unconverged field may be used to claim conservation or nonconservation at a
specified solver tolerance.

## Executed Production-Path Tests

All tests used the same periodic HEXAPIC Landau input and the real KDE deposit,
PFMG potential solve, derivative-of-shape gather, and particle mover.

| Job | Operator treatment | Result |
|---|---|---|
| 4854047 | production negative-semidefinite matrix | 500 iterations; residual about `9e30` |
| 4854110 | algebraically equivalent SPD sign | 500 iterations; residual about `1e31` |
| 4854123 | SPD plus symmetric point gauge | 500 iterations; residual about `4e10` |
| 4854135 | SPD plus uniform `1e-6` shift | 500 iterations; residual about `1e31` |

The independent stencil residual agrees in order of magnitude with HYPRE's
reported residual. Requested tolerances `1e-5` and `1e-9` produce identical
fields because both runs stop at `max_iter=500`.

The point-gauge test reduced the divergence substantially but did not produce a
valid solve. The apparent one-step total-force defects from these runs are not
publication results because their force magnitudes are unphysical.

## Instrumentation Added

The audit now supports:

- warm or true HYPRE cold start (`HYPRE_StructPFMGSetZeroGuess`);
- independently computed periodic stencil residual;
- instantaneous `sum(qE)` and normalized total force;
- pre/post-push total particle momentum;
- cumulative momentum drift and integrated electric impulse;
- equal and unequal prescribed KDE widths.

## Gate

Before the 1500-step warm/cold array is run, a short test must satisfy both:

```text
iterations < max_iter
manual_relative_residual <= requested_tolerance (within a small reporting factor)
```

Only then can the audit distinguish solver-residual leakage from the intrinsic
translation defect of the variational grid gather. Job `4854145` tests PFMG's
documented default weighted-Jacobi relaxation in place of the production
red/black Gauss-Seidel override.

