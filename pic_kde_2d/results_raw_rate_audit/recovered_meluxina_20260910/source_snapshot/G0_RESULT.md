# G0 Result: Covariance Opportunity Passes, Conservation Precheck Stops H2

## Decision

`STOP_H2_VARIATIONAL_PROFILE_NOT_IMPLEMENTED`.

The covariance calculation clears its numerical headroom threshold, including
against the analytic Landau ceiling. The exact profile is not representable by
the validated variational production chain, so neither the analytic H2 rung nor
learned training is authorized.

## Numerical result

The originally proposed approximately 64-unit box fails before dynamics:

- the 10-percent core contains only `0.4` carrier cycles;
- the relative packet bandwidth is `1.592`;
- the largest admissible field-noise gain is `1.325x`, below the `1.5x` gate;
- the density-noise gain at that point is `8.503x`.

The fixed-resolution narrowband candidate is ten times longer in x:

- four carrier cycles in the protected core;
- relative packet bandwidth `0.159`;
- best admissible `sigma_free = 384 dx`;
- packet-field noise gain `1.5953x`;
- domain density-noise gain `9.0206x`;
- deterministic protected-field error `2.253 percent`;
- ideal outside-noise-deletion upper bound `3.4956x`.

Against global/protected `sigma = 2.453 dx`, the same calculation gives:

- packet-field noise gain `1.6223x`;
- domain density-noise gain `8.8604x`;
- deterministic protected-field error `2.586 percent`.

The baseline choice therefore does not create the pass.

## Conservation precheck

The validated variational wrapped-Gaussian KDE has
`KDE_H_MAX_CELLS = 20`. The requested free width is `384 dx`; running it
without a new implementation silently clamps the profile. At the largest
supported width (`20 dx`), the field-noise gain is only `1.000033x`, far below
the `1.5x` gate.

The wrapped kernel also sums only three periodic images on either side. For the
thin y axis (`31` periodic cells), this is not a valid representation of a
`384 dx` isotropic Gaussian.

Consequently:

- grad-h force magnitude: not measured; requested profile is unrepresentable;
- integrated momentum drift: not measured for the same reason;
- static refresh work: exactly zero, because a static profile has no refresh;
- H2 status: stopped before geometry or force execution.

Substituting the clamped `1 -> 20 dx` profile would answer a different question
and is explicitly prohibited.

## Interpretation

The logarithmic estimate is directionally correct but not an implementation
model. A finite Gaussian does not delete distant source noise because its net
charge remains visible outside the kernel. The field gain only clears `1.5x`
when the free-region Gaussian is very broad.

That broad width is representable by the Python binned spectral-W estimator,
where cost scales with width-bin FFTs. That estimator is not the validated
variational grad-h chain used by the conservation audit. Changing to it would
be a method change requiring its own moving-mask refresh and energy audit, not
a continuation of the present gate.

Because a Langmuir packet moves, a frozen initial particle-to-bin assignment is
not sufficient. The analytic mask must follow the packet causally, refresh
particle bin assignments, and include the preregistered refresh-work audit.

## Reopening condition

H2 may be reconsidered only after one chain simultaneously supports the
`2.453 -> 384 dx` moving C0 profile, matched deposition/gather, refresh-work
accounting, and a completed grad-h/momentum audit. No geometry build or network
training starts before that implementation exists.
