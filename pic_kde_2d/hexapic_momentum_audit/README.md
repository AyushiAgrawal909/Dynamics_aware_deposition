# HEXAPIC Momentum-Conservation Audit

This audit tests the production periodic HYPRE/PFMG path, not the idealized
spectral `S.T @ A @ S` model.

The default production behavior is unchanged. The patch adds three opt-in
environment variables:

- `HEXAPIC_MOMENTUM_AUDIT=1` logs the PFMG residual and the gathered total
  electric force `sum_p(q_p E_p)` plus its normalization by
  `sum_p(abs(q_p E_p))`.
- `HEXAPIC_HYPRE_TOL` overrides the existing default PFMG tolerance `1e-9`.
- `HEXAPIC_HYPRE_SPD_FORM=1` multiplies both the periodic Poisson matrix and
  right-hand side by `-1`. This leaves the mathematical solution unchanged but
  presents PFMG with its expected positive-semidefinite operator.
- `HEXAPIC_HYPRE_COLD_START=1` zeros the PFMG solution vector before each solve;
  `0` retains the production warm start from the previous potential.
- `HEXAPIC_HYPRE_PIN_GAUGE=1` removes the periodic constant nullspace by fixing
  `phi(0,0)=0` and symmetrically eliminating that degree of freedom. It is an
  audit-only prerequisite for a meaningful PFMG tolerance comparison.
- `HEXAPIC_HYPRE_DIAGONAL_SHIFT` adds a uniform positive diagonal shift to the
  SPD periodic operator. Unlike point pinning, it preserves the geometric
  stencil; the audit uses `1e-6` and keeps the mean source at zero.
- `HEXAPIC_HYPRE_RELAX_TYPE` selects the PFMG smoother. The production setting
  is `2` (symmetric red/black Gauss-Seidel); the nullspace audit tests `1`
  (weighted Jacobi), HYPRE's documented default for PFMG.
- `HEXAPIC_KDE_SPLIT_SIGMA_CELLS=h_left,h_right` prescribes a fixed bandwidth
  in each half of the periodic x domain. Equal values test a global shape;
  unequal values test spatially varying per-particle shapes through the exact
  production KDE deposit and gather.

The matrix compares `(0.5,0.5)`, `(3,3)`, and `(0.5,3)` cell widths at PFMG
tolerances `1e-5` and `1e-9`, using the same seed and one field/push step.

Exact particle momentum conservation requires the reported normalized total
force to be at roundoff. A dependence on PFMG residual diagnoses solver error;
a nonzero floor that remains at `1e-9` diagnoses the discrete deposit/gather
chain itself.

For long runs the audit also reports the cumulative particle momentum drift,
the integrated electric impulse `sum_n F_n*dt`, and their balance error. The
momentum comparison includes the nonzero random initial momentum by subtracting
the measured pre-push value at step zero.
