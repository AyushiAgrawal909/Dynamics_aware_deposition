# Paper 1 conservation audit

Audit date: 2026-09-09 (Europe/Luxembourg).

## Outcome

The static, globally filtered Python chain conserves represented charge and
matched spectral momentum to floating-point round-off.  Its energy is not an
exact invariant: the archived histories contain finite drift, and eight of the
nine benchmark/filter confidence intervals for a full-history linear drift
slope exclude zero.  A small fixed-space refinement test recovers the expected
second-order time convergence in successive differences, while the absolute
energy drift approaches a spatial/coupling floor.

The spatially varying HEXAPIC prototype has a different numerical formulation
and does **not** reach the spectral momentum floor.  The manuscript's 42.8%
grad-h improvement and 1817-fold discontinuity contrast are both x-component
comparisons.  The corresponding two-dimensional drift-norm changes are
+0.0014% and 16.03-fold.  A new four-point PFMG scan shows that tightening the
achieved residual through six orders of magnitude does not reduce the
one-step force, so the observed force floor is not controlled by solver
tolerance in this configuration.

## Evidence and preserved inputs

Historical outputs were copied without modification into:

* `pic_kde_2d/results/energy_conservation_5048437/` (job 5048437; 90 per-seed
  NPZ histories plus the original reducer output);
* `pic_kde_2d/results/hexapic_momentum_audit/grad_h_full_4855533/` (four
  1500-step cases, raw logs and diagnostics);
* `pic_kde_2d/results/hexapic_momentum_audit/grid_phase_full_4858823/` (four
  additional 1500-step mesh phases; phase zero is the smooth-with-grad-h case
  above);
* `pic_kde_2d/results/grid_phase_one_step_4858774/` (five-phase compact
  one-step table).

The new, diagnostic-only PFMG result is preserved at
`pic_kde_2d/results/hexapic_momentum_audit/hypre_tolerance_one_step_5195488/`.
It records the Git revision, dirty-tree status, executable hash and hashes of
the four relevant C++ sources.  It used the current executable with SHA-256
`5f29c47c00ef0ddb6908a7994eadd68cdf28cee61502c73c7bb1e8ef44e760d4`.
This hash is evidence for job 5195488 only; the old jobs did not embed an
executable hash.

The independent reducer is `reproducibility/analyze_conservation.py`.  Its
machine-readable products and plots are under
`reproducibility/results/conservation_audit/`.  The controlled local checks are
implemented by `reproducibility/run_conservation_small_tests.py` and written
under `reproducibility/results/conservation_audit/small_tests/`.

## Commands run

Archived-data reduction and figures:

```sh
env MPLCONFIGDIR=/private/tmp/paper1_matplotlib \
    XDG_CACHE_HOME=/private/tmp/paper1_cache MPLBACKEND=Agg \
    python3 paper1_source_package/reproducibility/analyze_conservation.py
```

Controlled operator, complete particle-force and time-step tests:

```sh
env MPLCONFIGDIR=/private/tmp/paper1_matplotlib \
    XDG_CACHE_HOME=/private/tmp/paper1_cache MPLBACKEND=Agg \
    python3 paper1_source_package/reproducibility/run_conservation_small_tests.py
```

Regression suite:

```sh
python3 -m unittest -v \
  paper1_source_package/reproducibility/tests/test_conservation_audit.py
```

Result: 6/6 tests passed.

The exact submitted tolerance diagnostic is
`reproducibility/run_hypre_tolerance_scan.sbatch`.  It was uploaded unchanged
and submitted with:

```sh
ssh u103576@meluxina \
  'cd /home/users/u103576/hexapic-main/pic_kde_2d/hexapic_momentum_audit && \
   sbatch run_hypre_tolerance_scan.sbatch'
```

Slurm job 5195488 completed in 26 s with all four tasks at exit code zero.
No production physics run was submitted.

## Charge

### Definitions

For the Python histories,

```text
epsilon_Q(t) = |Q_particles(t)-Q_particles(0)| / |Q_particles(0)|.
```

The uniform positive background is represented by zero-mode removal in the
periodic Poisson solve and is not included in `Q_particles`; using the nonzero
electron charge in the denominator avoids division by a neutral total.

For the HEXAPIC audit, the relevant normalized assembled error is

```text
|assembled_signed-expected_signed| / expected_abs,
```

where `expected_abs` is the sum of absolute macro-particle weights.  The audit
also checks each sampled particle shape's partition-of-unity error and the
absolute-weight defect.

### Results

* All 90 Python histories have bitwise-constant represented particle charge.
* Across the four 1500-step HEXAPIC cases, the maximum per-particle
  normalization error is `1.1102230246251565e-15` and the absolute-weight
  defect is zero.
* The largest assembled signed error is `2.7464e-17` of absolute weight among
  the smooth cases and `5.1326e-16` in the discontinuous negative control.

These values are round-off scale, but they do not support the old wording that
one value (`1.29e-17`) is independent of every width/profile.

## Total force and momentum

### Static matched filters

For a complete particle force evaluation the normalized diagnostic is

```text
R_F = ||sum_p q_p E_p||_2 / sum_p ||q_p E_p||_2.
```

Using 4096 fixed random particles on the E2 `128 x 16` grid gives:

| sigma/dx | absolute net-force norm | sum of particle-force norms | R_F |
|---:|---:|---:|---:|
| 0.0 | 4.6322e-14 | 3.3784e3 | 1.3711e-17 |
| 0.5 | 3.2024e-14 | 3.1918e3 | 1.0033e-17 |
| 3.0 | 2.6738e-15 | 2.5133e3 | 1.0639e-18 |

The absolute number varies with the arbitrary force scale; the normalized
number is the appropriate conservation diagnostic.  A separate quadratic-form
test uses the identical random density and Gaussian widths for (i) exact
spectral Poisson/gradient symbols and (ii) the periodic five-point Laplacian
with a centered-difference gradient.  Both give normalized total force below
`3.4e-17`.  Thus centered finite differences do not generically break momentum
conservation when the discrete charge-to-field map is skew-adjoint and the
coupling is matched.

### Spatially varying variational gather

The later C++ prototype evaluates the particle potential

```text
psi_p = sum_ij s_ij(x_p,h(x_p)) phi_ij
```

and gathers its full local derivative.  For x,

```text
E_px = -(1/dx) sum_ij phi_ij [
          (partial_x s_x) s_y
        + (dh_x/dx) (partial_hx s_x) s_y
        + (dh_y/dx) s_x (partial_hy s_y)] .
```

`fill_w_derivatives_exact` differentiates the directly sampled,
per-particle-normalized wrapped Gaussian, including normalization derivatives.
The test profile is a periodic `tanh(beta*sin(...))` interpolation between 0.5
and 3.0 cells with `beta=6`; `h_y=0.5` is fixed.  The implementation is opt-in
through `HEXAPIC_KDE_GRAD_H=1`; it is a later research modification, not part
of the supplied baseline commit.

The 1500-step values are:

| case | Delta P_x | Delta P_y | ||Delta P||_2 | cumulative relative force (x,y) |
|---|---:|---:|---:|---:|
| discontinuous control | -2.39010e-17 | 1.70519e-18 | 2.39617e-17 | (3.396e-4, 1.549e-2) |
| smooth, grad-h off | 1.31518e-20 | -1.49490e-18 | 1.49495e-18 | (1.043e-5, 1.305e-2) |
| smooth, grad-h on | 7.51993e-21 | -1.49495e-18 | 1.49497e-18 | (5.964e-6, 1.305e-2) |
| uniform smooth | -6.02150e-20 | -2.48923e-18 | 2.48996e-18 | (4.563e-5, 1.628e-2) |

Consequently:

* grad-h on/off is `0.571781` for `|Delta P_x|` (a 42.8% reduction) but
  `1.00001368` for the 2D norm (a 0.0014% increase);
* discontinuous/smooth-off is `1817.32` for `|Delta P_x|` but `16.0284` for
  the 2D norm;
* the local analytic derivative agrees with the centered finite difference to
  `4.94e-7` in the original one-step gate.  This validates the implemented
  local derivative, not translation invariance of a prescribed grid-anchored
  profile.

The five-phase one-step forces reproduce the sign-changing sequence
`{+4.4195e-11, +1.2229e-13, -4.2900e-11, +2.6587e-14, +4.4199e-11}`.
The 1500-step vector norms are all approximately `1.49e-18` and are dominated
by y; their x components retain a phase dependence but do not simply integrate
the one-step sign pattern.  This is evidence of grid anchoring, not evidence
that the full drift is negligible in a dimensionless physical norm.

### Solver-tolerance isolation

Job 5195488 held the input, particles, profile, grad-h term, gauge handling,
PFMG relaxer and cold start fixed.  Only `HEXAPIC_HYPRE_TOL` changed:

| configured tolerance | iterations | manual residual | F_x | |F_x| / sum|qE_x| |
|---:|---:|---:|---:|---:|
| 1e-5 | 202 | 9.59768e-6 | 4.4195005e-11 | 6.6926598e-4 |
| 1e-7 | 282 | 9.87790e-8 | 4.4195425e-11 | 6.6926605e-4 |
| 1e-9 | 363 | 9.60113e-10 | 4.4195429e-11 | 6.6926605e-4 |
| 1e-11 | 443 | 9.89252e-12 | 4.4195429e-11 | 6.6926605e-4 |

The force's fractional span is `9.60e-6` while the solver residual changes by
roughly six orders of magnitude.  It is therefore incorrect to attribute this
floor to insufficient PFMG tolerance.  The result is specific to the tested
configuration; it does not certify every HYPRE mode, profile or parallel
decomposition.

## Energy diagnostic and uncertainty

The archived Python code defines

```text
K = 0.5 sum_p |w_p| (v_x^2+v_y^2)
U_E = 0.5 sum_ij (E_x^2+E_y^2) dx dy
E = K + U_E.
```

It reports both `max|Delta E|/|E(0)|` and
`max|Delta E|/max_t(U_E)`.  The first can look artificially small because the
thermal kinetic energy dominates; the field-energy normalization is therefore
retained beside it.  One ordinary least-squares slope of `Delta E/E(0)` is fit
over all diagnostic times in each run.  The audit bootstraps 10,000 resamples
of the ten independent seed-level statistics.  It does not resample correlated
times within a run.

Campaign settings recovered from the per-run JSON are:

| benchmark | alpha | PPC | dt | duration | diagnostic interval | widths |
|---|---:|---:|---:|---:|---:|---|
| weak Landau | 0.01 | 1024 | 0.05 | 90 | 0.1 | CIC, sigma=1, sigma=6 |
| strong Landau | 0.5 | 1024 | 0.05 | 45 | 0.1 | CIC, sigma=1, sigma=6 |
| E2 longitudinal (12,0) | 0.001 | 32 | 0.05 | 60 | 0.1 | CIC, sigma=0.1844549, sigma=12 |

Each cell has seeds 0--9.  The E2 C4 energy run uses the old equal-axis scalar
width and is not the later directional corner choice.

Selected ensemble means are:

| benchmark/case | max|Delta E|/|E(0)| | max|Delta E|/max U_E | slope per time | 95% slope CI contains zero? |
|---|---:|---:|---:|:---:|
| weak/CIC | 4.9303e-6 | 2.5559e-2 | 4.0199e-8 | no |
| weak/C4 | 3.2998e-7 | 1.7820e-3 | 1.0658e-9 | yes |
| strong/CIC | 8.9050e-5 | 4.1931e-4 | 2.3281e-7 | no |
| strong/C4 | 8.9445e-5 | 4.2181e-4 | 2.1769e-7 | no |
| E2/CIC | 6.7297e-3 | 3.9380e-2 | 1.1091e-4 | no |
| E2/C4 | 4.9703e-3 | 2.8891e-2 | 8.3875e-5 | no |

Across all three cases (CIC, C4 and the deliberately over-smoothed control) and
three benchmarks, eight of nine slope CIs exclude zero.  A linear slope over a
finite trajectory is not proof of eventual unbounded growth, but it directly
contradicts the old blanket claim of “no secular growth.”

The C++ `audit.inp.energy.csv` files contain step zero only because the
diagnostic cadence equals the full 1500-step duration.  They cannot support a
HYPRE energy-evolution or spectral-versus-HYPRE energy comparison.

## Controlled time-step convergence

The diagnostic scan fixes the `64 x 16` grid, PPC 4, seed 24680, alpha 0.5,
duration 5, initialization and spatial operators, then uses
`dt={0.2,0.1,0.05,0.025,0.0125}` for CIC and Gaussian sigma=1.  It is not a
new production measurement.  Successive final-energy differences give orders:

* CIC: 1.979, 2.322, 2.152;
* C4: 2.014, 2.046, 1.965.

All charge changes are exactly zero.  At the finest step the maximum absolute
energy drift is still `1.10e-4` for CIC and `3.10e-5` for C4, showing the
fixed-space/coupling floor that a time-step refinement cannot remove.

## Unresolved issues

1. Jobs 4855533, 4858774 and 4858823 did not store their executable or source
   hashes.  The recovered patch matches the current source used by job 5195488,
   but byte identity with the historical executable remains unknown.
2. No evolving HEXAPIC energy history or HYPRE energy/timestep convergence
   campaign exists in the recovered data.  A spectral-versus-HYPRE energy claim
   is therefore unresolved.
3. The long-run momentum drift has not been normalized by a benchmark-scale
   particle momentum or impulse.  Its dimensional smallness alone does not
   establish dynamical irrelevance.
4. The grad-h test differentiates a prescribed one-dimensional smooth `h_x(x)`
   with fixed `h_y`.  It does not validate differentiation through a live,
   translation-equivariant selector/controller, and the vector norm is
   dominated by the uncorrected y component.
5. The PFMG tolerance scan is one step, one smooth profile, one rank and one
   relaxer.  It isolates the reported floor for that case only.

## Manuscript changes justified by this audit

`paper1.tex` now:

* uses normalized spectral total force and lists the complete particle-force
  values instead of the scale-dependent `5e-15` absolute number;
* states that a periodic centered-finite-difference operator is also
  skew-adjoint when exactly and consistently coupled;
* labels 42.8% and 1817 as x-component results and gives the corresponding 2D
  norm changes;
* adds the six-decade PFMG tolerance result and removes attribution of the
  residual to solver tolerance;
* removes “adaptivity does not degrade momentum” and “same discretisation
  floor” claims;
* replaces “bounded drift, no secular growth” with the explicit energy metric,
  seed-level slope protocol and measured values;
* restricts the energy result to the spectral Python campaign and leaves a
  HYPRE energy comparison unresolved;
* replaces one width-independent charge number with the observed smooth and
  discontinuous-profile maxima.
