# Theory and numerical method

Notation follows the paper. A perturbation behaves as exp(i**k**·**x** − iωt) with
ω = ω_r + iγ: γ > 0 means growth and γ < 0 means damping. ε₀ = 1 unless it is set.

## 1. Implemented coupling

Particles of shape S are deposited on a periodic mesh. The density is filtered by
W, the field is solved, filtered again by W, and gathered with S (momentum
scheme). In Fourier space a mesh mode **k** receives charge from all its spatial
aliases **q** = **k** + 2π**p**/Δ, with **p** an integer vector. The force on a
particle at alias **q** is S(**q**) times the mesh field. The linearised Vlasov
equation then gives

  ε(**k**, ω; a) = 1 + a G(**k**, ω),
  G(**k**, ω) = Σ_**q** g(**q**) Σ_s χ_s(**q**, ω),

with the geometric factor

| scheme | g(**q**) |
|---|---|
| momentum (interpolated field) | S(**q**)² (**q**·κ(**k**)) / K²(**k**) |
| energy (gradient of interpolated potential) | S(**q**)² \|**q**\|² / K²(**k**) |

and the round-trip transfer

| `filter.coupling` | a |
|---|---|
| matched (density and field) | W(**k**)² |
| density only | W(**k**) |

The shape of order p (NGP p = 0, CIC 1, TSC 2, PQS 3) has
S(**q**) = Π_i sinc(q_iΔ_i/2)^(p+1). The solve enters through two symbols:

| option | Laplacian K² | gradient κ_i |
|---|---|---|
| `spectral` | \|**k**\|² | k_i |
| `fd2` / `centered` | Σ_i (2 sin(k_iΔ_i/2)/Δ_i)² | sin(k_iΔ_i)/Δ_i |

The filter acts on mesh quantities, so W(**k**) multiplies every alias term
equally. **For one mode, the whole filter dependence of the dispersion
relation is the single number a.** For the principal alias alone with spectral
operators, the relation becomes the paper's Eq. 8, D = 1 + S²W²χ.

Aliases are summed up to |p_i| ≤ `dispersion.aliases`. Terms with
|g| < 1e-16 |g_principal| are dropped. With CIC, four aliases per axis agree with
twelve to about 1e-7 in the paper's growth rate.

## 2. Susceptibility of a drifting (bi-)Maxwellian

A species with velocity distribution N(**u**, diag(v_th,i²)) has, for an
electrostatic wave along **q**, the 1D Maxwellian projection. Its mean is
**q̂**·**u** and its variance is v_q² = Σ_i q̂_i² v_th,i². Hence

  χ_s(**q**, ω) = ω_ps² / (\|**q**\|² v_q²) · [1 + ζ Z(ζ)],
  ζ = (ω − **q**·**u**_s) / (√2 \|**q**\| v_q),
  ω_ps² = n_s q_s² / (ε₀ m_s).

Z(ζ) = i√π w(ζ) uses the Faddeeva function. Because w is entire, this is the
Landau-continued Z in the whole plane, so damped roots need no special
treatment. For \|ζ\| > 30 the combination 1 + ζZ is taken from its asymptotic
series to avoid cancellation; below the real axis the series includes the
Landau term 2i√π ζ e^(−ζ²). A cold species (v_q = 0) contributes
−ω_ps²/(ω − **q**·**u**_s)². The derivative ∂ε/∂ω is analytic:

  d(1 + ζZ)/dζ = Z − 2ζ(1 + ζZ).

## 3. Time integration

| `time.integrator` | response |
|---|---|
| `continuous` | the expression above |
| `leapfrog` | sum over temporal aliases (below) |

In the `leapfrog` model, a leapfrog or velocity-Verlet push, with charge and
field at integer steps, has the orbit recurrence
δx^(n+1) − 2δx^n + δx^(n−1) = Δt² (q/m) E. Each 1/Ω² of the continuous
response (Ω = ω − **q**·**v**) is then replaced by

  (Δt/2)² / sin²(ΩΔt/2) = Σ_l 1/(Ω − 2πl/Δt)².

The leapfrog susceptibility is therefore the continuous one summed over the
temporal aliases ω − 2πl/Δt:

- **Cold species.** The sum is evaluated in closed form.
- **Warm species.** The sum is explicit for \|l\| ≤ `time.temporal_aliases`.
  The remaining tail uses the large-\|Ω\| expansion of the warm response,
  χ ≈ −ω_p²/Ω² − 3ω_p² q²v_q²/Ω⁴. Its sums are written in closed form:
  - Σ_l 1/Ω_l² = h² csc²(hΩ);
  - Σ_l 1/Ω_l⁴ = h⁴ csc²(hΩ)(csc²(hΩ) − 2/3);
  - here h = Δt/2, and the l = 0 term is removed analytically.

  With 6 explicit aliases, the tail is accurate to better than 1e-7 relative
  (see the tests).

Roots are reported in the principal strip \|ω_r\| ≤ π/Δt. Not modelled are the
explicit-midpoint δf update of the paper, implicit and energy-conserving time
schemes, and sub-cycling.

## 4. Roots, branches and continuation

**Starting points.**

- The roots of cold-fluid and warm-fluid (adiabatic index 1 and 3)
  polynomials of the principal mode;
- a grid of complex starting points spanning the drifts, the thermal spreads
  and ω_p;
- an optional user guess.

**Newton iterations** are damped. Distinct converged roots are kept.

**Branch rules.**

| rule | picks |
|---|---|
| `most_unstable` | largest γ |
| `least_damped` | largest γ (for damped spectra) |
| `nearest` | the root closest to `omega_guess` |

`sign` restricts the choice to the forward (+1) or backward (−1) wave. For a
growing branch, the argument principle counts the roots above
γ_selected (1 + 2%) on a contour that is refined adaptively. A non-zero count
raises a warning that a faster root exists.

**Continuation in a.**

- *Predictor:* the tangent dω/da = −G/(a G′), which is Eq. 9 of the paper,
  since ∂ₐε = G.
- *Corrector:* Newton.
- *Step control:* a step is accepted only if the corrector stays within 20% of
  the predicted displacement; otherwise the step is halved.

For a growing branch, tracking stops where γ reaches 0. The **marginal
coupling** a_c is then the real solution (ω_r, a_c) of ε(ω_r + i0, a_c) = 0,
found with a least-squares Newton iteration. For the paper's symmetric beams,
ω_r = 0 and this is the Penrose threshold of the principal model; the tests
compare it with the Penrose integral directly. Between the last tracked point
and a_c, roots are recovered from the square-root law γ ∝ (a − a_c)^(1/2)
(Eq. 10).

**Root status of a candidate** at coupling a, for each protected mode:

| status | meaning |
|---|---|
| `growing` / `damped` | a root of the tracked branch is found; R_γ = γ(a)/γ₀ is defined |
| `marginal` | the branch reached γ = 0 at a_c > a; no growing root |
| `unresolved` | the continuation failed |

Only the first can be admissible. A failed solve is never read as stability.

## 5. Admissible set and selection

A candidate θ is admissible at tolerance ε if, for every protected mode,
its status is resolved and \|R_γ(**k**; θ) − 1\| ≤ ε. Optionally the
frequency must also satisfy \|ω_r/ω_r0 − 1\| ≤ ε_ω. Because a mode depends on
the filter only through a, admissibility is the interval a_lo(**k**) ≤ a. Here
a_lo is the first exit from the tolerance band along the branch from a = 1,
located by bisection to 1e-12.

- **Gaussian family.** W² = exp(−Σ_i (k_iΔ_i)² h_i²), with h_i = σ_i/Δ_i. The
  constraints Σ_i (k_iΔ_i)² h_i² ≤ −ln a_lo(**k**) are linear in u_i = h_i².
  `dabs` evaluates the objective on a lattice covering the admissible
  polytope (1.3 times its extent). It then refines the best feasible points
  with SLSQP under the linear constraints and checks the optimum by a direct
  root solve. For a noise-dominated objective the optimum lies on the
  boundary, so R_γ = 1 − ε. Two options are available:
  - `isotropic: true` optimises one physical σ;
  - `axes` restricts smoothing to chosen axes.
- **Compensated binomials.** W_{p,n}(k) = (1−s)^n Σ_{j≤p} C(n+j−1, j) s^j,
  with s = sin²(kΔ/2), applied separably.
  - p = 0 is the raw n-pass [1/4, 1/2, 1/4] filter.
  - The transfer error is O((kΔ)^(2p+2)).
  - The real-space stencil has 2(n+p)+1 taps; it is printed in the summary
    and by `dabs stencil`.
- **Alternative shapes** (`candidates.shapes`). For each shape, the branch is
  continued from the baseline root by a homotopy between the two coupling
  models. R_γ is always measured against the baseline unfiltered kernel.

The **selected kernel** minimises the objective over all admissible
candidates. The summary also reports:

- the best admissible candidate of each family;
- the choice at the acceptance tolerance;
- the unconstrained optimum, which shows what error minimisation alone would
  pick.

## 6. Objectives

**Snapshot objective (reference-assisted).** Paired realisations r are given
at the production marker count (ρ) and at a higher count (ρ_ref). Both are
deposited with the baseline shape and saved at common times. For any mesh
filter,

  MSE(W) = Σ_k w(k) [W² A − 2W B + C] / (N_r N_t N_g²),

with
- A = Σ_{r,t} \|ρ̂\|²;
- B = Σ Re(ρ̂ ρ̂*_ref);
- C = Σ \|ρ̂_ref\|²;
- weight w = 1 for density, and w = \|κ\|²/K⁴ for the field after the solve
  and before return filtering.

These are the paper's Eqs. 13–14. A, B and C are stored per realisation, so
selection and held-out evaluation use disjoint realisations. Reduction
factors N = MSE(baseline)/MSE(θ) get paired bootstrap intervals over
realisations. `dabs stats` reduces snapshots to these sums, which take
3/(2N_t) of the storage of the snapshots themselves.

For a spectral solve, a derivative at an axis's Nyquist wavenumber is
projected out when the field is made real. With real-to-complex transforms
(`grid.fft: rfft`, the default, as in `numpy.fft.rfftn`), this happens only
where the last-axis index is 0 or Nyquist. With complex transforms
(`grid.fft: complex`) it happens everywhere. The choice changes field errors
by about 0.1% in the paper's case.

**Model objective (reference-free).** For uncorrelated markers with equal
weights within each species, the per-mode noise power of the deposited
charge density is

  P_n(**k**) = Σ_s (q_s n_s)² / PPC_s · Σ_**p** S(**k** + 2π**p**/Δ)²,

with the alias sum in closed form (Hockney and Eastwood), and zero at
**k** = 0 because the total charge is fixed. Each signal mode of relative
amplitude A_m (relative to Σ_s \|q_s\| n_s) adds a bias (G − 1)² S_base² A_m²/2,
where G = W S_cand/S_base. The expected error is

  E = mean_k w(k) [ (G − 1)² P_s + W² P_n ].

PPC_s defaults to `objective.ppc` × n_s/Σn. A Monte Carlo test checks the
noise term to 3%. This objective ignores marker correlations, both from quiet
starts and from the plasma's own shielding. On the paper's data it ranks
candidates almost exactly as the snapshot objective does, but it
underestimates field-error reductions.

## 7. What is not covered

- The finite-marker corrections to the rates (the paper's finite-marker
  analysis).
- Nonlinear evolution.
- Magnetised or electromagnetic modes.
- Non-periodic boundaries.
- Non-uniform equilibria.
- Particle-shape smoothing applied to the particles themselves (as opposed
  to the mesh).

A selected kernel should be checked in a linear test run, as in the paper.
