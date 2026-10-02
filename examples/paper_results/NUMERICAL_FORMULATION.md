# Particle--field coupling audit

Audit date: 2026-09-09 (Europe/Luxembourg)

This document distinguishes two implementations that must not be conflated:

1. the Python 2D2V model used by the recovered Paper 1 rate/noise campaigns;
2. the supplied C++ HEXAPIC repository and its uncommitted research changes.

The historical result files do not contain source hashes.  Therefore the paths
below establish what the recovered code currently does, not byte-for-byte proof
of what any expired Slurm job executed.

## 1. Python Paper 1 model

### Update and diagnostics

The full-f step is velocity Verlet: deposit, solve, gather, first half kick,
drift, then a second deposit, solve, gather, and half kick
([`pic_loop.py`, lines 50--72](../../pic_kde_2d/pic_kde_2d/core/pic_loop.py)).
The diagnostic path separately deposits and solves, and records the Fourier
coefficient of that grid field
([`pic_loop.py`, lines 94--107](../../pic_kde_2d/pic_kde_2d/core/pic_loop.py)).
Consequently, for a spectral-filter method the reported grid-field mode has one
factor of `W` from deposition; the force that changes particle trajectories has
two factors of `W`.

The linearized delta-f path uses frozen equilibrium characteristics and a
midpoint weight update.  It calls the same deposit, Poisson, and gather
functions at the initial and midpoint states
([`pic_loop_delta_f.py`, lines 101--140](../../pic_kde_2d/pic_kde_2d/core/pic_loop_delta_f.py)).

### Deposition

Baseline CIC deposits a particle to the four periodic nodal neighbours with
bilinear weights and divides by `dx*dy`
([`cic.py`, lines 19--49](../../pic_kde_2d/pic_kde_2d/methods/cic.py)).
For a spectral-filter method, CIC is performed first and `W(kx,ky)` is then
applied once to the nodal density
([`spectral_W_wrappers.py`, lines 18--32](../../pic_kde_2d/pic_kde_2d/methods/spectral_W_wrappers.py)).

### Poisson solution and grid field

The grid is periodic in both axes.  The solver uses NumPy `rfft2` and the
continuum Fourier Green function

```text
phi_hat(k) = rho_hat(k) / (kx^2 + ky^2),  k != 0
phi_hat(0) = 0
E_hat      = -i k phi_hat.
```

This is an exact spectral derivative on the represented grid modes, not a
finite-difference derivative
([`grid2d.py`, lines 40--63](../../pic_kde_2d/pic_kde_2d/core/grid2d.py)).

### Gather and precise smoothing locations

Baseline CIC interpolates the already calculated nodal `Ex` and `Ey` values;
it does **not** differentiate an interpolated potential
([`cic.py`, lines 52--75](../../pic_kde_2d/pic_kde_2d/methods/cic.py)).
The spectral wrapper filters each grid-field component once immediately before
that interpolation
([`spectral_W_wrappers.py`, lines 34--46](../../pic_kde_2d/pic_kde_2d/methods/spectral_W_wrappers.py)).

For a single Fourier mode, relative to the Python CIC baseline:

| Location | Additional filter factor |
|---|---:|
| density after deposition | `W` |
| electric field returned by Poisson | `W` |
| electric field immediately before gather | `W^2` |
| force gathered to particles | `W^2` |
| mode recorded by `run_simulation` diagnostics | `W` |

The Gaussian definition is
`W=exp[-(kx^2 sigma_x^2 + ky^2 sigma_y^2)/2]`
([`spectral_W.py`, lines 24--28](../../pic_kde_2d/pic_kde_2d/methods/spectral_W.py)).
Thus `W^2`, not `W`, is the added force-chain attenuation relative to CIC.
The CIC deposition and CIC gathering shapes remain present on either side; they
cancel only when a filtered result is reported as a ratio to the same CIC
baseline at the same particles and mode.

## 2. C++ HEXAPIC

### Provenance split

The supplied repository commit is
`ed6dd96d16d0b70a53bf9486fd78a270ab3486d6`.  The current worktree contains
substantial uncommitted research changes.  In particular, the supplied commit
already uses HYPRE PFMG, while the current worktree changes its tolerance from
`1e-5` to `1e-9` and maximum iterations from 200 to 500.  Which version any
historical manuscript job used remains unknown.

The C++ time loop deposits, updates the HYPRE right-hand side, solves, starts
the potential halo exchange, moves interior particles, finishes the halo, and
moves edge particles
([`main.cpp`, lines 163--243](../../hexapic-main/source/main.cpp)).

### Charge deposition and scaling

Baseline C++ CIC deposits to four nodes with the same bilinear weights
([`particles.cpp`, lines 936--967](../../hexapic-main/source/particles.cpp)).
The deposited species charge number is multiplied by

```text
Sconst = -(q_e * ParticleWeight)/(epsilon_0 dx dy) * dx^2.
```

The extra `dx^2` matches the dimensionless HYPRE matrix coefficients
([`particles.cpp`, lines 880--889](../../hexapic-main/source/particles.cpp)).
This yields the finite-difference equation `laplacian(phi)=-rho/epsilon_0`.

There are three materially different research paths:

| C++ path | Deposit | Smoothing location | Gather |
|---|---|---|---|
| CIC | four-node CIC | none | derivative of bilinear nodal potential |
| `cic_plus_filter` legacy | CIC | one separable `[1/4,1/2,1/4]` density pass in both axes | derivative of bilinear nodal potential; filter is **not** repeated at gather |
| action `cic_plus_filter_weak` | CIC | one normalized `[1,6,1]` density pass in both axes | baseline potential derivative; filter is **not** repeated |
| action `cic_plus_filter_medium` | CIC | one normalized `[1,4,6,4,1]` density pass in both axes | baseline potential derivative; filter is **not** repeated |
| direct Gaussian action | normalized separable 3x3/5x5/7x7 stencil | in particle shape itself | derivative of potential interpolated with the same stencil |
| legacy Gaussian lookup mode | normalized compact Gaussian lookup | in particle shape itself | compact interpolation of centred-difference nodal `E` |
| periodic KDE/von Mises | normalized separable periodic kernel | in particle shape itself | derivative of kernel-interpolated potential, holding bandwidth fixed |

The legacy binomial pass is implemented only after density deposition
([`particles.cpp`, lines 969--1015](../../hexapic-main/source/particles.cpp)).
The action filters are likewise applied to their CIC-deposited density before
copying into the HYPRE source
([`particles.cpp`, lines 1380--1481](../../hexapic-main/source/particles.cpp)).
Their normalized kernels are defined in
[`deposition_action_library.cpp`, lines 332--406](../../hexapic-main/source/deposition_action_library.cpp).
No C++ field-side compensator corresponding to the Python Paper 1 `W` wrapper
was found.

For periodic modes, the additional deposition transfers of the three C++
post-filters are therefore

```text
legacy cic_plus_filter:        (1-s_x)(1-s_y)
cic_plus_filter_weak:          (1-s_x/2)(1-s_y/2)
cic_plus_filter_medium:        (1-s_x)^2(1-s_y)^2
s_a = sin^2(k_a Delta_a/2).
```

These factors occur once in the C++ force chain, unlike the twice-applied
Python spectral wrapper.

### What field particles receive

There is no single answer across the C++ modes:

* Baseline CIC evaluates the derivative of the bilinear interpolant of the four
  nodal potentials
  ([`particles.cpp`, lines 1712--1757](../../hexapic-main/source/particles.cpp)).
* Direct action-library Gaussian shapes differentiate their own interpolated
  potential using analytic stencil-weight derivatives
  ([`particles.cpp`, lines 1236--1288](../../hexapic-main/source/particles.cpp)).
* The KDE path also differentiates the kernel interpolation of potential
  ([`kde_deposition.cpp`, lines 529--573](../../hexapic-main/source/kde_deposition.cpp)).
* The legacy compact Gaussian lookup path is the exception: it first forms a
  centred-difference nodal electric field (one-sided at nonperiodic boundaries)
  and interpolates that field to particles
  ([`particles.cpp`, lines 1032--1110 and 1206--1233](../../hexapic-main/source/particles.cpp)).

The mover selects among those paths at runtime
([`particles.cpp`, lines 1760--1822](../../hexapic-main/source/particles.cpp)).
Therefore a manuscript statement that HEXAPIC always interpolates nodal fields,
or always differentiates interpolated potential, is false.

Spatially varying KDE bandwidths are read by cell, but the gather derivative
holds `h` fixed; there is no visible `partial h/partial x_p` (grad-h) term in
the force expression.  Conservation claims for adaptive bandwidths therefore
remain unresolved pending the dedicated derivation and test requested in the
conservation task.

### HYPRE algorithm, tolerances, and boundaries

The implementation uses the HYPRE structured-grid interface with a 2D
five-point finite-difference stencil
([`field_solver_hypre.cpp`, lines 279--375](../../hexapic-main/source/field_solver_hypre.cpp)).
The coefficients are centre `-2[1+(dx/dy)^2]`, x neighbours `1`, and y
neighbours `(dx/dy)^2` away from boundaries.

The solver is `HYPRE_StructPFMG` with relative-change stopping disabled, one
pre-relaxation and one post-relaxation sweep, relaxation type 2
(red-black Gauss--Seidel), and logging disabled
([`field_solver_hypre.cpp`, lines 377--405](../../hexapic-main/source/field_solver_hypre.cpp)).

| Source state | Relative tolerance | Max iterations |
|---|---:|---:|
| supplied commit `ed6dd96` | `1e-5` | 200 |
| current uncommitted worktree | `1e-9` | 500 |
| historical Paper 1 executions | **unknown** | **unknown** |

The current and supplied code set the periodic length to `Nxg-1`/`Nyg-1`
for axes whose parsed boundary is `Periodic`.  `Equipotential` produces a
Dirichlet row with the configured potential on the right-hand side.
`Dielectric` omits the outward stencil coefficient, implementing a zero-normal-
difference (discrete Neumann) boundary
([`field_solver_hypre.cpp`, lines 74--140 and 302--369](../../hexapic-main/source/field_solver_hypre.cpp)).
For a fully periodic solve, the global mean is removed from both the right-hand
side and recovered potential.  This mean removal is already present in the
supplied commit, not only in the later worktree.

No residual norm or achieved iteration count is stored by this implementation,
so solver-tolerance error cannot be inferred from a run's output alone.

### PETSc status

PETSc is **not implemented or selectable** in the supplied/current source.
There are no PETSc headers, calls, link flags, or objects.  The comment in
`main.cpp` saying “MPI/PETSc” is stale.  The project changelog records that
PETSc was replaced by HYPRE PFMG in version 0.7.0
([`CHANGELOG.md`, lines 22--27](../../hexapic-main/CHANGELOG.md)); the current
makefile links HYPRE and openPMD only
([`makefile`, lines 3--20](../../hexapic-main/source/makefile)).  PETSc is thus a
historical predecessor, not an optional solver in the audited implementation.

## 3. Small diagnostic tests run before transfer simulations

Commands:

```text
python3 -m pytest -q tests/test_phase0_setup.py \
  tests/test_per_particle_momentum.py tests/test_c4_closed_form.py

python3 tests/run_plain_tests.py tests/test_phase0_setup.py \
  tests/test_per_particle_momentum.py
```

Results:

* The direct pytest command did not run because the active Python 3.14
  environment has no `pytest` module.  No dependency was installed into the
  project environment.
* The repository's dependency-free runner passed 9/9 import, FFT-convention,
  unit-documentation, and matched/mismatched momentum tests.
* `test_c4_closed_form.py` could not be loaded by the dependency-free runner
  because that file imports `pytest`; this is an environment limitation, not a
  passing test.

## 4. Consequences for manuscript language

The manuscript must identify which implementation a claim concerns.  In
particular:

1. “matched smoothing applies `W` at deposition and gather” is true for the
   Python spectral wrapper, not for the C++ post-deposition filter actions;
2. Python saved field-mode amplitudes contain one `W`, while particle dynamics
   see `W^2` relative to CIC;
3. C++ particle field evaluation depends on the selected deposition action;
4. HYPRE PFMG is the implemented C++ solver; PETSc is historical only; and
5. no conservation attribution to HYPRE is justified without residual and
   tolerance-controlled comparisons.

No manuscript text was changed during this audit stage.  These findings will
be applied only after the transfer-function and conservation tests determine
which claims remain supportable.

