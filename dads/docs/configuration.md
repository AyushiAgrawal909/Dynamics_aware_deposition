# Configuration reference

Configurations are YAML (`.yaml`, `.yml`) or JSON. Relative file paths are
resolved from the configuration file's directory. Unknown keys are rejected,
and the error suggests the closest valid key. `dads example <name>` writes
commented templates.

## `grid` (required)

| key | default | meaning |
|---|---|---|
| `N` | — | cells per axis, 1 to 3 axes |
| `L` or `dx` | — | domain lengths, or cell sizes |
| `shape` | `cic` | deposit and gather shape: `ngp`, `cic`, `tsc`, `pqs` |
| `poisson` | `spectral` | `spectral` (K² = \|k\|²) or `fd2` (second-order Laplacian) |
| `gradient` | `spectral` | `spectral` (κ = k) or `centered` (sin(kΔ)/Δ) |
| `scheme` | `momentum` | `momentum` (interpolated field) or `energy` (gradient of interpolated potential) |
| `fft` | `rfft` | Nyquist handling of the field metric: `rfft` (real-to-complex solver) or `complex` |

## `time`

| key | default | meaning |
|---|---|---|
| `integrator` | `continuous` | `continuous` or `leapfrog` (velocity Verlet) |
| `dt` | — | time step; required for `leapfrog`; also used for \|ω\|Δt warnings |
| `temporal_aliases` | 8 | explicit temporal aliases for warm species (leapfrog) |

## `species` (required, list)

| key | default | meaning |
|---|---|---|
| `q`, `m`, `n` | — | charge, mass, number density |
| `drift` | 0 | drift velocity; a scalar means along the first axis |
| `vth` or `T` | 0 | thermal speed sqrt(T/m), or temperature (k_B = 1); scalar or one value per axis; 0 means cold |
| `ppc` | — | markers per cell of this species (model objective) |
| `mobile` | true | `false` excludes the species from the response |
| `name` | auto | label |

`eps0` (top level, default 1) sets the permittivity. A neutralising immobile
background is always implied.

## `protected_modes` (list) and `auto_protect`

| key | default | meaning |
|---|---|---|
| `mode` | — | integer mode indices m, with k_i = 2πm_i/L_i |
| `branch` | `most_unstable` | `most_unstable`, `least_damped` or `nearest` |
| `omega_guess` | — | [Re, Im] starting value; required for `nearest` |
| `sign` | 0 | +1 or −1 restricts the choice to forward or backward waves |
| `tolerance` | global | mode-specific rate tolerance |
| `frequency_tolerance` | global | mode-specific frequency tolerance |

`auto_protect` scans every grid mode and protects the growing ones. Mirror
images (same \|m_i\| on every axis, same root) are merged. Its keys:

| key | default | meaning |
|---|---|---|
| `min_growth` | 0 | absolute growth-rate threshold |
| `min_growth_fraction` | — | keep modes growing at least this fraction of the fastest |
| `max_modes` | 64 | at most this many distinct modes |
| `aliases` | 0 | aliases used in the scan (principal by default) |
| `branch` | `most_unstable` | branch rule of the added modes |

## Tolerances

| key | default | meaning |
|---|---|---|
| `tolerance` | 0.05 | operating tolerance ε: \|R_γ − 1\| ≤ ε for every protected mode |
| `acceptance` | 0.10 | wider tolerance; the result is reported for comparison |
| `frequency_tolerance` | none | also require \|ω_r/ω_r0 − 1\| ≤ value (oscillating modes) |

## `dispersion`

| key | default | meaning |
|---|---|---|
| `aliases` | 4 | spatial aliases per axis in the response sum |
| `a_min` | 0.02 | smallest coupling the branches are followed to |
| `verify_roots` | true | argument-principle check of `most_unstable` roots |

## `filter`

| key | default | meaning |
|---|---|---|
| `coupling` | `matched` | `matched` (filter density and field; a = W²) or `density_only` (a = W) |

## `candidates`

| key | default | meaning |
|---|---|---|
| `shapes` | [grid shape] | extra deposit shapes to consider, e.g. `[tsc]` |
| `gaussian.enabled` | true | include the Gaussian family |
| `gaussian.h_max` | 3.0 | upper bound on h_i = σ_i/Δ_i (scalar or per axis) |
| `gaussian.axes` | all | smooth only these axes |
| `gaussian.isotropic` | false | one physical σ for all axes |
| `gaussian.grid` | 400 / 61 / 21 | lattice points per axis (1D / 2D / 3D) before refinement |
| `binomial.enabled` | true | include compensated binomials |
| `binomial.p` | [0..4] | compensation degrees (0 = raw binomial) |
| `binomial.n` | [1,2,3,4,6,8] | numbers of passes |
| `binomial.per_axis` | false | different n on each axis (including 0) |
| `extra` | — | explicit candidates, e.g. `{family: gaussian, h: [0, 0.24]}` or `{family: binomial, p: 2, n: 2}` |

## `objective`

| key | default | meaning |
|---|---|---|
| `mode` | `model` | `model` (reference-free) or `snapshots` (reference-assisted) |
| `metric` | `density` | `density` or `field` |
| `ppc` | — | total markers per cell (`model`) |
| `signal` | [] | `model`: list of `{mode: [...], amplitude: A}`; A is relative to Σ\|q_s\|n_s |
| `snapshots` | — | `.npz` with paired meshes of shape (realisations, times, *N) |
| `rho_key`, `ref_key` | `rho`, `rho_ref` | array names in the snapshot file |
| `statistics` | — | alternative: compact file from `dads stats` |
| `select`, `evaluate` | first and second half | realisation indices used for choosing and for the held-out report |
| `extra_shapes` | — | `{tsc: rho_tsc}`: production meshes deposited with other shapes |

## Command line

| command | does |
|---|---|
| `dads select CONFIG [-o DIR] [--tolerance X] [--acceptance X] [--no-plot] [-q]` | runs the selection |
| `dads roots CONFIG [--mode ...] [--a W2]` | lists the dispersion roots and the unstable count |
| `dads scan CONFIG` | tabulates R_γ(a) and the thresholds of the protected modes |
| `dads stats SNAPSHOTS.npz --L ... -o STATS.npz` | reduces snapshots to compact spectral sums |
| `dads stencil --p P --n N` | prints the real-space taps of a compensated binomial |
| `dads example [NAME] [-o FILE]` | lists or writes example configurations |
| `dads validate` | runs the reproduction checks |

Exit status of `dads select`: 0 if an admissible kernel was found, 2 if not.
