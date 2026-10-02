# Data contract (schema `nta-1`)

Every run of every backend (`pic1d1v`, `vlasov1d1v`, later `hexapic`, and `synthetic` for the T1 suite) writes **one `.npz` bundle**. `attractor_analysis.schema.read_bundle` validates it, and nothing is analysed until it passes. The validator lives in `attractor_analysis/schema.py`; this file is the human-readable version.

## Metadata (`metadata_json`)

| Key | Meaning |
|---|---|
| `schema_version` | `nta-1` |
| `backend` | `pic1d1v`, `vlasov1d1v`, `hexapic` or `synthetic`. Reference-code output is never labelled as HEXAPIC. |
| `git_revision` | commit of the code that produced the run |
| `config` | full run configuration, including the deterministic seed spectrum (amplitudes and phases) |
| `normalization` | ω_pe of the **total** electron density = 1. Velocities are in units of the beam speed v_b, lengths in v_b/ω_pe. Give the conversion to λ_D. |
| `seed`, `nested_level` | the independent seed, and the nested regulator level (0 = finest) |
| `dt`, `dx` | time step, grid spacing |
| `staggering` | the convention for when x, v and E are synchronised at a diagnostic. For the KDK pusher used in T2 they are all at integer steps. |
| `wall_time_s` | wall time |
| PIC only | `particles_per_cell`, `n_particles`, `macro_weight` |
| Vlasov only | `dv`, `v_max` |

## Arrays (time is always the leading axis)

| Array | Shape | Definition |
|---|---|---|
| `times` | (nt,) | exact diagnostic times, strictly increasing |
| `number`, `total_charge`, `momentum`, `kinetic_energy`, `field_energy`, `total_energy` | (nt,) | budgets. The field energy is the one consistent with the implemented discrete operator; periodic endpoints are not double-counted. |
| `eps_edges` | (nb+1,) | frozen common edges, starting at 0 (linear below `eps_lo`, logarithmic above) |
| `N_eps` | (nt, nb) | N(ε) = dN_phys/dε, with N_b = Σ w_p / Δε_b |
| `eps_bin_weight`, `eps_bin_energy` | (nt, nb) | exact Σw and Σwε per bin, so moments are exact |
| `eps_bin_neff` | (nt, nb) | Kish effective count (Σw)²/Σw² |
| `eps_overflow` | (nt, 2) | weight and energy at or above the last edge (never clipped into a bin) |
| `eta_edges` | (nη+1,) | frozen positive η edges |
| `rho_eta`, `rho_bin_volume`, `rho_bin_mass` | (nt, nη) | ρ(η) = phase volume per unit η; volume and Σ f·cell per η bin |
| `rho_reconstruction` | (3,) | [Δx_b, Δv_b, η_floor], where η_floor = one macroparticle per reconstruction cell (0 for a grid f) |
| `k`, `Ek2` | (nk,), (nt, nk) | wavenumbers and the field spectrum \|E_k\|² |
| optional `phase_space_*`, `mode_*` | | decimated x–v snapshots and selected mode amplitudes |

## Definitions the analysis relies on

- The active energy is ε = m v_x²/2 in the initial centre-of-momentum frame. The physical mass enters ε; the macro weight enters only the sum.
- **dN/d ln ε = ε N(ε) is a different quantity from N(ε).** Plots say which one they show and never overlay the two.
- **Log-tail analysis** uses only bins with a positive lower edge. **Moment checks** include the low-energy bins and the overflow.
- **No per-snapshot renormalisation, and no clipping.**
- **ρ(η) from particles or a binned f is coarse-grained.** Fine-grained Vlasov Casimirs are conserved; coarse-grained estimates need not be. A feature within ~10× η_floor is flagged `resolution_limited`.
- **Resampling.** The independent seed is the resampling unit. Snapshots of one run are correlated and are never treated as independent samples. Quiet loads are not Poisson.
