# Paper 1 density-error and noise-metric audit

Audit date: 2026-09-10 (Europe/Luxembourg)

Status: **PASS for metric reproduction and estimator checks.**  The audit
reproduces every archived central value and bootstrap interval read by the
Gaussian, raw-binomial, compensated-binomial, and E2 directional reducers.
It also exposes a manuscript protocol mixture: Table 4 and the Gaussian line
in Fig. 7a use different Landau velocity-loading layouts.

## Direct answer: weak-Landau Gaussian C4 at `sigma = 1 dx`

| Source | Reported reduction | Metric and spectral scope | CIC/reference definition | Averaging | PPC |
|---|---:|---|---|---|---:|
| Table 4 selector matrix, job `4867515` | **2.559393** `[2.541134, 2.576672]` | Ratio of ensemble-mean, time-averaged **grid-density squared-error power**; real-space mean over all `64x16` nodes, hence all resolved Fourier modes by Parseval. No high-k mask. The protected `(1,0)` mode is included. `k=0` is formally included but is zero to round-off because both deposits have the same charge and `W(0)=1`. | Numerator: unfiltered low-PPC CIC snapshot density minus paired high-PPC unfiltered CIC density. Denominator: once-Gaussian-filtered low-PPC density minus that same reference. Both trajectories use the live matched Gaussian `sigma=0.25`-cell coupling. | No fit. Mean over 51 samples on `t omega_pe in [5,30]`, cadence `0.5` (`dt=0.05`, stride 10), then mean over ten seeds. | `1024`; reference `16384` |
| Fig. 7a Gaussian overlay, job `4868286` | **5.643223** `[5.613435, 5.674546]` | The same grid-density error-power metric and the same all-mode scope. | The same CIC numerator, high-PPC reference, and live coupling. | The same 51-sample `[5,30]` average and ten seeds. | `1024`; reference `16384` |

**One-line explanation:** `2.559` versus `5.643` is explained by the
velocity-quadrature layout---randomized per cell in job `4867515`, globally
shuffled in job `4868286`; metric, mode scope, CIC/reference definitions,
window, cadence, and PPC are otherwise the same.

The per-cell layout is the same `32x32` velocity tensor later rejected by the
confirmation preflight.  At PPC 1024 it supplies only 32 distinct
longitudinal nodes, reaches `|vx|=2.15387`, and has no node at or above the
weak-Landau resonant phase velocity `3.21264`.  Therefore the job-4867515 C4
and Silverman values are retained as density-error powers for that constructed
ensemble, not as resonant-dynamics evidence or a same-campaign trade-off with
the global rate histories.  No recovered historical Landau rate history is
identified as per-cell; exact revisions for the oldest rate jobs remain
unknown.

The premise that the binomial reducer itself reports `5.64` needs one precise
correction.  The Fig. 7a builder loads the Gaussian overlay from job `4868286`
through `LANDAU_NOISE`.  The raw-binomial reducer from job `5048431` gives
**5.871509** `[5.831015, 5.910255]` for `n=2`, whose low-wavenumber equivalent
width is `sigma_eq=1 dx`.  It uses the same metric/reference/window/PPC but 251
samples (stride 2).  The two filters are matched only to leading order near
`k=0`; they are not identical over the full resolved spectrum used by the
error metric.

## Exact implemented metric

For seed `s`, the stored `mise_time_avg` or `mise` is

```text
E_rho,s(H) = mean_time mean_grid [(rho_H - rho_ref)^2].
```

The first mean is over the recorded window and the second is over all grid
nodes.  On a uniform periodic grid, `mean_grid` equals the discrete spatial
integral divided by the domain area.  The per-seed quantity is therefore a
time-averaged realised ISE divided by `|Omega|`; the mean across independent
seeds is the MISE estimate.  The reported reduction is

```text
N_rho(H) = mean_seed E_rho,s(0) / mean_seed E_rho,s(H).
```

It is a ratio of squared errors (power), not an amplitude ratio or a standard
deviation ratio.  The reducer resamples the ten paired seed indices and takes
the ratio of bootstrap means.  It never resamples time samples as independent
observations.

The code evidence is direct:

- `run_landau_mise_gate.py:75-95` deposits the paired low- and high-PPC CIC
  densities, applies a candidate `W` only once, and stores
  `mean((rho-rho_ref)**2)`.
- `reduce_noise_shards.py:99-147` forms the seed vectors and the ratio of their
  means with paired seed bootstrap intervals.
- `run_binomial_noise_seed.py:128-175` uses the identical density-error
  construction for raw and compensated filters.
- `reduce_binomial_noise.py:51-87` builds the seed-by-pass matrix and computes
  `means[0]/means`.

No code in those paths constructs a Fourier mask.  The controlled Parseval
test gives identical real- and Fourier-space powers to printed precision and
retains deliberately injected `k=0` and protected-mode contributions.

## Species normalization and charge conversion

Both benchmark initializers assign every electron macro-particle
`w_p=-Lx*Ly/N_p`; hence `sum_p w_p=-|Omega|`.  CIC divides deposited charge by
`dx*dy`, so the mean electron charge density is `-1` and the normalized
electron number density is `n_e=-rho_e=1`.  The uniform immobile ion
background is represented in the field solve by subtracting the mean; it is
not deposited into the density-error metric.  Because the low- and high-PPC
electron deposits have the same total represented charge, their density-error
`k=0` coefficient cancels.

The controlled Landau and E2 checks preserve total charge exactly at printed
double precision under CIC, Gaussian filtering, and four raw-binomial passes.
The recovered E2 directional campaign independently bounds the maximum
relative charge error and Poisson-RHS neutrality defect in
`e2_grid_density_field_noise.csv`.

The sign conversion from electron charge density to electron number density
does not change squared-error ratios.  No species-count or elementary-charge
factor is missing from the reported normalized ratios.

## Reference and independence

The reference is a paired high-PPC **CIC snapshot density**, not an analytic
density and not a filtered copy of the low-PPC candidate.  This avoids putting
the candidate in its own reference.  However, low and reference trajectories
share the integer seed and live matched `sigma=0.25`-cell coupling, and the
same ten seed pairs are used both to choose the MISE width and report its
density-error reduction.  The reported MISE/noise values are therefore
in-sample estimator comparisons, not held-out validation.  Independent
selection/evaluation is a Task 9 requirement and remains open.

## Actual leave-one-out self contribution

The current exact LOO implementation evaluates the snapshot mean of
`c_p^T (C W C^T) c_p` using each particle's actual four bilinear CIC weights.
For normalized probability density, it subtracts `mean_self/N_p` from the
empirical inner product and multiplies by `N_p/(N_p-1)`.

The audit compares this quadratic-form implementation with eleven explicit
particle-deletion and renormalization calculations.  Maximum absolute
differences are:

| Filter | Absolute score difference |
|---|---:|
| CIC | `8.33e-17` |
| Gaussian, `sigma=1` cell | `2.78e-17` |
| raw binomial, `n=4` | `8.33e-17` |

The raw exact-selector trees `5008392` and `5008393` were recovered from
MeluXina and contain 540 `(job, seed, sigma)` self-term rows.  They are
regenerated as `lscv_self_terms.csv`.  This verifies the current exact
implementation; the earlier eight-offset approximation remains superseded.

## Grid density, grid field, and particle force are different quantities

The E2 directional density metric is the density error above.  Its field
metric is

```text
mean_grid [(Ex_candidate-Ex_ref)^2 + (Ey_candidate-Ey_ref)^2]
```

after applying `W` once to deposited density and solving Poisson.  The mean is
removed before each solve, so field `k=0` is explicitly absent; all other
resolved modes, including the protected mode, remain.  This metric is a
grid-field **error-power reduction**, not modal attenuation.

It does not apply the return-path filter to the electric field and does not
gather to particles.  The force used by the dynamics follows
`C -> W -> A -> W -> C^T`, so its modal filter contribution is `W^2` in
addition to the two CIC assignments.  Consequently the E2 field-noise column
cannot be presented as particle-force attenuation or particle-force noise.

## Recovered evidence and verification commands

Recovered byte-for-byte from MeluXina under
`pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/noise_metric_campaigns/`:

- `p1_landau_noise_4867515/`: 42 files, historical Table 4 campaign.
- `p1_landau_noise_4868286/`: 42 files, aligned Gaussian campaign.
- `binomial_noise_landau_weak_5048431/`: 22 files.
- `binomial_noise_e2_5048435/`: 22 files.
- `landau_lscv_static_5008392/` and `5008393/`: 33 files each.

For every tree, a sorted per-file SHA-256 manifest was compared with the
remote source and produced an empty `diff`.  Scheduler accounting confirms all
six simulation jobs completed with exit code zero.  `scontrol write
batch_script` no longer retains jobs `4867515` and `4868286`; therefore their
exact submitted source snapshots remain unavailable.  The current script and
batch file timestamps are 17:02 on 2026-07-23, between the `4867515` and
`4868286` submissions.  The archived `4868286` JSON serializes
`quiet_start_layout=global`; the earlier tree predates that field.  The
contemporaneous audit record explicitly identifies `4867515` as randomized
per-cell and `4868286` as the alignment rerun.  Exact historical source
revisions remain **unknown**.

The complete regeneration command is:

```bash
python3 paper1_source_package/reproducibility/audit_noise_metrics.py
```

It fails on missing seeds or any archived central-value/CI mismatch.  Outputs:

- `results/noise_metric_audit/noise_metric_audit.json`
- `results/noise_metric_audit/weak_landau_sigma1_comparison.csv`
- `results/noise_metric_audit/weak_landau_gaussian_noise.csv`
- `results/noise_metric_audit/filter_family_noise.csv`
- `results/noise_metric_audit/e2_grid_density_field_noise.csv`
- `results/noise_metric_audit/lscv_self_terms.csv`

## Manuscript changes justified by this audit

The authoritative `Dynamics_constrained_Deposition/introduction.tex` now:

1. defines the actual domain-normalized discrete estimator and identifies all
   reported reductions as squared-error/power ratios;
2. removes the unsupported claim that the empirical noise metric integrates a
   high-k band excluding protected modes;
3. states the high-PPC reference, live coupling, seed/window protocol, and
   in-sample limitation;
4. records `2.559` and `5.643` as distinct per-cell/global-loading results and
   corrects the raw-binomial matched-width value to `5.872`;
5. identifies the E2 field metric as pre-gather grid-field error rather than
   particle-force attenuation; and
6. documents the exact particle-phase CIC LOO self term.

No reported measurement was replaced by a theoretical prediction, no
acceptance threshold was changed, and no production simulation was required
for Task 8.
