# Task 9 — independent selector and fair-comparison audit

Audit date: 2026-09-10.  Historical simulation outputs were read or copied to a
separate preservation tree; none was edited, renamed, or deleted.  New runs use
new output directories and seed IDs.  The density-selection audit does not use
rate data to choose a width, and the new rate test does not reuse the earlier
seed IDs.

## Evidence recovered before new simulations

The complete selector-snapshot campaign was found on MeluXina at
`/home/users/u103576/hexapic-main/pic_kde_2d/results/landau_ppc_dynamic_selectors_5049182`
and copied read-only to
`pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/landau_ppc_dynamic_selectors_5049182`.
It contains 192 files (about 24 MiB).  The sorted aggregate SHA-256 is
`3e1c254f4211f2d8a22276bf4b7d9721facb4eb9fe16ecf593ddc518c2bf0f2b`
on both systems.

The archive schema was inspected rather than inferred:

```text
NPZ keys: times, sigmas, probability_density, lscv_scores
times: (51,), 5.0 through 30.0
sigmas: (22,), 0 through 60 cells
probability_density: (51,64,16)
lscv_scores: (51,22)
```

The metadata records globally shuffled velocity quadrature, live matched
Gaussian coupling of 0.25 cell, and 51 snapshots.  PPC 1024 is the candidate
ensemble and PPC 4096 is the paired same-seed CIC reference.

Representative diagnostic commands were:

```bash
ssh u103576@meluxina 'find /home/users/u103576/hexapic-main/pic_kde_2d/results/landau_ppc_dynamic_selectors_5049182 -type f | sort'
python3 -c "import json,numpy as np; p='.../ppc1024/seed0'; print(json.load(open(p+'/metadata.json'))); a=np.load(p+'/selector_snapshots.npz'); print(a.files)"
python3 paper1_source_package/reproducibility/validate_selectors_independent.py
```

The full command is encoded by the script defaults; it refuses to overwrite an
existing result directory.

## Fixed held-out weak-Landau comparison

Seeds 0--4 select a bandwidth and disjoint seeds 5--9 evaluate it.  MISE and
LOO-CV receive exactly the same 22 isotropic Gaussian candidates, the same five
training seeds and the same 51 snapshots: 5,610 width--snapshot evaluations for
each selector.  C4 and Silverman are fixed rules and therefore do not consume
this data-search budget.  All density reductions use the same held-out metric:
the ratio of paired CIC to candidate time-averaged integrated squared
probability-density errors against PPC 4096.

| method | selected width(s), cells | held-out density-error power reduction, 95% CI | selection result |
|---|---:|---:|---|
| CIC | `(0,0)` | `1` | fixed baseline |
| C4 historical rate-CI operating point | `(1,1)` | `3.32021 [3.30651,3.33788]` | fixed before this audit |
| C4 largest candidate below the analytic central ceiling | `(2,2)` | `4.54442 [4.51091,4.57912]` | **not rate-CI certified**; the historical width-2 rate upper CI is 1.10959 |
| historical weak-Landau “Silverman” implementation | `(1.83297,1.83297)` | `4.44132 [4.41021,4.47342]` | isotropized longitudinal-width proxy |
| actual per-direction Silverman formula | `(1.83297,0.458243)` | `3.29838 [3.27734,3.32196]` | held-out density diagnostic only; no corresponding production-rate run |
| MISE | `(7,7)` | `4.99199 [4.95617,5.02573]` | training and test hindsight minimum both 7 |
| LOO-CV | `(12,12)` | `4.94856 [4.91313,4.98055]` | 0.8776% test regret versus the width-7 hindsight minimum |

Independent train/test bootstrapping gives the same MISE interval and
`[4.91518,4.98509]` for LOO-CV.  The MISE training-width stability is 100%; LOO
selects 10--12 cells across bootstrap draws with 94.32% at the reported width.
Across all 252 balanced five-seed/five-seed partitions, MISE always selects 7
cells with zero test regret and LOO-CV always selects 12 cells, with median
regret 0.942% and maximum regret 1.037%.  These results establish only
held-out density performance on this 22-point isotropic Gaussian family.  They
do not establish dynamical or global anisotropic optimality.

Machine-readable evidence is in
`results/selector_independent_audit/heldout_selector_comparison.csv`,
`split_sensitivity.csv`, and `selector_independent_audit.json`.

## Silverman implementation discrepancy

The frozen comparison configuration states
`h_j=(L_j/sqrt(12))*N_particles^(-1/6)` and says to apply an anisotropic pair.
The weak-Landau production scripts instead pass the x-direction cell width as a
single scalar and construct `(sigma*dx,sigma*dy)`, giving 1.83297 cells on both
axes rather than the formula's `(1.83297,0.458243)`.

The E2 evidence is also inconsistent with the promised pair.  The noise runner
uses the scalar as the same number of cells on both axes.  The supplied m=8 rate
runner instead sets one physical width `sigma_phys=sigma_dx*dx` on both axes.
Neither implements the configured `(5.81933 dx,0.727416 dy)` pair.  No raw rate
campaign for that exact directional pair was recovered, and the comparison CSV
contains a planned/pending row rather than an independent measurement.  The
paper must therefore call the historical rows scalar proxies, and it must not
describe the E2 directional pair's erased growth window as measured.  The
analytic admissibility failure can be reported separately from the unresolved
measurement.

## Dense E2 anisotropic search

Only after the transfer and implementation checks passed, a new density/field
noise campaign searched 85 predeclared, model-admissible diagonal Gaussian
points at the binding mode `(12,1)`: CIC plus four radial budget fractions
`{0.25,0.50,0.75,1.00}` and 21 longitudinal allocations from 0 to 1 in steps of
0.05.  The grid/time-discrete budget was frozen at
`q=0.009086217804777913`.  The run used `128x16`, `L=(64,16)`, PPC 32 versus a
paired PPC-1024 reference, `alpha=0.001`, `v_beam=0.8`, `vthermal=0.05`,
`dt=0.05`, 80 steps, stride 2 and evaluation over `t=[0.5,4]`.  Seeds 100--104
selected the point and disjoint seeds 105--109 evaluated it.

Commands and completed jobs:

```bash
sbatch submit_E2_anisotropic_dense_noise_cpu.sbatch
# 5197174, ten-array tasks, all COMPLETED 0:0
sbatch --dependency=afterok:5197174 --export=ALL,NOISE_JOB_ID=5197174 submit_E2_anisotropic_dense_analysis_cpu.sbatch
# 5197184, COMPLETED 0:0
python3 paper1_source_package/reproducibility/analyze_E2_anisotropic_dense_noise.py --input .../e2_anisotropic_dense_5197174/runs --output .../e2_anisotropic_dense --seed-start 100 --seeds 10 --train-seeds 5 --nboot 5000
```

The 43-file raw/result tree was copied to the preservation directory.  The
remote/local aggregate SHA-256 is
`d43ffbe5fccb48010e2aea9fcba12af747bceb6b1ac7851ca1f692246718425b`.

Both training objectives and both held-out objectives select
`sigma_x=0`, `sigma_y=0.2427345998 dy` on the full-budget boundary.  Held-out
density-error reduction is `1.158374 [1.154085,1.161965]`; held-out pre-return,
pre-gather grid-field reduction is `1.075093 [1.070638,1.079892]`.  Nested
train/test bootstrap intervals are `[1.154085,1.162260]` and
`[1.070638,1.079892]`; every training bootstrap selects the same endpoint.
Relative to the nearest sampled competitor (`x` allocation 0.05), its held-out
error advantage is only 0.2908% `[0.2733%,0.3060%]` for density and 0.2181%
`[0.2014%,0.2361%]` for grid field, positive in every test bootstrap.  Charge
error is at most `4.44e-16`, and the neutrality defect is at most `4.69e-15`.

This establishes discrete-design optimality only among the 85 searched,
model-admissible diagonal Gaussian points for the two stated grid metrics.  It
does not establish a continuous optimum, a filter-family-global optimum, or a
particle-force optimum.

## New protected-mode test

The independently chosen new mode is `(10,1)`.  Before launch, the verified
grid/time-discrete calculation predicted baseline growth `0.2960428039`, C4
growth `0.2928595715`, ratio `0.9892473916`, and force transfer
`0.9909549371`, with zero fitted parameters.  The fixed measurement uses
projected linearised delta-f at PPC 512, new seeds 100--109, `alpha=1e-8`,
`128x16`, `L=(64,16)`, `v_beam=0.8`, `vthermal=0.05`, `dt=0.02`, `t_end=60`,
stride 2 and the predeclared fit window `[20,44]`.  Acceptance is the full
paired 95% rate-ratio CI inside `[0.90,1.10]`, at least two candidate e-foldings,
`R^2>=0.98`, and fit-start SNR at least 5.

Commands and completed jobs were:

```bash
sbatch submit_E2_new_mode_m10n1_cpu.sbatch
# 5197185, two array arms, both COMPLETED 0:0
sbatch --dependency=afterok:5197185 --export=ALL,RATE_JOB_ID=5197185 submit_E2_new_mode_m10n1_analysis_cpu.sbatch
# 5197186, COMPLETED 0:0
python3 paper1_source_package/reproducibility/analyze_E2_new_protected_mode.py --input .../e2_new_mode_m10n1_5197185/rate --output .../e2_new_mode_m10n1 --labels base,c4 --mode-x 10 --mode-y 1 --seed-start 100 --seeds 10 --sigma-x-dx 0 --sigma-y-dy 0.24273459979982182 --fit-t-start 20 --fit-t-stop 44 --nboot 5000
```

The 63-file tree was copied unchanged to the preservation directory.  The
remote and local aggregate SHA-256 is
`2f7d42677cb9697a42e2fa963233d8a06ff61b14c9d71603e902b0ce4305f5c4`.
The cluster reducer and local reducer reproduce the same measured values; the
local reduction additionally records the verified discrete prediction.

The CIC rate is `0.2938481 [0.2902463,0.2965732]`; the C4 rate is
`0.2906505 [0.2870336,0.2933860]`.  Their paired ratio is
`0.9891181 [0.9889282,0.9892526]`, within `1.30e-4` of the zero-fit-parameter
prediction.  C4 spans 6.9756 fitted e-foldings, has `R^2=0.99999345`,
fit-start SNR 232.77 and maximum delta-f ratio 0.1010 in the fit window.  It
passes the rate, duration and signal gates.  This independent mode result
supports transfer of the C4 rate prediction from the original `(12,1)` binding
corner toward the less marginal `(10,1)` branch; it does not expand the
protected set retroactively or establish nonlinear fidelity.

## Manuscript and validation changes justified by Task 9

The authoritative source is
`Dynamics_constrained_Deposition/introduction.tex`.  It now:

- reports the fixed held-out MISE/LOO comparison and its PPC-4096 reference;
- retains the historical PPC-16384 table measurements but labels their
  selection/evaluation as in-sample and keeps the loading protocols separate;
- calls the weak-Landau historical Silverman result an isotropized
  longitudinal-width proxy and does not substitute the per-direction held-out
  density result for an unrun production rate;
- reports the E2 Silverman directional pair as analytically inadmissible with
  the direct rate unresolved, because the supplied launchers do not implement
  that pair consistently;
- replaces the three-point/global anisotropic wording with the independent
  85-point result and explicitly limits optimality to the sampled diagonal
  Gaussian grid metrics; and
- adds the measured `(10,1)` transfer test with its fixed gates and
  zero-fit-parameter prediction.

The validation audit was regenerated into a new, non-overwriting
`results/validation_audit_v9` directory.  It contains 40 conditions: 8 pass,
31 fail and 1 unresolved.  The new `(10,1)` row passes.  The configured E2
Silverman pair fails model admissibility while its measurement fields remain
unmeasured; no scalar-proxy result fills them.

## Regression status and unresolved scope

After incorporating the new-mode measurement, the cumulative dependency-free
suite passes 70/70 tests:

```bash
python3 -m unittest discover -s paper1_source_package/reproducibility/tests -p 'test_*.py'
```

Still unresolved are an exact production run of the per-direction Silverman
pair, continuous/global anisotropic optimality, a selector comparison on
particle-force error, and equal end-to-end computational budgets between the
model-based C4 rule and data-driven selectors.  MISE and LOO-CV alone have equal
candidate and data-evaluation budgets in this audit.
