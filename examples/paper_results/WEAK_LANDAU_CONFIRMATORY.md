# Weak-Landau window diagnosis and confirmatory specification

Audit date: 2026-09-11. This record begins with the recovered PPC-1024
histories from MeluXina job `4864819` as development evidence, preserves the
failed v1 campaign and its diagnosis, and then records the completed 40-seed
global-loading v2 confirmation. It never replaces a measurement with a
transfer prediction.

## Existing ten-seed result

The archived files are
`pic_kde_2d/results_raw_rate_audit/landau_sigma_refinement_4864819/shards/ppc1024/sigma{0,1}/seed{0..9}/mode_history.npz`.
Each file contains 901 complex mode samples at cadence 0.1 over `t=[0,90]`.
The reducer uses

```text
abs(mean_seed(complex Ex_hat(1,0,t)))
 -> centered 49-sample geometric envelope
 -> one log-linear fit over the accepted contiguous prefix.
```

It does not average positive magnitudes or average per-seed fitted rates.

| arm | simulation end | floor window and floor | first persistent `5*floor` crossing | look-ahead exclusion | accepted fit | gamma | R2 | fitted amplitude e-foldings | start-to-threshold capacity | exact end reason |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| CIC | 90.0 | `[80,85]`, `3.7575364e-4` | 26.3 | exclude `t>=23.9` | `[5.0,23.8]` | -0.06973751 | 0.999604 | 1.311065 | 1.428401 | noise threshold |
| C4, sigma=1 cell | 90.0 | `[80,85]`, `3.1176968e-4` | 29.0 | exclude `t>=26.6` | `[5.0,26.5]` | -0.06933922 | 0.998368 | 1.490793 | 1.608363 | noise threshold |

The look-ahead exclusion is necessary because a centered 49-point envelope at
time index `i` includes the next 24 diagnostics. The first threshold sample
cannot therefore be used as the fitted endpoint. Neither interval ended at the
simulation end, through a recurrence classifier, or through an envelope-
distortion rule. The historical reducer has no such recurrence or distortion
end classifier; both accepted fits have high R2.

The fitted e-folding product and estimated signal-to-threshold capacity are
different diagnostics. For example, CIC has 1.311 fitted e-foldings but a
1.428 log-capacity from the fitted start to the estimated five-floor threshold.
The initial-time capacity is 1.867. These quantities are retained separately
in `window_diagnosis.json`.

There is also a deterministic hard-limit problem. With the verified continuum
reference magnitude `0.06612796`, two amplitude e-foldings require 30.2444 time
units. From the fixed start `t=5`, the required endpoint is 35.2444. The old
analysis cap at `t=30` permits at most 1.6532 reference e-foldings even if the
history is noiseless. The simulation duration itself is adequate (`t=90`), but
the old fit specification is not.

The complex arrays, coherent histories, fitted envelopes, plot, and subset
scaling table are preserved under
`results/weak_landau_confirmatory_diagnosis/`. The PDF was rendered with
Poppler and visually checked after generation.

## Can 40 seeds help?

For independent random complex fluctuations, the ideal amplitude scaling from
10 to 40 seeds is `sqrt(10/40)=0.5`, which adds `ln(2)=0.6931` of
signal-to-floor capacity. Applied only as an ideal diagnostic to the current
start capacities, that would give 2.122 for CIC and 2.302 for C4. It is not a
prediction of the accepted fitted duration.

All `C(10,n)` development subsets were also evaluated with the same coherent
floor estimator. A log-log fit to the subset-median floors gives exponents
`-0.257` for CIC and `-0.317` for C4 rather than the ideal `-0.5`. Extrapolating
those ten-seed diagnostics alone gives floor ratios 0.700 and 0.645 and capacity
gains 0.357 and 0.439. The corresponding start capacities would be 1.785 and
2.048. A separate tail variance decomposition over `[80,85]` gives projected
40/10 coherent-RMS ratios 0.707 for CIC and 0.568 for C4.

Therefore 40 seeds can plausibly help and C4 is plausibly capable of crossing
the two-e-fold threshold, but the existing ten-seed scaling does not support a
guarantee; CIC is especially borderline. These projections are development
diagnostics only. Seed-coherent discretization or initialization errors will
not average away. This limitation is consistent with the ensemble-averaging
analysis of Touati et al., *Plasma Physics and Controlled Fusion* 64, 115014
(2022), DOI `10.1088/1361-6587/ac9016`.

## Frozen confirmation

The complete machine-readable preregistration is
`weak_landau_confirmatory_v1.json` (SHA-256
`a800d3ae94ec0de9a76537fcd92600c8c47e4ffe02d095f418c5cc561a501508`).
The settings are:

| item | frozen value |
|---|---|
| physics | weak Landau, `k*lambda_D=0.4`, `alpha=0.01`, zero perturbation phase |
| grid | `64 x 16`, `Lx=5*pi`, `Ly=8`, `dx=0.2454369261`, `dy=0.5`, periodic |
| particles | PPC 1024; regular `2048 x 512` marker lattice; randomized per-cell `32 x 32` inverse-normal velocity quadrature; thermal velocity 1 |
| integrator | electrostatic velocity-Verlet, `dt=0.05`, 1800 steps, no early stopping, final time 90 |
| solver | direct periodic `rfft2` Poisson solve; spectral field derivative; no iterative tolerance |
| mode record | complex grid-field `Ex_hat(1,0)` every 0.1 |
| density/field snapshots | every 0.5 over `[5,30]` |
| fresh seeds | exactly `202609100..202609139`; no sequential additions |
| baseline | CIC |
| proposed | Gaussian C4 at one cell, with `sigma_x=dx`, `sigma_y=dy`, same W on deposition and return |
| comparator | exact LOO-CV width `7.244293147028428` cells from static selector job `5008392`; not the 6.61 proxy |
| fit | coherent complex mean, magnitude, centered one-period geometric envelope; start 5, cap 45; floor from `[70,90]`; first persistent five-floor crossing; 24-sample look-ahead guard |
| absolute reference | fully discrete CIC rate `-0.06636237273118674`; two-sided 10% interval gate |
| bootstrap | 5000 paired fixed-seed resamples, seed `20260910`; central windows locked first |

The comparator's selection campaign used global velocity shuffling, whereas
this confirmation uses per-cell loading. That protocol difference is explicit;
the width remains a fixed external statistical comparator and is not reselected
after seeing new rates.

Within each fresh seed, one PPC-1024 particle state is byte-copied into all
three arms and the `x,y,vx,vy,w` SHA-256 is saved. The previous ten seeds remain
development data. Any 50-seed pooled result must be secondary and labeled
pooled.

Stage 1 writes `window_decisions.json`, including raw-file hashes, accepted
endpoints, exact end reasons, fitted e-foldings, recurrence diagnostics and
envelope-quality flags, while computing no comparative ratios. Stage 2 refuses
to run if the configuration, histories, seed set, or recorded windows changed.
It then uses those frozen intervals for the central estimates and paired
bootstrap.

Promotion requires every item below; none is replaced by a CIC/CIC ratio:

1. CIC and C4 each span at least two fitted amplitude e-foldings.
2. Both have fit-start SNR at least 5, R2 at least 0.90, and satisfy the frozen
   point, duration, and 80% bootstrap-quality requirements.
3. The 95% interval for `abs(gamma_CIC)/abs(-0.0663623727)` lies wholly within
   `[0.90,1.10]`.
4. The paired 95% interval for `gamma_C4/gamma_CIC` lies wholly within
   `[0.90,1.10]`.
5. All 40 raw seed sets, three-arm particle hashes, source/config hashes, mode
   histories, and noise snapshots pass traceability checks.

## Same-campaign noise measurements

Each fresh seed also has a PPC-4096 unfiltered, per-cell CIC reference. It is a
finite-PPC reference, not an exact continuum solution. For all three PPC-1024
arms the reducer computes, on the same `[5,30]` window:

- density-error power: `mean_grid[(rho_arm-rho_ref)^2]`, after the arm's
  deposition filter, retaining all grid modes; and
- grid-field-error power:
  `mean_grid[(Ex_arm-Ex_ref)^2+(Ey_arm-Ey_ref)^2]`, after deposition filtering
  and Poisson but before return filtering and CIC gather.

The reduction is the ratio of ensemble-and-time mean CIC error power to arm
error power, with seed-level bootstrap intervals. These are power ratios, not
amplitude attenuation and not particle-force error. The saved per-seed complex
traces and single-run fit diagnostics are reported separately from the primary
coherent-ensemble validation.

## Implementation checks and commands

Commands already run locally:

```text
PYTHONPATH=pic_kde_2d python3 paper1_source_package/reproducibility/diagnose_weak_landau_window.py
PYTHONPATH=pic_kde_2d python3 pic_kde_2d/scripts/run_weak_landau_confirmatory.py smoke-test --output .../weak_landau_confirmatory_smoke_v3
PYTHONPATH=pic_kde_2d python3 pic_kde_2d/scripts/run_weak_landau_confirmatory.py preflight --output .../weak_landau_confirmatory_preflight_v5
python3 -m unittest paper1_source_package/reproducibility/tests/test_weak_landau_confirmatory.py -v
bash -n paper1_source_package/reproducibility/meluxina/submit_weak_landau_confirmatory_*_v1.sbatch
ssh u103576@meluxina "sacct -j 5049265 --format=JobID,State,Elapsed,Timelimit,AllocCPUS,MaxRSS,ReqMem -P"
```

The miniature end-to-end capture path passes for all three arms and is clearly
labelled nonphysical. The protected-mode preflight measures deposition,
post-Poisson grid field, and full particle-force transfer. Maximum absolute
prediction/measurement error is `2.22e-16`. C4 gives
`W=0.9951924507`, `W^2=0.9904080139`; the LOO comparator gives
`W=0.7765396556`, `W^2=0.6030138368`. All seven targeted tests pass, including
a synthetic 40-seed end-to-end exercise of the stage lock and reducer.
Production commands additionally require the exact passing `preflight.json`;
they abort if any runtime source hash differs and copy those hashes into every
seed's metadata. Stage 1 verifies that all arms share that preflight and source
record, and stage 2 applies the same check to the PPC-4096 references.

Historical scheduler accounting for job `5049265` shows the recovered
PPC-4096 groups completed in about 1.43--1.57 hours under the four-hour short
QOS and used about 7.9--8.7 GiB for their multi-seed tasks. This supports, but
does not guarantee, the requested four-hour/12-GiB reference-task allocation.

The current MeluXina source differs bytewise in three files. Audit diffs show
that the remote particle initializer only adds an optional
`position_jitter=False` argument, and the two filter files only add binomial
families; the default regular positions, per-cell velocity branch, Gaussian
transfer, matched deposition, and matched gather used here are unchanged.
The cluster-native smoke and preflight were then run after deployment and
passed with maximum transfer error `2.22e-16`. Their retained local copies are
under `results/weak_landau_confirmatory_remote_preflight_20260911/`. The
preflight records the actual remote hashes, including the three known added-
function differences, and is required unchanged by every production task.

After those checks passed, the fixed arrays were submitted on 2026-09-11:

- `5199341`: 40 PPC-1024 tasks, each running the paired CIC, C4, and exact
  LOO-CV arms from one byte-identical per-seed initial particle state;
- `5199342`: 40 PPC-4096 CIC reference tasks for the error-power metrics.

No stage-1 window or stage-2 ratio job was submitted concurrently. Stage 1 was
allowed only after all 40 low-PPC tasks completed; the v1 Stage 2 remained a
separate manual step and was never run after the loading failure was found.

The low-PPC array completed all 40 tasks.  A subsequently generated Stage-1
record has SHA-256 `c50ae4a356942fb3ee0f446c982a97716f129769c50ce9c79fc7bdaceb09beda`.
It verifies the 40 fixed seeds, within-seed three-arm particle hashes, frozen
configuration hash `a800d3ae...1508`, remote preflight hash
`48236b28...0ad7`, and runtime-source hashes, and explicitly records
`rate_ratios_computed=false`.  It nevertheless finds no accepted interval:
all arms cross the five-floor threshold at `t=5.0`, giving zero duration and
zero fitted e-foldings.  The recorded floors are `0.00587779` (CIC),
`0.00640412` (C4), and `0.0209183` (LOO-CV).  These are anomalously high
relative to the development campaign and must be diagnosed before Stage 2;
no comparative-rate result is inferred from them.

Reference-array tasks 31 and 32 were left running and every completed result
was preserved.  Only still-pending elements 33--39 were canceled after their
final and partial output paths were verified absent.  Their exact seed set
`202609133..202609139` was resubmitted as packed GPU-partition job `5200213`.
The job completed in 1:03:53 with exit code zero.  The reference tree now has
exactly 40 final seed directories and no partials.  The continuation requested
one GPU and eight CPU cores and ran the seven independent seeds concurrently.
The simulation is NumPy/Numba CPU code, so this changed scheduler placement,
not the numerical implementation or source hashes, and does not claim CUDA
acceleration.

The complete low/reference production tree was copied without overwriting to
`results/weak_landau_confirmatory_production_20260911/`.  The relative-path
file manifest has matching local and MeluXina SHA-256
`91a4ce387dc70d2adb006710876133f14a14be0979af8a25d551f4de9910f7d0`.

## v1 loading-admissibility failure

The Stage-1 anomaly is now isolated.  Passing `grid_shape=(64,16)` at
`run_weak_landau_confirmatory.py:178` selects the branch at
`particles2d.py:139--164`: each field cell receives a separately permuted but
identical 32 by 32 inverse-normal velocity tensor.  Thus PPC 1024 supplies only
32 distinct unperturbed longitudinal velocities per cell.  Its extreme node is
`|vx|=2.15387469`, but the frozen mode has `omega_r=1.28505697`, `k=0.4`, and
`v_phase=3.21264242`.  The resonant velocity lies outside the represented
longitudinal support.

This defect is common to every seed and cannot fall as `1/sqrt(S)`.  Across the
40 v1 histories, the complex relative seed scatter stays below `0.7%` through
`t=30`; CIC and C4 reach envelope minima near `t=20.8/20.9` and regrow by
factors `3.78/3.94` by `t=45`.  The late `[70,90]` estimator therefore measures
the shared recurrence/regrowth rather than an independent particle-noise
floor.  In contrast, the preserved job-4864819 traces retain the same regular
spatial marker lattice but shuffle a global inverse-CDF velocity template;
their CIC/C4 envelopes continue decreasing through the required `t=35.2444`.

The v1 campaign is classified `FAIL_LOADING_ADMISSIBILITY`, not a physical
rate failure.  Stage 2 must not be run and the completed v1 error-power files
must not be promoted as a same-protocol Landau validation, because the PPC-1024
and PPC-4096 per-cell tensor rules have different, both truncated,
longitudinal supports.  The machine-readable diagnosis and the visually
checked PDF are in
`results/weak_landau_confirmatory_stage1_failure_diagnosis_20260911_v4/`.
The earlier no-PDF attempts remain preserved in their original directories.

## Corrective diagnostic pilot

Before any replacement production campaign, a three-seed pilot changes only
the velocity assignment to the historical global inverse-CDF shuffle.  Spatial
positions remain a regular 2048 by 512 lattice, hence exactly 1024 particles
per field cell before the density perturbation.  The global longitudinal rule
has 1,048,576 distinct nodes, reaches `|vx|=4.90096421`, and places 690 nodes at
or above the positive resonance.  Grid, perturbation, solver, filter chain,
time step, duration and diagnostic cadence remain those of v1.

The MeluXina-native preflight passed with pilot-source SHA-256
`c95411c5...fe6c` and the unchanged base configuration/source preflight.  Fresh
pilot seeds `202609180..202609182` run paired CIC/C4 arms and are expressly
excluded from any future 40-seed confirmation.  GPU-partition job `5200354`
was submitted only after the support and inherited transfer checks passed.

The pilot completed exit zero.  Its frozen `[5,20]` diagnostic gave CIC/C4
rates `-0.0700812/-0.0715235`, `R^2=0.99887/0.99884`, and only
`1.051/1.073` fitted e-foldings.  It passed the loading-behaviour check and
computed no comparative rate ratio.

## Completed global-loading v2 confirmation

The replacement specification is
`weak_landau_confirmatory_v2_global_velocity.json` (SHA-256
`d107eeda9957de42419cd0a0307887336544e20fc4ab208ce1b99620d134d707`).
It retains the v1 grid, time step, duration, cadence, solver, filters, PPC,
perturbation and stopping/window rules, but switches the longitudinal velocity
quadrature to the verified global inverse-CDF assignment.  It fixes 40 new
seeds `202609300..202609339` and does not pool v1 or pilot seeds.

All 17 active CPU/GPU-partition Slurm elements in jobs
`5200457/58`, `5200485/86/87/88` completed with exit code zero.  The local and
MeluXina raw-tree manifests agree at SHA-256 `47019511...8a7a1`; the local tree
contains exactly 40 low-PPC seed directories, 40 reference directories, no
partials and no nonempty stderr.  CPU placement consumed 6.8958 node-hours;
GPU requests affected scheduling only because the NumPy/Numba solver remained
CPU-backed.

Stage 1 was run and hash-locked before inspecting a comparative ratio.  Its
SHA-256 is `cb724768fad3950d5ca181882285284b7f253d75643cee0c3e446fd1958b0cbf`:

| arm | accepted fit | exact end reason | gamma | fitted e-foldings | R2 | fit-start SNR |
|---|---|---|---:|---:|---:|---:|
| CIC | `[5.0,30.1]` | persistent `5*floor` crossing at 32.6; centered-envelope guard excludes `t>=30.2` | -0.0685811 | 1.7214 | 0.999674 | 31.07 |
| C4, sigma=1 cell | `[5.0,35.6]` | persistent `5*floor` crossing at 38.1; centered-envelope guard excludes `t>=35.7` | -0.0678552 | 2.0764 | 0.998440 | 45.74 |
| exact LOO-CV comparator | `[5.0,28.0]` | persistent `5*floor` crossing at 30.5 | -0.129881 | 2.9873 | 0.995398 | 109.48 |

Thus C4 reaches two fitted amplitude e-foldings, but the paired CIC baseline
does not.  This duration failure was known before Stage 2 and is not replaced
by the larger signal-to-threshold capacities.

The subsequently run locked Stage 2 gives C4/CIC
`0.989415 [0.967884,1.013827]` and CIC/fully-discrete-reference
`1.033434 [0.989774,1.078710]`; both rate intervals pass.  The complete
promotion nevertheless remains **FAIL** because the frozen rule requires both
CIC and C4 to span two e-foldings.  Same-campaign C4 density and pre-return-
filter grid-field squared-error-power reductions are
`3.25865 [3.25180,3.26500]` and `1.61280 [1.60345,1.62155]`.  The exact LOO-CV
comparator fails its rate gate with ratio
`1.89383 [1.67902,2.13875]` and worsens the grid-field error
(`0.63238` reduction ratio).

The primary estimator averages the 40 complex Fourier coefficients before
taking their magnitude and one-period geometric envelope.  It does not average
positive magnitudes or fitted seed rates.  The full coherent complex values,
envelopes, fit curves, floors, censor markers and accepted masks are under
`results/weak_landau_confirmatory_v2_window_diagnostic_20260911_r2/`; the PDF
was rendered and visually checked.

## Common-window sensitivity analysis

The registered arm-specific windows and result above remain primary.  The
separate non-overwriting reducer
`analyze_weak_landau_common_window_sensitivity.py` verifies the same raw-file,
configuration and Stage-1 hashes, then fixes both CIC and C4 to the intersection
`[5.0,30.1]`.  It preserves the same operation order (complex seed mean,
magnitude, 49-sample geometric envelope, log-linear slope) and resamples the
same seed indices in both arms for 5000 draws.

The common-window slopes are `-0.06858111997` and `-0.06986857364`, giving
C4/CIC `1.0187727 [1.0153586,1.0219306]`.  Relative to the primary
arm-specific `0.9894154`, the point estimate shifts by `0.0293573` (2.936
percentage points) and crosses unity.  Both intervals are inside the 10% rate
band, but the common-window fits span only 1.7214 and 1.7537 e-foldings, so the
sensitivity analysis does not change the primary duration failure.

## Files added and changed

- Added `diagnose_weak_landau_window.py` and its new, non-overwriting result
  directory.
- Added the frozen JSON specification.
- Added `pic_kde_2d/scripts/run_weak_landau_confirmatory.py`.
- Added `analyze_weak_landau_confirmatory.py` with locked stage-1/stage-2
  reduction.
- Added four MeluXina submission scripts for low-PPC, high-PPC reference,
  stage-1 windows, and stage-2 comparisons, plus the narrowly scoped packed
  GPU continuation for missing reference seeds 33--39.
- Added the loading-failure reducer/figure, the global-loading pilot runner and
  its GPU launcher.
- Added regression/integration tests for the locked reducers and artifacts.
- Added the separate paired common-window reducer and retained its 5000 draw
  rates in `results/weak_landau_confirmatory_v2_common_window_sensitivity_20260911/`.
- Updated the authoritative manuscript to document the actual window-ending
  mechanism, the structural `t=30` cap, the `t=35.24` requirement, and the
  independent-noise limitation of coherent ensemble averaging. Added the
  verified Touati et al. citation.

## Remaining limitations

- The v1 production campaign remains `FAIL_LOADING_ADMISSIBILITY`; its Stage 2
  correctly remains unrun and its outputs are not pooled into v2.
- The completed v2 campaign fails the unchanged two-e-fold baseline-duration
  gate.  Its passing rate intervals are conditional findings, not a complete
  weak-Landau validation pass.  Adding seeds sequentially is prohibited and
  extending only the already decayed tail is not a justified remedy.
- The PPC-4096 CIC error reference is finite-PPC; its residual sampling error is
  represented through paired seed resampling rather than called exact.
