# E2 symmetric-beam-drift scan

## Scope and status

This is a preregistered, mode-projected linearized-delta-f test of the E2
linear mechanism at `(mx,my)=(12,1)`.  It does not establish unrestricted
nonlinear full-f fidelity.  The scientific design was frozen before any scan
simulation started.  GPU job `5201218` failed after 20 seconds before any
simulation initialized because its launcher pre-created runner-owned output
directories.  All 40 first-batch tasks raised the runner's non-overwrite
`FileExistsError`; zero mode histories were produced.  The failed 43-file tree
and scheduler logs are preserved locally.  After a launcher-only fix, the
unchanged campaign was resubmitted as GPU job `5201231`.  That job exhausted
the preregistered four-hour limit after completing 120/150 histories.  All
three arms are complete at drifts `0.60`, `0.70`, `0.76`, and `0.80`; the 30
histories at `0.818` were not written and remain
`UNRESOLVED_BUDGET_TIMEOUT` in the original four-hour result. No continuation
was silently folded into that result. After the estimator diagnostic below,
the same 30 preselected histories were submitted as a separately labelled
budget extension. Job `5202227` completed all 30 files, but every arm fails
the frozen fit-start-SNR rule and three of ten seeds per arm leave the
linear-`delta f` regime. The extension is therefore computationally complete
and scientifically unresolved; its descriptive rates are not accepted
measurements.

The initial density-selection statement is deliberately narrow.  At fixed
seed, positions, charges, thermal samples, beam membership, the initial full
density, initial perturbation weights, and initial perturbation density are
identical across drift.  Later evolving snapshots are not asserted to have an
identical density objective.

## Frozen numerical design

- periodic projected mode `(12,1)` on `128x16`, `L=(64,16)`;
- `dx=0.5`, `dy=1`, `vthermal_x=vthermal_y=0.05`;
- PPC 512 (`2048x512`, 1,048,576 markers), `alpha=1e-8`;
- explicit-midpoint frozen-characteristic linearized delta-f, `dt=0.02`,
  diagnostic stride 2;
- perturbation CIC deposition, periodic spectral Poisson, nodal field,
  matched Gaussian return filter, and CIC gather;
- fresh paired seeds `200..209`; no development-seed pooling or seed
  rejection;
- 5000 paired seed bootstraps, bootstrap seed `120912` plus the frozen drift
  index; no time-sample resampling;
- fixed fit start `t=24.32`; endpoints below were set from the model to provide
  at least five predicted constrained-arm e-foldings, with a two-time-unit
  recorded tail.

The exact source used on MeluXina was repository revision
`ed6dd96d16d0b70a53bf9486fd78a270ab3486d6`; all six production source hashes
matched the audited local copy before submission.  The source was not modified
for this campaign.  Each result writes a separate campaign manifest containing
the frozen config hash and pairing key.

## Prediction before simulation

The audited warm, oblique grid/time-discrete dielectric uses spatial aliases
through `qmax=8` and temporal aliases through `lmax=1024`.  The tracked purely
growing branch becomes marginal at

`vbeam = 0.8191540596575886`.

The cold longitudinal threshold was not used.  Comparison with `qmax=10`,
`lmax=2048` changes the marginal drift by `3.55e-9`, any selected baseline rate
by at most `4.50e-8`, the 10% bandwidth boundary by at most `9.08e-8 dy`, and
the safety width by at most `6.52e-8 dy`.

| `vbeam` | numerical CIC gamma | fixed-filter ratio | 10% boundary `sigma_y/dy` | frozen constrained `sigma_y/dy` | fit stop | run stop |
|---:|---:|---:|---:|---:|---:|---:|
| 0.600 | 0.32083553 | 0.992700 | 0.87734430 | 0.62841615 | 40.76 | 42.76 |
| 0.700 | 0.26590942 | 0.985457 | 0.62322734 | 0.44621394 | 44.12 | 46.12 |
| 0.760 | 0.19897475 | 0.969398 | 0.43120806 | 0.30878397 | 50.80 | 52.80 |
| 0.800 | 0.11746394 | 0.900000 | 0.24273460 | 0.17385924 | 69.16 | 71.16 |
| 0.818 | 0.02928248 | 0 (no positive growing root) | 0.05930194 | 0.04248035 | 204.08 | 206.08 |

The fixed directional Gaussian is `(sigma_x/dx,sigma_y/dy)=(0,0.2427346)` for
every drift.  Its implemented transfer at `(12,1)` is
`W=0.9954671953989629` per application and
`W^2=0.9909549371154771` through matched deposition and gathering.  Controlled
FFT tests measured all fixed and constrained transfers to `3.34e-16` absolute
error.

The constrained arm solves for predicted rate ratio 0.95, rather than using
the exact 0.90 admissibility boundary.  This freezes a five-percentage-point
allowance for measurement uncertainty.

## Sample size and budget

Ten seeds were fixed from the existing directional Gaussian E2 result: its
ten-seed paired ratio bootstrap had SD `0.00398` and a 95% interval width
`0.01535`, sufficient to distinguish the 0.95 target from the 0.90 boundary.
No sequential seed addition is allowed.

The 15 configurations and ten seeds give 150 histories.  Four batches run at
most 40 histories concurrently.  Extrapolating job `5195640` gives 2.83 node
hours; the hard reservation is one node for four hours.  CPU raw/reducer jobs
`5201211/5201212` never started and were cancelled after Slurm returned
`AssocGrpGRESMinutes`; `sshare` showed 3945 CPU-node minutes (65.75 hours),
confirming that the CPU allocation had been exhausted.  The scheduler-only
amendment preserves every scientific field and moves the same four-hour cap to
one GPU node, whose host CPUs execute the existing NumPy/Numba workload.
The first GPU attempt, `5201218`, then exposed an output-ownership error in the
launcher: it placed cache/log files inside a path the runner requires not to
exist.  The replacement puts those launcher artifacts in sibling trees.  The
focused regression suite now checks this interface and passes 7/7 tests; no
model, seed, arm, window, duration, or acceptance field changed.

Job `5201231` ran for `04:00:38` and Slurm recorded `TIMEOUT`.  Its four
completed drift blocks contain 120 mode histories, 120 result JSON files and
120 campaign manifests.  The copied 513-file source tree matches MeluXina at
aggregate SHA-256 `0d00fe2d...75114`.  The original strict reducer fails at the
first missing `0.818` input.  Its explicit timeout-censored mode accepts only a
whole missing drift block backed by the recorded Slurm timeout, writes null
measurements for that block and leaves all frozen estimators unchanged.

## Frozen measurement and reporting rules

The rate is the slope of `ln(mean_seed(abs(Ex_hat(12,1))))` on the frozen
per-drift window.  Rates and ratios use the same paired seed-index bootstrap.
For every arm the reducer reports the central rate and 95% interval, predicted
rate, prediction-inside-interval flag, relative prediction error, fit R2,
fit-start SNR from the existing two-time-unit pre-fit floor estimator, fitted
e-foldings, and maximum delta-f weight ratio.

A filtered rate is resolved only with positive central and lower-limit growth,
`R2>=0.98`, SNR `>=5`, at least two fitted e-foldings, and delta-f ratio `<0.5`.
The rate-preservation pass additionally requires the paired 95% ratio interval
inside `[0.90,1.10]`.  Otherwise the result remains `UNRESOLVED` or `FAIL` under
the frozen rule; no seeds or windows are added.  Model calibration is reported
separately from measured rate preservation.

## Measured result under the fixed budget

| `vbeam` | CIC gamma [95% CI] | fixed ratio [95% CI] | constrained ratio [95% CI] | status |
|---:|---:|---:|---:|:---|
| 0.600 | 0.32379 [0.31998, 0.32741] | 0.99278 [0.99268, 0.99287] | 0.95059 [0.94991, 0.95122] | measured gates pass |
| 0.700 | 0.27170 [0.26507, 0.27716] | 0.98595 [0.98537, 0.98636] | 0.95173 [0.94970, 0.95317] | measured gates pass |
| 0.760 | 0.20978 [0.19891, 0.21714] | 0.97269 [0.96971, 0.97426] | 0.95546 [0.95052, 0.95804] | measured gates pass |
| 0.800 | 0.14475 [0.12005, 0.15455] | 0.94064 [0.90797, 0.94692] | 0.96986 [0.95376, 0.97301] | measured gates pass |
| 0.818 | --- | --- | --- | unresolved - budget timeout |

All eight completed filtered comparisons have their paired interval inside
`[0.90,1.10]`, R2 at least `0.9990`, fit-start SNR at least `21.17`, and
5.06--6.30 fitted e-foldings.  The fixed transfer nevertheless produces an
increasing relative change as the branch approaches marginality, whereas the
per-drift constrained width remains within tolerance.

Absolute rate prediction is less accurate than preservation.  The CIC
relative prediction error increases from 0.91% at drift `0.60` to 18.85% at
`0.80`; at `0.80` its predicted `0.11746` is below the measured interval
`[0.12005,0.15455]`.  Both filtered absolute predictions are also below their
measured intervals there.  At `0.76`, the filtered predictions fall just below
their narrow intervals even though their measured ratios pass.  Model
calibration and useful measured rate preservation are therefore separate
findings.

## Post-hoc estimator diagnostic

The preceding values and gates remain the registered primary results. The
diagnostic fits every seed separately on the same frozen samples and records
the time-dependent contribution
`abs(Ehat_s(t))/sum_s(abs(Ehat_s(t)))`. At `vbeam=0.80`:

| quantity | CIC diagnostic value |
|:---|---:|
| registered slope of `ln(mean(abs(Ehat_s)))` | 0.1447459 |
| mean individual-seed slope | 0.1244508 |
| individual-seed SD; median; range | 0.0233284; 0.1227335; [0.0938376, 0.1638252] |
| fitted-exponential mixture slope | 0.1447311 |
| effective seed count, fit start -> end | 7.405 -> 3.280 |
| largest amplitude fraction, fit start -> end | 0.2213 -> 0.4808 |
| correlation of seed rate with contribution change | 0.7225 |

The arithmetic-mixture construction reproduces the registered estimator to
`1.48e-5`. Its `0.0202951` excess over the mean individual rate is 74.39% of
the registered absolute theory gap. The residual mean-individual discrepancy
from numerical theory is `0.0069869`, or 5.61% using the manuscript's measured
denominator convention. This establishes estimator reweighting as the main
source of the 18.85% number in these ten histories, while leaving a smaller
finite-particle/model residual. It does not establish that all of the residual
is a theoretical error.

For the filtered comparisons, the registered `vbeam=0.80` ratios are `0.94064`
and `0.96986`, whereas ratios of the two individual-rate means are `0.90891`
and `0.95446` for fixed and constrained filtering. These are explicitly
diagnostic alternatives, not replacement estimates. The CIC-versus-filter
individual rates are correlated above `0.9998` within paired seeds, consistent
with a seed-loading contribution shared across arms; the implemented transfer
audit independently still agrees to `3.34e-16`.

The endpoint check is small: refitting the current seeds through the historical
stop `59.68` changes CIC from `0.1447459` to `0.1431473`. The earlier corner
campaign gives `0.1268009` on `[24.32,59.68]`, but used seeds `0..9` rather than
`200..209`. The recorded alpha, benchmark, time step, mode, marker dynamics,
PPC, zero CIC widths, target-mode projection, and validity threshold agree.
The historical output does not record its source revision, however, so the
remaining `0.0163464` shared-endpoint campaign difference cannot uniquely
separate seed loading from implementation history. It remains an explicit
identifiability limit rather than an intrinsic-theory diagnosis.

## Explicit post-timeout extension

The extension budget was frozen after the diagnostic and before any extension
result. It changes no scientific field: drift index 4 (`vbeam=0.818`), seeds
`200..209`, the three original arms, window `[24.32,204.08]`, run stop `206.08`,
and every acceptance rule are unchanged. It adds a one-node, eight-hour GPU
placement cap for exactly 30 histories, with no sequential seed addition.
MeluXina job `5202227` was submitted at `2026-09-11T21:36:21`, completed with
exit code `0:0` in `02:21:40`, and wrote all 30 histories. Its separate output
tree and reporting label remain "post-timeout extension of the preregistered
campaign". The original `5201231` status remains 120/150 and incomplete.

On the frozen `[24.32,204.08]` window, the reducer returns descriptive rates
`0.114044`, `0.101783`, and `0.113689` for CIC, fixed Gaussian, and constrained
Gaussian, respectively. The corresponding filtered/CIC ratios are `0.892489`
and `0.996893`. These quantities are not promoted: fit-start SNR values are
only `3.563`, `2.854`, and `3.531`, and the linear-`delta f` guard is exceeded.
The fixed-filter model has no positive growing branch. An increasing finite-
time mean amplitude is not evidence of a resolved continuum instability or
stabilisation.

## Commands and artifacts

```bash
python3 paper1_source_package/reproducibility/design_E2_beam_drift_scan.py \
  --output paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911
python3 paper1_source_package/reproducibility/verify_E2_beam_drift_scan_transfer.py \
  --config paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911/frozen_campaign.json \
  --output paper1_source_package/reproducibility/results/e2_beam_drift_scan_transfer_20260911
python3 paper1_source_package/reproducibility/diagnose_E2_rate_estimator.py \
  --config paper1_source_package/reproducibility/results/e2_beam_drift_scan_design_20260911/frozen_campaign.json \
  --scan-root pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260911/e2_beam_drift_scan_5201231/runs \
  --registered paper1_source_package/reproducibility/results/e2_beam_drift_scan_analysis_20260911_timeout_r6/E2_beam_drift_scan.json \
  --transfer-summary paper1_source_package/reproducibility/results/e2_beam_drift_scan_transfer_20260911/transfer_summary.json \
  --corner-root pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/e2_joint_corner_4913408 \
  --corner-launcher pic_kde_2d/scripts/submit_E2_joint_corner_sweep_gpu.sbatch \
  --output /a/new/nonexistent/diagnostic/output
python3 -m unittest discover -s paper1_source_package/reproducibility/tests -p 'test_*.py'
```

The pre-submission suite passed 76/76 tests; after adding launcher, timeout,
result-anchor and estimator-diagnostic regressions the full suite passes 84/84
(the original focused E2 suite passes 9/9 and the diagnostic suite passes 5/5).
The frozen configuration SHA-256
is `e664460d392a216b953bc87d3859c3bce9e1cce2f1652ab9417c1a58b3baf446`.
The production command is preserved in
`meluxina/submit_E2_beam_drift_scan_gpu.sbatch`.  The final timeout-censored
CSV/JSON and paired draws remain byte-identical in
`results/e2_beam_drift_scan_analysis_20260911_timeout_r8`; only its visually
verified three-panel PDF/300-dpi PNG carry the requested wording changes. The
diagnostic tables, including every seed rate and every time-dependent
contribution, are in `results/e2_rate_estimator_diagnostic_20260911_r3`
(aggregate SHA-256 `8eb00f8d...ddbb`). Exact incident, timeout, diagnostic and
extension-budget evidence is in
`results/e2_beam_drift_scan_design_20260911/launcher_incident_5201218.json` and
`campaign_timeout_5201231.json`, `extension_budget_amendment_20260911.json`, and
`extension_submission_5202227.json`. The completed extension is preserved at
`pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260912/e2_beam_drift_scan_extension_5202227/`;
the combined, gate-aware reduction and visually checked figure are in
`results/e2_beam_drift_scan_analysis_20260912_complete_extension_r9/`.
