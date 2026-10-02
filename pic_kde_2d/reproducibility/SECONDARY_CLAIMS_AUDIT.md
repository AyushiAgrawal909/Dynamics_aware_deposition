# Task 10 — Secondary-claims audit

Date: 2026-09-10

Status: complete with narrowed claims. No new production simulation was launched.
The recovered raw histories were sufficient to test the retained statements,
and the missing nonlinear/controller observables cannot be manufactured by
extending a run that did not save them.

## Evidence recovered from MeluXina

The historical paths were found under `/home/users/u103576/logs` and
`/home/users/u103576/hexapic-main/pic_kde_2d`. Copies are preserved under
`results_raw_rate_audit/recovered_meluxina_20260910/`.

| Evidence | Files | Bytes | Remote/local aggregate SHA-256 |
|---|---:|---:|---|
| `df_breakdown_4873222` | 47 | 247,640 | `70abf8066e831172369676f0a25a8259cdb55ad74618d1be897009ccd8ef736e` |
| `df_fullf_overlay_4873285` | 17 | 222,227 | `ee6d8cb3a7127ea2c619791e633390c922858d7c0046b5ceb51b6ccb6b9d0ebe` |
| `component_cost_5049224` | 32 | 22,678 | `467e3ee0cbea67e950ed238de0365796e28a7fa8544628946e9edf05c1b329b3` |
| `G0_locality_headroom_ceiling_baseline` | 4 | 244,396 | `d17ee858d86432fb6e13780352565bd6c4b7995763d1444b60fca18ad7d45fc2` |
| `G0_conservation_precheck` | 3 | 3,538 | `707bd94bf5fe3b4b2ef7df786ec80d368985dad342b303a32ae15d930adfeb12` |

The delta-f breakdown contains 15 independent `mode_history.npz` files: three
amplitudes and seeds 0–4. The overlay contains five full-f histories and
explicitly reuses the `alpha_0p1` delta-f histories from job 4873222. The cost
campaign contains every per-method JSON/CSV record for five order-rotated
repeats. Exact cost and G0 generator/reducer/config/test snapshots are preserved
in `source_snapshot/`.

The exact delta-f source revision used by jobs 4873222 and 4873285 is unknown.
Neither log records a commit, and the recovered result schema is older than the
current runner schema. This remains unknown rather than assigning the present
worktree revision retrospectively.

## Commands used

Discovery:

```text
ssh u103576@meluxina "find /home/users/u103576 /mnt/tier2/users/u103576 ..."
ssh u103576@meluxina "find ... -path '*4873222*' -o -path '*4873285*' ..."
ssh u103576@meluxina "grep -R ... '1.000033\|384.*dx' ..."
```

Preservation and hash verification:

```text
scp -r u103576@meluxina:/home/users/u103576/logs/df_breakdown_4873222 ...
scp -r u103576@meluxina:/home/users/u103576/logs/df_fullf_overlay_4873285 ...
scp -r u103576@meluxina:/home/users/u103576/hexapic-main/pic_kde_2d/results/component_cost_5049224 ...
find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

Regeneration and independent audit:

```text
python3 scripts/analyze_delta_f_breakdown.py --input <recovered>/df_breakdown_4873222 --output reproducibility/results/secondary_claims_audit/regenerated_breakdown_v2 --threshold 0.5 --seeds 5 --nboot 100000 --bootstrap-seed 9102026
python3 scripts/analyze_delta_f_fullf_overlay.py --fullf <recovered>/df_fullf_overlay_4873285/fullf --deltaf <recovered>/df_breakdown_4873222/alpha_0p1 --output reproducibility/results/secondary_claims_audit/regenerated_overlay --seeds 5
python3 <recovered>/source_snapshot/reduce_component_cost.py --input <recovered>/component_cost_5049224 --output reproducibility/results/secondary_claims_audit/regenerated_cost --expected-repeats 5
python3 reproducibility/audit_secondary_claims.py --recovered <recovered> --h1 results/H1_analytic_local --learned-root paper3_learned_controller --output reproducibility/results/secondary_claims_audit --nboot 100000
pytest -q tests/test_phase5.py tests/test_secondary_claims_audit.py tests/test_adaptive_sigma_controller.py
```

The historical breakdown and cost JSON files regenerate byte-for-byte with
their historical reducers. The overlay JSON agrees except for approximately
`1e-16` last-bit differences in two `polyfit` slopes. The corrected breakdown
reducer now bootstraps the seed mean; the old reducer had mislabeled the 2.5th
and 97.5th percentiles of five observed seed values as a confidence interval.

## Delta-f implementation

The current implementation, which is consistent with the metadata embedded in
the recovered outputs, is linked here:

- [`methods/delta_f.py`](../pic_kde_2d/methods/delta_f.py): two signed charge
  weights, perturbation-only deposition, exact Maxwellian/two-stream velocity
  scores, weight RHS, matched filtering, and the weight guard.
- [`core/pic_loop_delta_f.py`](../pic_kde_2d/core/pic_loop_delta_f.py): the
  nonlinear two-weight loop and the distinct production linearized loop.
- [`scripts/run_linearized_delta_f_mode.py`](../scripts/run_linearized_delta_f_mode.py):
  production marker construction and target-mode density projection.
- [`tests/test_phase5.py`](../tests/test_phase5.py): closed-form score/RHS,
  neutrality, free-streaming, matched-filter, and weight-guard tests.

For the production reference:

1. Markers are sampled from `g=f0` with equal signed equilibrium charge
   `w_full`.
2. Only `w_delta` is deposited; a finite-marker DC residual is removed at
   initialization and after midpoint/full weight updates.
3. Maxwellian and equal-bi-Maxwellian scores are analytic. The stable
   two-stream form is
   `-(vx-vbeam*tanh(vx*vbeam/vth**2))/vth**2` in x and `-vy/vth**2` in y.
4. The RHS is `-(q/m) E.grad_v(log f0) w_full`.
5. Production marker velocities are frozen; positions free-stream and an
   explicit midpoint method advances only the perturbation weights.
6. Gaussian `W` is applied once after deposition and once to each field before
   CIC gather. Protected-mode runs additionally keep only the requested real
   Fourier conjugate pair in the deposited density.

The manuscript now calls this a *mode-isolated linearized reference*. It is not
a nonlinear delta-f simulation and cannot validate cross-mode competition.

### Weight diagnostic

The implemented diagnostic is

```text
max_p |w_delta,p| / mean_p |w_full,p|.
```

Because the production markers have equal `|w_full|`, this is the largest
sampled `|delta f/g|` in charge-weight units. It is an extreme statistic: it is
not an L2 norm, changes with velocity-tail sampling and particle count, and the
threshold `0.5` is a heuristic guard. The manuscript no longer calls it proof
of delta-f validity.

With 100,000 bootstrap resamples of independent seed means:

| Alpha | Seed-mean maximum ratio [95% CI] | Guard result |
|---:|---:|---|
| 0.001 | `0.113810 [0.110355, 0.119697]` | no seed crosses 0.5 |
| 0.01 | `1.138096 [1.103547, 1.196975]` | all cross; median first crossing `t=6.1` |
| 0.1 | `11.380959 [11.035472, 11.969748]` | all cross at `t=0`; mean initial ratio `1.397808` |

Every per-seed maximum at alpha 0.01 and 0.1 is exactly 10 and 100 times its
alpha 0.001 counterpart to floating-point precision. That is expected for the
linear projected operator and is not independent nonlinear validation.

## Finite-amplitude behavior

The recovered five paired alpha-0.1 histories support one suitable nonlinear
observable: target-mode electric-field energy proxy
`|Ehat_x(1,0)|^2`. Fixed windows were used; temporal samples were averaged
within a seed and only the five paired seed indices were resampled.

| Quantity | Ratio [95% paired-seed CI] |
|---|---:|
| full-f mean energy, `t=30–40` / `t=12–18` | `4.3776 [4.2568, 4.5129]` |
| projected linearized delta-f, same ratio | `0.09159 [0.07119, 0.11337]` |
| full-f / delta-f late energy, `t=40–60` | `39.09 [22.12, 83.23]` |

The ensemble-mean amplitudes first differ persistently by more than 25% at
`t=9.5`; their late amplitude ratio is `6.7292`. Thus the raw data demonstrate
finite-amplitude divergence and a full-f energy rebound. They do **not** save
particle phase space or harmonic spectra, so trapping, phase-space vortices,
harmonic transfer, and saturation fidelity remain unresolved. The manuscript
now says “consistent with trapping” only where appropriate and does not claim
that trapping was directly measured.

## Adaptive and learned controllers

The H1 raw evidence consists of 10 paired seeds at density contrasts 10 and 100
for CIC, global C4, and a frozen initial-density C4-capped Abramson rule using
24 width bins. Reanalysis of the contrast-100 rows gives:

| Local/global C4 metric | Ratio [95% paired-seed CI] |
|---|---:|
| field MSE | `0.6343 [0.6092, 0.6592]` |
| density MISE | `1.0538 [1.0270, 1.0832]` |
| protected-mode error | `1.5784 [1.1797, 2.0821]` |

The preregistered decision is
`STOP_LEARNING_ANALYTIC_H1_NEGATIVE`. There is no learned-model artifact in the
controller tree, and the project record explicitly says that no network was
trained. The former manuscript phrase “adaptive and learned controllers were
tested” was false and has been removed. The result rejects only this frozen
density-following analytic rule on H1, not learned controllers in general.

The G0 numbers are also narrowed. A zero-fit covariance calculation predicts a
`1.6223x` packet-field-noise gain at a requested free width of `384 dx`. The
validated variational implementation caps widths at `20 dx`, where the same
calculation predicts `1.000032999x`, and the decision is
`STOP_H2_VARIATIONAL_PROFILE_NOT_IMPLEMENTED`. No dynamics was run for the
384-cell profile.

## Cost claim

The historical source snapshot times one method in a fresh process. Each
velocity-Verlet or explicit-midpoint step contains two force evaluations. Five
same-node repeats rotate method order; each uses a 64x16 grid, PPC 1024,
1,048,576 particles, 10 warm-up steps, and 200 measured steps.

The independent paired-repeat bootstrap gives:

| Method | Total-step ratio [95% CI] | Extra explicit persistent arrays |
|---|---:|---:|
| C4 fixed width | `1.002263 [1.000415, 1.004509]` | 4,608 bytes |
| linearized delta-f | `1.048477 [1.046992, 1.050080]` | 8,388,608 bytes |

C4 filter applications are included and average `1.318992 ms/step`, or
`0.31975%` of total C4 step time. The extra-memory values are sums of explicit
persistent NumPy arrays. Process peak RSS is a high-water mark and is not used
as incremental memory.

Bandwidth selection, `W` construction, and particle/grid initialization all
occur before the timed loop. Consequently, these data are not a total
end-to-end selector benchmark. Rather than invent or substitute a different
selection protocol, the manuscript now limits the claim to fixed-width
steady-state time stepping and explicitly states all exclusions. No general
speed-up or selector-efficiency claim remains.

## Code changes and tests

- Added [`audit_secondary_claims.py`](audit_secondary_claims.py), which asserts
  raw schemas and settings before computing any result.
- Corrected [`analyze_delta_f_breakdown.py`](../scripts/analyze_delta_f_breakdown.py)
  to bootstrap independent-seed means and retain the old empirical-percentile
  fields separately.
- Added guard semantics tests to [`test_phase5.py`](../tests/test_phase5.py).
- Added [`test_secondary_claims_audit.py`](../tests/test_secondary_claims_audit.py).
- Reproducible machine outputs are in
  [`results/secondary_claims_audit/`](results/secondary_claims_audit/).

Targeted tests: `26 passed`.

## Manuscript changes justified by this audit

1. Described frozen/free-streaming marker characteristics, target-mode
   projection, perturbation-only deposition, midpoint weight evolution, and the
   exact guard definition.
2. Replaced the old five-value empirical-percentile bracket with a bootstrap CI
   for the seed-mean alpha-0.001 guard.
3. Added the paired finite-amplitude target-mode energy comparison and stated
   that trapping/harmonics are not directly observed.
4. Corrected the H1 density claim (density MISE worsens; field MSE improves) and
   removed the claim that a learned controller was tested.
5. Relabeled the 384-cell statement as an analytic, unimplemented headroom
   result.
6. Limited the cost table and conclusion to fixed-width steady-state stepping,
   disclosed that selection/construction/initialization are excluded, and
   stated exactly what the memory numbers count.

## Unresolved items

- Exact source revision for jobs 4873222 and 4873285: unknown.
- Nonlinear particle phase space, harmonic spectra, trapping and saturation
  fidelity: not recorded and unresolved.
- End-to-end bandwidth-selection runtime and memory: not measured; the claim
  was removed rather than filled with a theoretical or different-protocol
  estimate.
- Learned-controller performance: untested because the preregistered stop rule
  prevented training.
- H2 moving 384-cell width profile, controller refresh work, and complete
  controller-response grad-h term: unimplemented and unmeasured.
