# E2 q050 exact-width validation result

## Outcome

MeluXina job 5233486 completed with exit code 0 after 00:32:26. It ran only
the ten frozen q050_x000 histories for seeds 200-209 and reused the audited CIC
histories from job 5201231. The elapsed allocation was 0.5406 CPU node-hours,
below the preregistered 1.0 CPU-node-hour cap. No seed was added, discarded,
rerun, or automatically resubmitted. The fit window remained [24.32, 69.16].

The frozen reducer returned `COMPLETE`. All registered arm-level and per-history
quality gates passed. The primary ratio of the arithmetic means of the paired
individual-seed fitted rates is 0.95564473, with paired 95% bootstrap interval
[0.94527331, 0.96400598]. The complete interval lies inside [0.90, 1.10], so
rate preservation is established under the frozen primary rule.

The secondary ratio of mean-magnitude ensemble slopes is 0.97063444, with
paired 95% bootstrap interval [0.95535636, 0.97370726]. Its interval also lies
inside [0.90, 1.10], so the secondary observable separately passes.

The initialization-only numerical model predicted 0.95130085. The measured
primary ratio is higher by 0.00434388, or 0.4566% relative to the prediction.

## Quality and initialization evidence

The archived CIC arm has ensemble mean-magnitude gamma = 0.14474590,
R-squared = 0.999411, 6.4904 fitted e-foldings, and fit-start SNR = 22.249.
The new q050 arm has gamma = 0.14049536, R-squared = 0.999245, 6.2998 fitted
e-foldings, and fit-start SNR = 22.392. The mean individual rates are
0.12445081 for CIC and 0.11893076 for q050.

All twenty individual fits have R-squared above 0.98, positive fitted growth,
more than two fitted e-foldings, and maximum delta-f ratio below 0.5. Every new
q050 history matched its corresponding CIC production-runtime reconstruction
for the aggregate initial-state hash and for all six saved arrays: x, y, vx,
vy, w_full, and w_delta. Single-seed SNR is reported as a diagnostic. Seed 201
q050 has diagnostic start SNR 4.740, below 5; the registered SNR gate is the
arm-level ensemble value and passes. This flag is retained without excluding
the seed or changing the verdict.

## Historical provenance qualification

Job 5201231 did not save the historical particle arrays or their hashes.
Pairing therefore does not rest on matching seed numbers. The reuse audit
deterministically reconstructed each CIC state on MeluXina with the production
Python 3.11.10 / NumPy 2.3.5 runtime, the archived wrapper, frozen settings,
explicitly hashed initializer sources, and the original seed. Each q050 runner
had to reproduce the reconstructed aggregate state hash before integration;
the reducer then required equality of all six per-array hashes.

This is strong reconstruction evidence, but it is not a direct comparison with
historically saved arrays. The historical manifest did not hash transitive
`conventions.py` and `cic.py`; their observed mtimes predate job 5201231, which
is supporting but non-cryptographic evidence. The research Python tree was not
tracked by the recorded git revision. These limitations remain part of the
result's provenance statement.

Seeds 200-209 are disjoint from bandwidth-selection seeds 100-104 and held-out
static-error seeds 105-109, but they were analysed previously in job 5201231;
they are not fresh dynamics seeds. The result validates the exact selected
width in target-mode-projected linearized delta-f dynamics. It is not a new
same-configuration full-f error-and-rate demonstration.

## Artifact map

- `raw_remote_output/`: immutable copied run root, including all 20 complex
  histories, frozen reducer output, reuse audit, and input hashes.
- `frozen_campaign_package/`: exact submitted configuration, scheduler,
  wrappers, reducer, research-source snapshot, plan, and preflight evidence.
- `scheduler_logs/`: complete Slurm stdout and stderr.
- `tables/compact_results.md`: compact result and verdict table.
- `tables/per_seed_quality_checks.md`: all per-seed fits, durations, SNR
  diagnostics, linearity, validity, and initialization checks.
- `tables/per_seed_quality_checks.json`: machine-readable form of the same
  verification.
- `figures/fig_q050_exact_width_final.{pdf,svg,png}`: presentation figure made
  only from frozen histories and reducer results.
- `postprocess/finalize_results.py`: reproducible verification and figure code.

No manuscript file was read or edited during submission or reduction.
