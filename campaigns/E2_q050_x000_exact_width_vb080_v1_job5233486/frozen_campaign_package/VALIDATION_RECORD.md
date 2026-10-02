# Package validation record

Completed without running or submitting a production simulation:

- Python syntax compilation passed for the runner, reducer, preflight and
  reconciliation scripts.
- Shell syntax validation passed for the MeluXina batch file.
- Source hashes match the frozen audited research implementation.
- The Gaussian transfer was recomputed as `W=0.9977310235762052` and
  `W^2=0.9954671954064223` at `(12,1)`.
- Read-only construction verified identical CIC/q050 arrays within each tested
  runtime for every frozen seed `200--209`. The authoritative hashes were
  generated on MeluXina with Python 3.11.10/NumPy 2.3.5, not on the Mac.
- A Mac search found no exact target-mode CIC histories for this configuration;
  it found static q050 selector records and incompatible `(10,1)` histories.
- A MeluXina search recovered ten exact CIC histories from job 5201231 for
  seeds `200--209`; no exact q050 dynamics histories were found among 1,709
  parsed result metadata files. Exact q050 occurrences were confined to the
  ten static selector records from job 5197174.
- The recovered CIC histories each contain 1,780 samples from `0` through
  `71.16` at cadence `0.04`. Their saved configurations match mode `(12,1)`,
  PPC 512, `dt=0.02`, `alpha=1e-8`, zero width and target-mode projection.
- Job 5201231 did not save its particle arrays or array hashes. The archived
  wrapper, configuration, explicitly hashed initializer sources and seeds were
  reconstructed under the production runtime. Reconstructed and archived
  time-zero modes agree exactly there. The q050 runner now refuses to integrate
  unless its aggregate hash matches the production-runtime reconstruction; the
  reducer also compares all six actual q050 array hashes separately.
- The historical input manifest did not hash `conventions.py` or `cic.py`.
  Their observed mtimes predate the job, but that is supporting rather than
  cryptographic evidence. The recorded git revision cannot supply the missing
  hashes because the research Python tree was untracked. These limitations are
  retained in `reuse_audit.json` and `runtime_provenance.json`.
- Every recovered CIC history passes separately: complete history, individual
  `R^2 >= 0.98`, delta-f ratio below 0.5 and positive fitted rate. Applying the
  same thresholds diagnostically to each seed also gives SNR above 5 and more
  than two fitted e-foldings. Individual durations span `4.2077--7.3459`.
  `6.4904` is specifically the ensemble-amplitude fitted duration.
- The original fresh `105--114` design changed to `200--209` before any q050
  result existed. The final seeds are independent of bandwidth selection but
  were previously analysed in job 5201231.
- The archived confidence interval was reproduced from its original RNG stream.
- Exported table counts: 850 seed-candidate objective rows, 108,120 individual
  complex-history rows, three phase-decomposition rows and 30 joined
  finite-marker/loading rows.
- The Landau bootstrap archive contains 5,000 draws for every saved rate,
  ratio and accounting quantity.
- Both exported vector PDFs contain one unencrypted page, have no JavaScript or
  forms, rendered successfully, and were visually checked for clipping,
  overlap and unreadable glyphs.

The exact-width production reducer cannot yet be exercised end to end because
the ten q050 histories do not exist. The CIC half is present and audited. The
runner/reducer pair is source-locked, non-overwriting and fails on missing
files, configuration mismatches or reconstructed initial-state mismatches.
