# Repository provenance and limits

## Scope

- Repository: `dynamics_aware_deposition`
- Assembly/update date: 2 October 2026
- Source workspace: `/Users/ayushi.agrawal/Downloads`
- New simulations launched during packaging: none
- Retained scientific arrays modified: none
- Manuscripts, LaTeX sources, bibliographies, journal classes, paper-build
  scripts, and editorial/reviewer records: deliberately excluded

This is a curated code-and-data derivative of
`Paper1_Reproducibility_Package_20260930`. The numerical arrays and research
sources were copied without changing their scientific content. Two regression
tests that formerly inspected manuscript wording were changed to inspect the
underlying machine-readable scientific definitions and numerical anchors; the
scientific assertions are retained without requiring a manuscript tree.

Unrelated Paper 2 and Paper 3 research directories were also excluded.

## Lean-distribution curation

The public-ready copy was pruned after dependency tracing and end-to-end
reproduction tests. Removed material consisted of superseded `_rN` analysis
directories, three duplicate research-completion rebuild trees, the rejected
development Landau campaign with invalid longitudinal loading, pilot raw data,
large regenerable seed-contribution CSVs, ordinary run logs, and runtime
caches. The accepted weak-Landau histories, E2 histories, exact-width campaign,
frozen configurations, reducers, final tables, negative verdicts, and
regression anchors remain.

Twelve small conservation logs were retained because the conservation reducer
parses their configured HYPRE tolerances and grad-`h` diagnostic components.
The pruning changed neither retained numerical arrays nor scientific
acceptance criteria. A full pre-curation archive was kept outside this
distribution by the repository owner.

## Included research trees

- `examples/paper_results`: frozen article reducers, tests, records, and data
- `pic_kde_2d`: research implementation and recovered raw histories
- `pic_kde_2d_landau_cv_stage`: campaign implementation snapshot
- `campaigns/E2_q050_x000_exact_width_vb080_v1_job5233486`: completed frozen
  exact-width campaign

Disposable caches and operating-system metadata were excluded. Optional
`FILE_INVENTORY.csv` and `SHA256SUMS.txt` identify every distributed file;
they are integrity aids rather than scientific inputs.

## Exact-width source provenance

The completed job-5233486 material was recovered from:

- staged package:
  `/home/users/u103576/hexapic-main/pic_kde_2d/reproducibility/exact_width_q050_validation_20260921`;
- output:
  `/mnt/tier2/users/u103576/paper1_validation/e2_q050_exact_5233486`;
- scheduler logs:
  `/mnt/tier2/users/u103576/logs/p1_q050_exact_5233486.{out,err}`.

Frozen configuration SHA-256:
`3b763155abe28c4b105a1824969ff411a6d0b1b6a5f3d9319b23ff3c39ec4800`.
Production summary SHA-256:
`70a1a6ed7ed392a7bbfc962f6807b18aa9070602e74bf9a05129bd9bb9742b26`.

## Evidence hierarchy

Use evidence in this order:

1. seed-level arrays and embedded run metadata;
2. frozen configuration, source, and scheduler records;
3. deterministic reducers and machine-readable outputs;
4. generated scientific figures;
5. analytic/illustrative curves only where explicitly labelled.

## Known limits

- None of the copied source trees was a Git worktree, so no historical commit
  hash can honestly be assigned. Revisions are identified by file hashes and
  campaign records.
- Historical source revisions and software versions that were not recorded
  remain unknown.
- Some old Landau data lack sufficient provenance to infer loading from a
  current initializer; their classifications are retained in the audit data.
- Job 5233486's CIC initialization comparison uses deterministic
  reconstruction because saved historical CIC initialization arrays were not
  available. It is not a direct hash comparison to saved historical arrays.
- A different compiler, BLAS/FFT/MPI stack, or hardware may not produce
  bitwise-identical trajectories even when the scientific protocol is fixed.

See `examples/paper_results/MISSING_INFORMATION.md` for the
explicit unknowns.
