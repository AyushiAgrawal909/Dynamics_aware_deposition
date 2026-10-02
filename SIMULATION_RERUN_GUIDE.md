# Simulation rerun guide

The archived data are sufficient for the recommended existing-data
reproduction. Use this guide only when a genuine simulation rerun is required.

## Research implementations

`pic_kde_2d` is the current Python research tree. The recovered
`pic_kde_2d_landau_cv_stage` snapshot is retained separately because several
audited transfer, E2, and Landau-CV results were generated with that campaign
implementation. Do not replace it with the baseline HEXAPIC archive or with a
newer initializer when claiming an exact historical rerun.

The production particle-field coupling audit source is under
`pic_kde_2d/hexapic_momentum_audit/source/`. Its README records the opt-in HYPRE
audit controls. Cluster launchers and frozen configurations are distributed in:

- `examples/paper_results/meluxina/`
- `examples/paper_results/*.sbatch`
- `pic_kde_2d/scripts/*.sbatch`
- `pic_kde_2d/paper1_selector_comparison/slurm/`
- `campaigns/E2_q050_x000_exact_width_vb080_v1_job5233486/frozen_campaign_package/meluxina/`

The exact-width campaign directory contains the production configuration,
runner, reducer, source snapshot and completed outputs together. Use
`python3 reproduce.py exact-width` for an existing-data replay. Do not submit
its Slurm file merely to verify the published reduction.

## Before submission

1. Identify the campaign in
   `examples/paper_results/REPRODUCIBILITY_MANIFEST.md` and
   `CAMPAIGN_MANIFEST.json`.
2. Confirm the required implementation snapshot and source-file hashes.
3. Copy the corresponding frozen JSON/input and Slurm launcher; do not recreate
   field names from memory.
4. Record the new compiler, Python, package, module, FFT/BLAS, MPI, HYPRE,
   hardware, and scheduler versions.
5. Change only site plumbing such as account, partition, module path, and output
   root. Record those changes in a rerun manifest.
6. Keep seeds, particle loading, geometry, grid, time step, duration,
   diagnostics, solver, filters, estimators, windows, and stopping rules fixed.
7. Write to a new output directory. Never overwrite the archived histories.

## Scope warnings

The Slurm files are executable records, not portable one-click submissions.
They contain original MeluXina paths and allocation details. A friend using a
different cluster must review them rather than submitting blindly.

No rerun should be described as resolving an archived failed or unresolved
case unless it was separately specified and analysed under an explicitly
declared protocol. The package intentionally preserves negative and unresolved
outcomes.
