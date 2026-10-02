# Missing or unrecoverable information

Audit date: 2026-09-12.

These items remain unknown.  No value should be inferred for them in figures,
tables, or responses to review.

1. Exact Git commit and worktree diff for every Python Paper 1 campaign.  The
   Python research package was not committed in the supplied repository, and
   the outputs do not store source hashes.
2. Exact C++ checkout, executable checksum, compile command and complete linked
   library manifest for the historical grid-phase/conservation jobs `4855533`,
   `4858774` and `4858823`.  Their input decks and one-rank launch layouts were
   recovered, but the jobs did not embed source/executable hashes.  New job
   `5195488` records these items for the current matching implementation only.
3. Exact dependency versions for NumPy, Numba, SciPy, Matplotlib and the FFT
   backend used by historical Python campaigns.
4. Historical batch-script contents for completed jobs whose Slurm controller
   records have expired, including Gaussian noise jobs `4867515` and
   `4868286`.  `sacct` retains the submission line but `scontrol write
   batch_script` returns `Invalid job id specified`.  The archived `4868286`
   result records global loading; the per-cell assignment for `4867515` is
   supported by the contemporaneous audit record, not an embedded run field.
5. Individual job IDs for the copied `compensated_binomial_20260811`,
   `higher_order_compensation_20260811`, and
   `strong_landau_alpha025_20260811` campaign components.
6. Originating job ID, launcher, and complete run metadata for the
   `weak_landau_alias` erasure traces.
7. Full raw trajectories for several locally copied analysis-only directories
   outside the Task-6 through Task-8 recovery set.  The six Task-8 Gaussian,
   raw-binomial and LOO trees were copied byte-for-byte from MeluXina and
   hash-compared; other campaigns remain on MeluXina where available and must
   be hashed before any archival move.
8. Whether every current matching entry point is byte-identical to the file
   used at job submission.  Interface and schema matches are evidence of
   correspondence, not proof of identity.
9. Supplied-baseline build configuration.  The repository commit contains
   makefiles, but no immutable build manifest from the reported runs was found.
10. A Git commit or immutable source archive for the recovered
    `pic_kde_2d_landau_cv_stage/` campaign copy.  Its implementation and launch
    scripts match the binomial/compensated result schemas, but the copied files
    do not prove the source revision used by the historical jobs.
11. ~~Completed direct compensated-filter rate measurements at the E2 binding
    corner `(12,1)`.~~ Resolved by array `5195640` and reducer `5195641`, both
    exit 0.  Direct p2/n2 and p3/n4 corner ratios are
    `0.95608 [0.95120,0.95920]` and `0.97342 [0.97047,0.97529]`; both pass
    rate, duration and signal gates.  The rate job did not remeasure noise, so
    its rate results and the separate filter-family noise results must remain
    distinguishable.
12. An evolving HEXAPIC energy history and a time-step convergence comparison
    against the matched spectral chain.  The recovered 1500-step HYPRE audit
    writes energy at step zero only, so no “both solvers” energy conclusion is
    supported.
13. A benchmark-scale normalization for the long-run HEXAPIC momentum drift.
    The audit records SI momentum components and force-normalized sums, but no
    agreed dimensionless particle-momentum or physical-impulse scale with
    which to establish dynamical irrelevance.
14. A grad-h validation for a live translation-equivariant bandwidth selector
    in both spatial directions.  The available test uses a prescribed smooth
    `h_x(x)` and constant `h_y`; its 2D drift norm is dominated by y.
15. ~~Raw per-seed histories for the older isotropic E2 Gaussian mode-12 and
    linearized-delta-f reference/band reductions.~~ Resolved on 2026-09-11 by
    the job-ID paths under `/mnt/tier2/users/u103576/logs`: 30 histories from
    `4872599`, 40 nonzero-width histories plus the linked ten-seed baseline from
    `4872745`, and all 140 histories from `4873221`. Local/remote aggregate
    hashes and reducer comparisons are in
    `results/recovered_delta_f_campaigns_20260911/recovery_manifest.json`.
    The immutable source revisions used by those jobs remain unknown.
16. The two helper sources used by the historical preserved-stage E2 reducer.
    The stage copy imports `analyze_E2_mode_growth.py` and
    `analyze_linearized_delta_f_m12_sigma.py`, but does not contain them; the
    Task-6 rebuild used current matching files from the top-level research tree.
17. A fit-quality/rejection rule for filtered arms in the compensated E2
    bootstrap.  The recovered analyzer rejects non-finite draws and a
    non-positive CIC rate only; it does not test each filtered trace's duration
    or R-squared.
18. A strong-Landau run with at least two model e-foldings before trapping.
    The alpha-0.5 CV-width fits contain 1.242 e-foldings from `t=3` or 0.656
    from `t=5`; alpha 0.25 contains 0.964.  Their conditional rate CIs are
    narrow, but the validation duration gate fails.
19. A weak-Landau confirmation in which both C4 and its paired CIC baseline
    provide at least two fitted amplitude e-foldings.  The fresh fixed 40-seed
    global-loading campaign gives C4/CIC `0.98942 [0.96788,1.01383]` and 2.076
    C4 e-foldings, but CIC ends at its frozen floor guard with only 1.721.
    Earlier PPC-1024/PPC-4096 C4 histories span only `1.491/1.155`; the
    PPC-map's larger “resolved e-foldings” are signal-to-floor capacities, not
    fitted durations.  The secondary common-window fit gives
    `1.018773 [1.015359,1.021931]` but only `1.721/1.754` e-foldings, so it
    does not resolve this condition. More seeds were not added after the fixed
    campaign.
20. ~~A completed dynamics rate at the exact weak-Landau LOO-CV width
    `7.244293147028428 dx`.~~ Resolved by paired jobs `5195682/83`: ratio
    `1.82222 [1.48934,2.21398]`, `1.62678` fitted e-foldings, floor-censored
    with zero quality-passing bootstrap fraction.  The previous `1.690` value
    remains correctly attached to the distinct sigma-`6.61` proxy.
21. A production campaign implementing the configured per-direction
    Silverman pairs exactly.  The weak-Landau scripts apply the longitudinal
    cell width on both axes.  The supplied E2 noise script also applies one
    scalar in cell units, whereas the supplied E2 m=8 rate runner applies one
    equal physical width on both axes.  Neither E2 path implements the frozen
    configuration `(5.81933 dx,0.727416 dy)`.  No measured rate is therefore
    attributed to that directional pair.
22. Continuous or filter-family-global anisotropic optimality.  New jobs
    `5197174/84` establish only that the transverse endpoint is best among 85
    predeclared admissible diagonal-Gaussian points for density and pre-gather
    grid-field squared error.  Particle-force optimality was not evaluated.
23. Equal end-to-end computational budgets for the model-based C4 rule and the
    data-driven selectors.  The held-out audit gives MISE and LOO-CV identical
    22-width, five-training-seed, 51-snapshot budgets; C4 and Silverman are
    fixed rules and their theory/selection costs are structurally different.
24. Exact source revision for delta-f jobs `4873222` and `4873285`.  Their raw
    histories and launch settings were recovered, but neither output records a
    commit or source hash and the historical result schema predates the current
    runner schema.
25. Particle phase-space snapshots and harmonic spectra for the recovered
    finite-amplitude Landau campaigns.  The target-mode energy rebound is
    measurable, but trapping, phase-space vortices, mode coupling and
    saturation fidelity cannot be reconstructed from one saved Fourier mode.
26. End-to-end runtime and memory for bandwidth selection.  Job `5049224`
    measures steady-state stepping after construction and includes the live
    filter applications, but excludes selection, `W` construction and
    particle/grid initialization.  The manuscript efficiency claim has been
    narrowed accordingly.
27. Learned-controller performance for H1.  The analytic rung failed its
    preregistered gate and recorded `STOP_LEARNING_ANALYTIC_H1_NEGATIVE`, so no
    network was trained and no learned-model result exists.
28. A dynamics implementation of the G0/H2 moving profile with a free-region
    width of 384 cells.  The validated variational code caps the width at 20
    cells; the 384-cell result is analytic headroom only and was stopped before
    dynamics, refresh-work or controller-response-force measurements.
29. Author-approved CRediT roles.  Contributions cannot be inferred reliably
    from source ownership, job submitter names, or manuscript author order, so
    no contribution statement was invented.
30. A public immutable data/code archive identifier and its final contents.
    The local audited project and hashes exist, but no DOI or approved public
    repository record was found.  The manuscript therefore still needs an
    author-approved data-availability statement before submission.
31. ~~Files for the preregistered `v_b=0.818` E2 extension.~~ Resolved by
    separately frozen job `5202227`, which completed all 30 histories with
    exit code `0:0` in `02:21:40` under the unchanged eight-hour cap. The
    original job `5201231` remains a separately preserved 120/150 timeout.
    The physical near-marginal rate remains unresolved, rather than missing:
    every extension arm fails the frozen fit-start-SNR rule and three of ten
    seeds per arm exceed the linear-`delta f` guard. Descriptive estimator
    outputs must not be reported as accepted growth or stabilisation.
32. The exact source revision for historical E2 corner job `4913408`. Its
    recovered outputs record the relevant numerical settings but no revision.
    Therefore the remaining shared-endpoint difference from current job
    `5201231` cannot uniquely separate different seed sets from historical
    implementation history.
