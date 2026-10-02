# Paper 1 exact-width validation package

## Exact-width plan and cost

No job has been submitted. The prepared campaign is deliberately limited to
the one unresolved selected-width dynamics question.

| Item | Frozen value |
|---|---|
| Selected candidate | `q050_x000` |
| Exact width | `sigma_x/dx = 0`, `sigma_y/dy = 0.1716392815470568` |
| Model prediction | `gamma_filtered/gamma_CIC = 0.9513008502587932` |
| Arms | archived CIC and exact `q050_x000`, paired by reconstructed initial state |
| Evaluation seeds | `200--209`; disjoint from selection seeds `100--104`, but previously analysed in job 5201231 |
| Histories | 20 analysed: 10 archived CIC plus 10 new q050 histories |
| Fit window | fixed `[24.32, 69.16]`; run ends at `71.16` |
| Primary comparison | ratio of arithmetic means of paired individual-seed rates |
| Secondary comparison | ratio of slopes of `ln(mean_seed(abs(Ehat_12,1)))` |
| Model operating gate | predicted ratio at least `0.95` |
| Measured acceptance | entire paired 95% interval inside `[0.90, 1.10]` |
| Expected new cost | about `0.60` CPU node-hour |
| Hard cap | one CPU node for `01:00:00`, or `1.0` node-hour |

The ten-seed size is fixed from the completed directional-Gaussian campaign:
its paired ratio bootstrap had standard deviation `0.00398` and interval width
`0.01535`. Ten pairs are adequate to distinguish a result near `0.95` from the
`0.90` acceptance boundary at the observed uncertainty scale. The campaign must
remain unresolved if its frozen interval crosses `0.90`; seeds or duration must
not be added after inspecting the result.

The exact missing CIC histories were recovered from MeluXina job 5201231 for
seeds `200--209`. Each contains 1,780 mode samples on `t=0--71.16`, with the
required mode, grid, PPC, loading arguments, time step and target-mode
projection. The old campaign did not save the particle arrays or their hashes.
There is therefore no historical byte hash to compare with the new arm.

Instead, the archived wrapper, frozen arguments, explicitly hashed initializer
files and seeds are rerun without time integration under the same MeluXina
Python 3.11.10/NumPy 2.3.5 runtime. This produces a production-runtime
reconstructed hash for all six arrays. Before any q050 integration, the runner
must reproduce the aggregate hash for its seed. After the run, the reducer must
match the actual q050 hashes of `x`, `y`, `vx`, `vy`, `w_full` and `w_delta`
separately to the reuse audit. Matching seed numbers alone is never accepted.

This is adequate for conditional deterministic reuse, but it is not a direct
comparison to archived particle bytes. The historical manifest omitted
`conventions.py` and `cic.py`; their observed mtimes predate job 5201231, but
that is not cryptographic provenance. The recorded git revision is contextual
only because the research Python tree was untracked. The time-zero mode agrees
exactly in the production runtime, while the seed-specific velocity hashes are
fixed by the explicitly hashed initializer. These limits remain explicit in
the audit.

The recovered CIC ensemble-amplitude fit has `6.4904` e-foldings. This is not
an individual-seed duration. Individual fitted durations range from `4.2077`
to `7.3459` e-foldings; all ten histories, individual `R^2` checks, delta-f
checks and positive-rate diagnostics pass separately. The full per-seed table
is in `evidence/existing_cic_vb0800_5201231/CIC_QUALITY_GATES.md`.

## What this campaign can establish

The archived selector independently chose q050_x000 on seeds `100--104`; its
static error was evaluated on seeds `105--109`. Dynamics seeds `200--209` are
independent of both selection/evaluation sets, but they have already been
analysed in job 5201231 and are not fresh dynamics seeds. A successful result
would therefore validate the selected width across compatible experiments.

It would not be a same-configuration demonstration that statistical error falls
and rate preservation holds simultaneously. The selector experiment is an
offline full-grid, full-f PPC-32-versus-PPC-1024 error comparison, whereas the
dynamics experiment is a PPC-512, target-mode-projected, linearized delta-f
test. The exact compatibility audit is in
[`EXACT_WIDTH_VALIDATION_PLAN.md`](EXACT_WIDTH_VALIDATION_PLAN.md).

## Frozen quality rules

Both arms must satisfy, without seed rejection:

- fit-start signal-to-floor ratio at least `5`;
- ensemble log-amplitude fit `R^2 >= 0.98`;
- every individual-seed log-amplitude fit `R^2 >= 0.98`;
- at least two fitted amplitude e-foldings;
- maximum delta-f weight ratio below `0.5`;
- positive central growth and positive paired-bootstrap lower limit.

The floor is the quietest rolling median of the arithmetic mean of seed
magnitudes over a complete two-time-unit pre-fit window. The fixed fit window is
not changed after looking at the histories. Primary and secondary rate
estimands are reported separately. An interval crossing `0.90` leaves
preservation unresolved; it does not by itself prove distortion greater than
ten percent.

## Package map

- `config/`: the full-precision frozen campaign.
- `FINAL_CAMPAIGN_MANIFEST.json`: final seeds, costs, rules and synchronized
  SHA-256 identities for the runnable campaign files.
- `code/run_E2_q050_exact_arm.py`: source-locked runner and initial-state hash.
- `code/audit_existing_cic_reuse.py`: validates the recovered histories and
  reconstructs the archived CIC initial states.
- `code/analyze_E2_q050_exact.py`: paired reduction, quality gates and figures.
- `code/preflight_exact_width.py`: read-only source/transfer/pairing check.
- `code/verify_initial_state_compatibility.py`: reconstructs and hashes states
  under a specified runtime without integrating them.
- `code/validate_final_package.py`: asserts that seeds, cost, hashes, scheduler
  limits, audit and frozen rules agree across the final artifacts.
- `meluxina/`: reviewed CPU batch file; prepared but not submitted.
- `research_source/`: the exact audited Python source needed by the prepared
  runner, copied with the recorded file hashes.
- `evidence/data/`: per-seed candidate objectives, three-arm complex histories,
  phase decomposition, bootstrap draws and joined finite-marker/loading data.
- `evidence/figures/`: PDF and SVG vector figures from the existing-data work.
- `evidence/reconciled/`: corrected interpretation metadata and the resolved
  density-confidence-bound discrepancy.
- `provenance/`: original retrieval report, commands and source hashes.
- `evidence/existing_cic_vb0800_5201231/`: the recovered ten-history CIC set,
  per-file hashes and deterministic-reconstruction audit.
- `provenance/SEED_SET_CHANGE_RECORD.md`: why the fresh `105--114` design was
  changed to the previously analysed `200--209` set before q050 results existed.

See [`RUN_COMMANDS.md`](RUN_COMMANDS.md) for preflight, deployment and the
explicit not-yet-authorized submission command.
