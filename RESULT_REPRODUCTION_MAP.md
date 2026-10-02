# Result reproduction map

The reusable solver lives in `pic_kde_2d/`; the entries below are frozen
applications of that general code. All top-level commands read archived data
and write to a new output directory. They do not submit jobs.

| Result family | Top-level command | Main reducer | Principal archived inputs | Generated products |
|---|---|---|---|---|
| Landau floors, E2 estimands and finite-reference correction | `python3 reproduce.py bounded` | `examples/paper_results/analyze_bounded_floor_estimand_reference_20260914.py` | saved PPC-1024/PPC-4096 histories and E2 seed tables under `examples/paper_results/results/` and `pic_kde_2d/results_raw_rate_audit/` | CSV/JSON tables and vector/raster diagnostic figures |
| Transfer/operator checks, beam-drift scan, finite-marker estimator diagnostics, interior-width error and extension audit | `python3 reproduce.py completion` | `examples/paper_results/rebuild_research_completion.py` | frozen scan configuration plus recovered MeluXina histories under `pic_kde_2d/results_raw_rate_audit/` | 35 checked products organised by analysis stage |
| Exact selected-width E2 validation | `python3 reproduce.py exact-width` | `campaigns/E2_q050_x000_exact_width_vb080_v1_job5233486/frozen_campaign_package/code/analyze_E2_q050_exact.py` | ten q050 and ten audited historical CIC complex histories | per-seed table, paired bootstrap, summary and validation figure |
| Archived numerical anchors and scientific verdicts | `python3 reproduce.py tests` | 87 tests in `examples/paper_results/tests/` | machine-readable archived results and raw histories | pass/fail regression report |
| All maintained result workflows | `python3 reproduce.py all` | top-level orchestration | all inputs above | one timestamped result tree and `REPRODUCTION_COMPLETE.json` |

## General-code tests

The article regressions above are distinct from implementation tests for the
general solver:

```bash
python3 -m pytest pic_kde_2d/tests
```

## Full simulation reruns

The `.sbatch` files are historical executable records. Read
`SIMULATION_RERUN_GUIDE.md` before adapting them. Changing a cluster account,
partition, module path or output root is site plumbing; changing seeds,
loading, grids, time step, filters, windows or stopping rules creates a new
experiment rather than an exact protocol rerun.
