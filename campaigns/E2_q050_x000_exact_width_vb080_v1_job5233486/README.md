# E2 q050 exact-width validation campaign

This directory is the complete archived evidence for MeluXina job 5233486.
It combines the immutable pre-submission package with the downloaded run root,
all twenty complex mode histories used by the reducer, bootstrap draws,
scheduler logs, configuration, source snapshot, and result documentation.

The only new simulations were the ten q050 histories for seeds 200-209. CIC
histories from job 5201231 were reused only after the production-runtime
reconstruction audit. The historical CIC particle arrays were not saved, so
the provenance qualification in `RESULTS_REPORT.md` remains mandatory.

From the top-level reproduction package, rebuild the frozen reducer with:

```bash
python3 reproduce.py exact-width
```

That command writes to a new `reproduced/exact_width_*` directory, regenerates
the summary, per-seed table, bootstrap draws and PDF/SVG/PNG figure, and then
compares the scientific outputs and all reducer-input hashes with this archive.
It never submits a simulation or overwrites this directory.

Key locations:

- `raw_remote_output/runs/`: CIC and q050 histories for seeds 200-209.
- `raw_remote_output/analysis/`: frozen production reducer output.
- `frozen_campaign_package/`: submitted configuration, code, source snapshot,
  preflight records and scheduler.
- `scheduler_logs/`: complete job stdout and empty stderr.
- `RESULTS_REPORT.md`: result, interpretation and provenance qualification.
- `RUN_ACCOUNTING.json`: Slurm allocation and cost.
- `verify_reduction.py`: replay comparison used by the top-level command.
