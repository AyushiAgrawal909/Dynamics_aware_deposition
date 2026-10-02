# Dynamics-aware deposition for 2D electrostatic PIC

This is a general research code for two-dimensional electrostatic
particle-in-cell experiments with CIC deposition, Gaussian and binomial
filtering, compensated transfer functions, bandwidth selection, full-`f` and
mode-projected linearized `delta-f` evolution, and conservation/transfer
diagnostics.

The software is independent of any manuscript. The repository also contains a
frozen `examples/paper_results/` layer with the configurations, archived
seed-level histories, and reducers needed to reproduce the reported numerical
results. It contains no manuscript, LaTeX source, bibliography, or paper-build
workflow.

This is the lean distribution. Superseded reducer revisions, duplicate rebuild
trees, invalid-development raw runs, ordinary per-run console logs, caches, and
other regenerable intermediates are intentionally omitted. The retained raw
histories and final numerical anchors are sufficient for the maintained
commands below; `python3 reproduce.py all` rebuilds the omitted derived output.

## Install the general code

Python 3.12 is the verified environment. Python 3.10 or newer is declared by
the package.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-simulation.txt
python3 -m pip install -e ./pic_kde_2d
```

Run the general implementation tests after installation:

```bash
python3 -m pytest pic_kde_2d/tests
```

The recorded result is **121 passed and 2 strict expected failures**. The two
expected failures preserve the original tolerances of legacy low-resolution
Landau smoke tests; details are in `VERIFICATION_REPORT.md`.

The reusable implementation is under `pic_kde_2d/pic_kde_2d/`:

- `core/`: grids, particles, full-`f` and linearized `delta-f` PIC loops;
- `methods/`: CIC, Gaussian, spectral, compensated and adaptive filtering,
  bandwidth selection, and delta-`f` weights;
- `benchmarks/`: Landau and two-stream initializations and theory helpers;
- `validation/`: transfer, conservation, self-force, momentum, and rate-fit
  diagnostics.

Research-grade command-line runners and Slurm examples are in
`pic_kde_2d/scripts/`. They are examples, not a stable public API.

## Reproduce the reported numerical results

For analysis of the included histories, the smaller dependency set is enough:

```bash
python3 -m pip install -r requirements-analysis.txt
python3 reproduce.py verify
python3 reproduce.py tests
```

Run every maintained existing-data reproduction workflow with one command:

```bash
python3 reproduce.py all
```

This runs the 87 scientific regressions, the bounded Landau/E2 analysis, the
research-completion reductions, and the exact-width q050 reduction. It does
not submit simulations. New products go under `reproduced/` and archived data
are never overwritten.

Individual workflows are also available:

```bash
python3 reproduce.py bounded
python3 reproduce.py completion
python3 reproduce.py exact-width
```

See `RESULT_REPRODUCTION_MAP.md` for the code, input data, output tables, and
scientific scope of each workflow.

The exact-width replay should recover:

- mean-individual-rate ratio `0.9556447297592618`;
- paired 95% interval `[0.9452733092216409, 0.9640059783937591]`;
- ensemble-amplitude ratio `0.9706344447819388`;
- model ratio `0.9513008502587932`.

## Repository layout

- `pic_kde_2d/`: general current research implementation and its tests;
- `examples/paper_results/`: paper-specific reducers, frozen configurations,
  regression tests, scientific records, required raw data, and compact final
  result anchors;
- `campaigns/`: self-contained completed campaign snapshots;
- `pic_kde_2d_landau_cv_stage/`: historical implementation snapshot retained
  where exact reproduction requires it;
- `reproduce.py`: safe top-level existing-data reproduction entry point;
- `PROVENANCE.md`: origin, frozen hashes, qualifications, and known limits;
- `SIMULATION_RERUN_GUIDE.md`: requirements for genuine simulation reruns.

## Integrity files

`SHA256SUMS.txt` and `FILE_INVENTORY.csv` are optional integrity aids. They do
not calculate scientific results. If deleted, simulations, analyses, and tests
still run; `verify` checks the required structure and core frozen hashes while
skipping full-file verification.

Maintainers can regenerate them after an intentional repository update:

```bash
python3 tools/build_checksums.py
```

## Scientific qualifications

The repository preserves negative and unresolved outcomes. In particular, the
`v_b=0.818` extension remains signal/linearity-unresolved, and the weak-Landau
joint confirmation fails because the CIC arm lacks the required fitted
duration. Regression success means the code reproduces the archived evidence;
it does not change those verdicts.

Before publishing publicly, resolve the project-wide license and data
redistribution questions described in `LICENSE_NOTICE.md` and
`GITHUB_UPLOAD_GUIDE.md`. This preparation did not initialize Git or upload
anything.
