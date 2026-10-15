# Dynamics-aware deposition for 2D electrostatic PIC

This is a general research code for two-dimensional electrostatic
particle-in-cell experiments with CIC deposition, Gaussian and binomial
filtering, compensated transfer functions, bandwidth selection, full-`f` and
mode-projected linearized `delta-f` evolution, and conservation/transfer
diagnostics.

The repository also contains a frozen `examples/paper_results/` layer with the configurations, archived
seed-level histories, and reducers needed to reproduce the reported numerical
results. 

`python3 reproduce.py all` rebuilds the omitted derived output.

## Install the general code

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
- `benchmarks/`: Landau and two-stream initializations and theory;
- `validation/`: transfer, conservation, self-force, momentum, and rate-fit
  diagnostics.

command-line runners and Slurm examples are in
`pic_kde_2d/scripts/`. 

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
not submit simulations. New products go under `reproduced/`.

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

## Choose a filter for your own simulation

`dabs/` is a pip-installable package and command-line tool built on the
findings of the paper. For a given PIC configuration, it selects the
deposition/gather filter family and bandwidth with the smallest statistical
error whose predicted change in the growth or damping rates of the protected
modes stays within a tolerance you set. The configuration covers:

- the mesh, the particle shape and the field solve;
- any number of drifting Maxwellian species, in 1D, 2D or 3D.

```bash
python3 -m pip install -e ./dabs
dabs example two_stream -o case.yaml
dabs select case.yaml --tolerance 0.05 -o out/
```

See [dabs/README.md](dabs/README.md) for the method, the configuration
reference and the reproduction of the paper's selection.

## Repository layout

- `dabs/`: dynamics-aware kernel selector (package, CLI, docs and tests);
- `pic_kde_2d/`: general current research implementation and its tests;
- `examples/paper_results/`: paper-specific reducers, frozen configurations,
  regression tests, scientific records, required raw data, and compact final
  result anchors;
- `campaigns/`: self-contained completed campaign snapshots;
- `pic_kde_2d_landau_cv_stage/`: historical implementation snapshot retained
  where exact reproduction requires it;
- `reproduce.py`: existing-data reproduction entry point;
- `SIMULATION_RERUN_GUIDE.md`: requirements for simulation reruns.

