# pic_kde_2d

`pic_kde_2d` is a research implementation of periodic 2D electrostatic PIC for
studying deposition error, smoothing and dynamics-aware bandwidth selection.

## Numerical scope

- normalized units: `lambda_D = 1`, `omega_p = 1`, `q_e = -1`, and
  `m_e = 1`;
- periodic Cartesian grids with arrays indexed as `(x, y)`;
- structure-of-arrays particle state `(x, y, vx, vy, w)`;
- spectral Poisson solve and electric-field calculation;
- CIC deposition/gather plus Gaussian, binomial and compensated filtering;
- full-`f` and mode-projected linearized `delta-f` loops;
- Landau and two-stream benchmark helpers;
- transfer, rate-fit, conservation, momentum and self-force diagnostics.

The code is research software. Reproduce a historical campaign with its frozen
configuration and implementation snapshot rather than assuming the newest
initializer is historically interchangeable.

## Install

From the repository root:

```bash
python3 -m pip install -r requirements-simulation.txt
python3 -m pip install -e ./pic_kde_2d
```

## Test

```bash
python3 -m pytest pic_kde_2d/tests
```

## Layout

- `pic_kde_2d/core`: grids, particles and time-advance loops;
- `pic_kde_2d/methods`: deposition, filters, selectors and delta-`f` helpers;
- `pic_kde_2d/benchmarks`: benchmark setups and theory helpers;
- `pic_kde_2d/validation`: numerical diagnostics;
- `scripts`: executable research workflows and cluster launchers;
- `tests`: implementation and protocol tests.

The top-level `examples/paper_results/` directory contains the frozen analysis
layer used to reproduce the associated article results.
