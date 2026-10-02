# Recorded packaging and verification environment

The clean bundle was assembled and its core regression suite was run with:

- macOS on Apple Silicon
- Python 3.14.6
- NumPy 2.5.1
- Matplotlib 3.11.1
- Tectonic 0.17.0
- Poppler command-line tools available

SciPy, pytest, Numba, Torch, and h5py were not installed in the interpreter
used for the packaging check. They were therefore not falsely recorded as
tested exact versions. The 84 standard-library `unittest` regressions passed in
this environment. `requirements-analysis.txt` and
`requirements-simulation.txt` retain the project's documented compatible
ranges for broader workflows.

Historical production environment versions remain as recorded—or explicitly
unknown—in the campaign manifests and `MISSING_INFORMATION.md`.

