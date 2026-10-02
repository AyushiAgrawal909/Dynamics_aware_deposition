# Nonthermal-attractor programme

Do nonthermal tails from electrostatic beam relaxation evolve self-similarly with identifiable exponents? If so, can early data forecast later tail evolution? And are these properties of Vlasov dynamics or of the PIC regulator?

The programme runs as three gated tasks. Their specs are in `specs/`, together with the original single-task spec they replace.

| Task | Status | Content |
|---|---|---|
| T1 | done (`reports/T1_REPORT.md`) | Analysis library, data contract, synthetic identifiability suite S1–S7. No physics runs. |
| T2 | code and validations done (`reports/T2_STATUS.md`); production scan prepared for Meluxina, not run | 1D1V spectral PIC and semi-Lagrangian Vlasov, cost estimate, nested ppc / Δv regulator scan on one waterbag preparation. This is the decisive gate. |
| T3 | not started; runs only if T2 returns `GO_T3` | Scaling fits, frozen forecasts, attraction |

## Commands (from the repository root)

```bash
python3 -m pip install -r requirements-simulation.txt   # numpy, scipy, numba, pytest

# T1
python3 -m pytest experiments/nonthermal_attractor/tests -m "not slow"   # fast unit tests
python3 -m pytest experiments/nonthermal_attractor/tests -m slow         # reduced synthetic suite
python3 experiments/nonthermal_attractor/scripts/run_t1_synthetic_suite.py   # full suite (~3 min)
```

The T2 commands are listed in `reports/T2_STATUS.md`.

## Layout

| Path | Content |
|---|---|
| `attractor_analysis/` | T1 analysis library (pure numpy/scipy) |
| `attractor_sim/` | T2 backends, linear theory, validations, runner |
| `protocol/` | Data contract and the pre-registration template |
| `scripts/` | Entry points |
| `configs/` | Run configurations |
| `slurm/` | Meluxina scripts, with `# PLACEHOLDER` lines that must be edited |
| `tests/` | pytest |
| `reports/` | T1 report, prior art, cost estimate, T2 status |
| `results/` | Small artifacts only. Large outputs go outside the repository. |
| `specs/` | Task specifications |

## Rules carried from the specs

- **Synthetic data validate the analysis only.** Their bundles are labelled `backend="synthetic"`.
- **Never label reference-code output as HEXAPIC output.** HEXAPIC is not used in the 1D phase.
- **Freeze `protocol/T2_PROTOCOL.yaml` in its own commit before the scan.** The T1 thresholds are frozen and are not retuned on plasma data.
