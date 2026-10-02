# T2 status: code and validations done; production scan not run

**Gate outcome: `NOT_TESTED`.** The regulator scan has not been run. It needs about 500 wall-hours on this 4-core container, against an 8-hour local budget (`COST_ESTIMATE.md`), so it is prepared as SLURM jobs for Meluxina. The T2 spec says never to launch an underpowered run and interpret it, so none was. The smoke ladder below exercises the pipeline only.

## What exists

| Item | Where | State |
|---|---|---|
| 1D1V spectral PIC | `attractor_sim/pic1d1v.py` | The matched-CIC chain from `pic_kde_2d`, numba-parallel. Momentum conserved to round-off (V1). |
| 1D1V semi-Lagrangian Vlasov | `attractor_sim/vlasov1d1v.py` | Strang split. Spectral x-shift; conservative, positivity-preserving PCHIP-primitive v-shift. Velocity-boundary outflow is tracked. |
| Linear theory | `attractor_sim/theory.py` | Waterbag-beam quartic: cold limit and Δv_b = 0.2 reference values reproduced. Landau root for kλ_D = 0.5. |
| Validations V1–V4 | `scripts/t2_validate.py` → `T2_VALIDATION.md` | All pass: Landau 0.4% (both codes), waterbag growth 0.7% (PIC) and 1.2% (Vlasov), PIC momentum 2e-16. |
| Shot-noise floor | `scripts/t2_noise_floor.py` | Sets the coarsest ppc (below). |
| Cost estimate | `scripts/t2_cost_estimate.py` → `COST_ESTIMATE.md` | Does not fit locally. |
| Runner, scan, analysis | `attractor_sim/run.py`, `scripts/t2_scan.py`, `scripts/t2_analyze.py` | Writes T1-contract bundles; applies the frozen T1 gate; writes `T2_DECISION.json` |
| SLURM | `slurm/t2_array.sbatch`, `slurm/t2_analyze.sbatch` | `# PLACEHOLDER` lines must be edited |
| Protocol | `protocol/T2_PROTOCOL.yaml` | **DRAFT**, with open items listed in the file |
| Configs | `configs/t2_waterbag_dv020.yaml` (production), `configs/t2_smoke.yaml` (pipeline smoke only) | |

## Finding from the noise-floor measurement (changes the ladder)

A Halton quiet start is near-exact at t = 0. After one plasma period of free streaming, the per-mode density fluctuation in the seeded band is:

| ppc | Floor (per mode) |
|---|---|
| 64 | 2.3e-4 |
| 128 | 9.5e-5 |
| 256 | 6.0e-5 |
| 512 | 3.3e-5 |
| 8192 | 1.8e-6 |

The spec needs the seed to be at least 10× the floor and also at most 1e-3. With both constraints, **64 ppc cannot be the coarsest level**. The ladder is therefore 16384 → 128 ppc: 8 nested levels, 2.1 decades, and the seed ratio is 10.5× (marginal) at 128 ppc.

The alternative is to raise the seed to about 2.5e-3 per mode and keep 64 ppc. That is your call before the freeze; it is listed in `T2_PROTOCOL.yaml`.

## Commands

```bash
cd experiments/nonthermal_attractor
python3 scripts/t2_validate.py                                   # V1-V4 (~1 min)
python3 scripts/t2_noise_floor.py configs/t2_waterbag_dv020.yaml --ppc 128
python3 scripts/t2_cost_estimate.py configs/t2_waterbag_dv020.yaml
# after freezing protocol/T2_PROTOCOL.yaml:
python3 scripts/t2_scan.py configs/t2_waterbag_dv020.yaml --out $OUT --list > slurm/t2_jobs.txt   # 52 runs
sbatch --array=1-52 slurm/t2_array.sbatch
sbatch slurm/t2_analyze.sbatch
# local pipeline smoke (not physics):
python3 scripts/t2_scan.py configs/t2_smoke.yaml --out /tmp/t2_smoke
python3 scripts/t2_analyze.py configs/t2_smoke.yaml /tmp/t2_smoke --out results/t2_smoke --label SMOKE
```

## Not done in this revision

- **The optional pair-plasma arm.** It needs a mobile positron species. It is a distinct physical model, not another preparation.
- **The Δv_b = 0.15 v_b preparation.** Only its cost is estimated.
- **The L-doubling check.** It is listed in the protocol and runs as an extra job.
- **Exact Vlasov positivity.** The spectral x-shift can undershoot slightly (f_min is recorded per snapshot). A positivity-preserving x-advection (PFC) is the upgrade if the undershoots matter for ρ(η) at low η.
