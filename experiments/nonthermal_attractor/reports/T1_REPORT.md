# T1 report: analysis library and synthetic identifiability suite

**Status: `ANALYSIS_VALIDATED`, with the documented limitations below.** No plasma simulation was run in T1.

## What was built

All code lives under `experiments/nonthermal_attractor/attractor_analysis/`.

| Module | Content |
|---|---|
| `schema.py` | Data contract `nta-1`: writer, reader, and a validator that rejects incomplete bundles (`protocol/DATA_CONTRACT.md`) |
| `energy.py` | N(ε) estimator with exact per-bin moments, overflow (never clipped), and Kish n_eff. Also: 1V Maxwellian references, fixed-threshold tail moments, the quantile feature ε_q, the power-law upper-cutoff feature, and the core-temperature tail detector |
| `waterbag.py` | ρ(η) reconstruction from particles or grid f; closure checks; one-particle floor η_floor; half-height detector for the lower edge η_min of the η⁻¹ range, with the `resolution_limited` flag; bin-insensitivity check |
| `nested.py` | Nested quiet loads. Every power-of-two prefix of a Halton(2,3) load with Cranley–Patterson rotation is itself quiet. Inverse CDFs for the waterbag beams, Gaussian beams and a Maxwellian |
| `convergence.py` | Regulator gate on paired nested differences (`PASS`, `PASS_AFTER_ONSET_ALIGNMENT`, `FAIL_REGULATOR_DEPENDENT`, `INCONCLUSIVE`); onset times; ln N vs N vs constant onset-scaling comparison (AIC) |
| `scaling.py` | t₀-profiled leave-one-out collapse fit with no per-snapshot amplitude freedom and a minimum-overlap penalty. Reports the Hessian covariance, the boundary flag, the **valley test along the power-law degeneracy direction (p, 1)** returning `UNIDENTIFIABLE_DEGENERATE` with the measured α − pβ, `STATIONARY`, `NO_COLLAPSE`; and moving-sector budgets |
| `forecast.py` | Frozen scaling forecast with masking of unsupported energies. Baselines: stationary, matched Maxwellian, one-scale heating. The quasilinear baseline is a stub to be reported `NOT_TESTED` unless justified. The oracle Maxwellian is excluded from ranking |
| `pipeline.py`, `suite.py`, `synthetic.py` | End-to-end analysis, claim logic, and cases S1–S7 |

**Tests.** `python3 -m pytest experiments/nonthermal_attractor/tests -m "not slow"` gives 32 fast tests, which pass. The 6 `slow` tests run a reduced suite.

## Synthetic suite

Command:

```
python3 experiments/nonthermal_attractor/scripts/run_t1_synthetic_suite.py
```

This takes 152 s on 4 cores. The full table is in `results/t1_synthetic/confusion_table.md`, and the data in `summary.json`. Runs used 5 seeds and 64 cells, with particles per cell ∈ {512, 2048, 8192} at the finest level.

**There were no false positives.** 19 of 25 verdicts match the expected verdict for well-resolved data. Five of the misses are *conservative*, meaning the analysis declined to claim something the data did not resolve. The sixth is explained below.

| Case | Result |
|---|---|
| S1 stationary Maxwellian | `NO_TAIL` at every N. Mean energy is unbiased (within 3 standard errors). |
| S2 fixed core + self-similar tail sector, true (α, β) = (−0.25, −0.5) | At 2048 and 8192 ppc: `IDENTIFIED` within CI, and the frozen forecast beats every baseline by the 0.8 margin. **At 512 ppc: `UNIDENTIFIABLE_DEGENERATE`**, because the moving cutoff is not supported by counts. The analysis then correctly sees only a power law. |
| S3 stationary ε⁻² tail | `UNIDENTIFIABLE_DEGENERATE` at every N, with α − 1.99β = 0.010 ± 0.009. It never returns separate exponents. |
| S4 core heats while the halo stays fixed (not self-similar) | `NO_COLLAPSE`, so no claim. |
| S4b pure Maxwellian heating (self-similar but thermal) | The collapse is identified, but the one-scale heating baseline ties it and there is no nonthermal tail, so no claim. This is the "self-similar is not the same as nonthermal attractor" check. |
| S2 gate (positive control) | `PASS` once every level resolves the feature (finest 2048 or 8192). `INCONCLUSIVE` when the coarse levels (128–256 ppc) leave the feature unsupported. |
| S5 cutoff ∝ ln N, fixed in time | `FAIL_REGULATOR_DEPENDENT` with 5 nested levels. **`INCONCLUSIVE` with 3 levels.** |
| S6 onset t₀ = a + b ln N, then physical scaling | 5 levels: `PASS_AFTER_ONSET_ALIGNMENT`, with ln N preferred (ΔAIC ≈ 44) and (α, β) recovered after t₀ profiling. 3 levels (finest 2048): the raw gate already returns `PASS`, because the onset shift over a factor of 4 in N is below the noise of the paired differences. The onset fit still prefers ln N and the exponents are identified. The post-onset evolution is genuinely converged, so this is a weaker label rather than a false claim. |
| S7 ρ(η) with η⁻¹ range extending downward | η_min is recovered to within one η bin (\|ln error\| ≤ 0.13) wherever it is not flagged. Every estimate that approaches the one-particle floor is flagged `RESOLUTION_LIMITED`. There are no confident wrong answers. |

## What T1 tells T2 (this sets the run count)

1. **Five or more nested levels are needed** to detect regulator dependence of a feature (S5) and to separate an onset shift from noise (S6). With 3 levels both come back `INCONCLUSIVE` or weaker.
2. **Onset scaling (ln N vs N), from `power_analysis.md`, at the S6 effect size b = 0.25 per e-fold of N.**
   - Onset scatter σ = 0.05: 4 levels × 4 seeds or 5 levels × 2 seeds give ≥ 0.95 power.
   - σ = 0.1: 5 levels × 3–4 seeds (0.88–0.92 power).
   - σ = 0.2: even 5 levels × 8 seeds reaches only 0.73.
   - The false-preference rate for ln N when the truth is linear in N stays ≤ 0.07 throughout.
   - **T2 should therefore use the full two-decade ladder (7–8 levels, e.g. 64 → 8192 ppc) and at least 4 seeds,** and should measure σ from the seeds before interpreting the onset model.
3. **Every level of the ladder must support the feature** (≥ 100 effective counts above ε_q, and η_min ≥ 10 η_floor). Features defined deeper in the tail need a finer coarsest level. For the 1D1V T2 box (about 3,300 cells), 64 ppc is roughly 2 × 10⁵ particles, so ε_q at q = 0.01 is supported. That should be checked on the first coarse run before the scan is frozen.
4. **The moving cutoff has to be resolved by counts at the finest level,** or the scaling fit returns `UNIDENTIFIABLE_DEGENERATE` (S2 at 512 ppc). This is the designed behaviour. It is also the most likely physical outcome if the tail is a stationary ε⁻².

## Limitations

- **Seed-scatter uncertainties from 5 seeds are noisy**, so χ²_red at the best fit is 1.2–1.9. The Hessian errors are inflated by χ²_red, and a seed bootstrap of the fit is not yet automated.
- **The degeneracy and collapse thresholds were calibrated on S1–S6** (valley_k = 5, χ²_red,max = 3, and the gate's slope ≥ 0.25 with τ = 0.1). They are frozen here. T2 and T3 must not retune them on plasma data.
- **The ρ(η) detector assumes an η⁻¹ range.** If the plasma ρ(η) has a different asymptote, the plateau criterion must be re-specified before looking at the data.
- **Prior art is `INCOMPLETE`** (`PRIOR_ART.md`). The Ewart et al. run parameters, which T2 needs, have not been read from the paper.
