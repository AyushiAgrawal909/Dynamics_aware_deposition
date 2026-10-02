# Paper 1 research completion report

Completion date: 2026-09-12. This report continues from the verified
2026-09-11 repository state. No manuscript source was edited and no new
production simulation was submitted.

## Outcome

The evidence supports a narrower paper: an implementation-audited method for
choosing a filter bandwidth from declared protected linear modes, with a
separate interior safety margin and explicit measurement gates. It shows that
density-only selection can distort dynamics and that a constant filter
transfer has different rate effects as the equilibrium approaches marginality.

It does **not** establish unrestricted nonlinear full-`f` fidelity, universal
or global optimality, a calibrated near-marginal continuum model at finite
marker count, particle-force noise reduction for the E2 interior width, or an
end-to-end computational speedup.

## Sequential task results

### 1. Recover the submitted `v_b=0.818` extension — COMPLETE

MeluXina job 5202227 completed with exit code `0:0` in 02:21:40 under the
unchanged eight-hour amendment. All 30 requested histories, result files and
per-run manifests were recovered. Their frozen-campaign hash is
`e664460d392a216b953bc87d3859c3bce9e1cce2f1652ab9417c1a58b3baf446`;
the budget amendment hash is
`fc9ae62fcdd7321c546718852582a4759c977e55d3791673cbdd8274f4b15700`.
The original job 5201231 remains preserved as a 120/150 four-hour timeout.

The frozen reducer gives the following descriptive outputs:

| Arm | rate | ratio to CIC | fit-start SNR | max `delta f` ratio | disposition |
|---|---:|---:|---:|---:|---|
| CIC | 0.114044 | 1 | 3.563 | 5240.94 | signal gate fails |
| fixed Gaussian | 0.101783 | 0.892489 | 2.854 | 395.95 | unresolved |
| constrained Gaussian | 0.113689 | 0.996893 | 3.531 | 4862.15 | unresolved |

All three start below the required SNR 5, and three of ten seeds per arm later
exceed the linear `delta-f<0.5` threshold. The arithmetic mean becomes almost
single-seed dominated. These are therefore not accepted growth-rate or
stabilisation measurements. A post-hoc valid-prefix fit is retained only as a
diagnostic and does not replace the frozen window.

Evidence:
`results/e2_vb0818_extension_audit_20260912/` and
`results/e2_beam_drift_scan_analysis_20260912_complete_extension_r9/`.

### 2. Complete estimator comparisons — COMPLETE

The registered slope of log arithmetic-mean amplitude was reproduced exactly
for the completed drift scan, historical Table-5 corner, and direct
compensated corner. Individual-seed rates, distributions, fitted exponential
mixtures, every time-dependent seed share, effective contributor counts and
leave-one-seed-out influence were added.

The 5,000-resample sensitivity uses paired seed indices and RNG 20260911. At
`v_b=.80`, fixed/CIC is 0.908911 `[0.887432,0.926656]` and
constrained/CIC is 0.954460 `[0.944068,0.963103]` for the ratio of individual
rate means. Registered primary values remain 0.940636 and 0.969862. The
constrained result is robust; the fixed filter's location relative to 0.90 is
estimator-sensitive.

For the direct compensated corner, individual-rate intervals confirm failure
of `p1n1` and `p2n4` and preservation for `p2n2` and `p3n4`. Registered
results remain primary.

Evidence:
`results/e2_estimator_comparison_complete_20260912_with_extension/`.

### 3. Diagnose seed-rate variation — COMPLETE

Saved initial target amplitude and phase are seed-invariant to round-off.
Across the fixed window grid, shifts in mean individual rate are much smaller
than the seed SD. They cannot explain the approximately 19% coefficient of
variation at `v_b=.80`.

The actual finite-marker linear response of the audited loading does explain
it. At `v_b=.80` CIC, predicted versus observed seed rates have correlation
0.999885 and RMSE 0.000488. At `.818`, correlations are 0.997924 (CIC),
0.994422 (fixed), and 0.997863 (constrained). The observed `.80` CIC rate also
correlates `-0.9465` with beam-sign/thermal-velocity covariance. Thus two
effects are distinct:

1. finite-marker loading produces heterogeneous individual seed rates; and
2. arithmetic mean-amplitude fitting increasingly weights the fastest seeds.

Historical marker arrays were not saved; reconstruction with the current
initializer is only nominal. No higher-PPC run was launched because the
existing histories already identify the mechanism. A higher-PPC marginal
study would be needed only to test continuum convergence at `.818`.

Evidence: `ESTIMATOR_AND_SEED_DIAGNOSTIC.md` and
`results/e2_seed_rate_variation_20260912_with_extension/`.

### 4. Correct exact filter and operator inconsistencies — COMPLETE

The exact raw-binomial transfer is `cos(k dx/2)^(2n)`.
`sigma_eq=sqrt(n/2) dx` is only its low-`k` Gaussian correspondence. At the E2
longitudinal `m=12,n=4` point, using the Gaussian proxy would produce a 0.512%
one-way and 1.022% round-trip transfer difference. The production filters,
transfer regressions and pass-budget helper already use exact binomial
transfers, so no particle histories, measured reductions or pass ceilings
change.

The Python production solve is spectral and must use `k/|k|^2`, not
finite-difference `D_h/Lambda_h`, in its force-noise equation. The E2
dispersion/root code already uses the spectral symbols, so predicted roots and
frozen widths also remain unchanged. These are manuscript equation/wording
corrections only.

Evidence: `OPERATOR_AND_TRANSFER_AUDIT.md` and
`results/operator_transfer_correction_20260911/`.

### 5. Measure noise at the actual `R=.95` interior setting — COMPLETE for `v_b=.80`

No saved arrays existed at exactly `sigma_y=0.1738592384 Delta y`. A
deterministic replay of the compatible ten-seed full-`f` short-window protocol
was therefore used. Before evaluating the requested width, it reproduces each
saved `q=.50` and boundary scalar metric to absolute tolerance `5e-15`.

At the exact interior width, the density-error-power reduction is 1.081314
`[1.079778,1.082929]`, and the pre-gather grid-field-error-power reduction is
1.038554 `[1.037245,1.040182]`. The boundary width `0.2427346 Delta y` remains
separate at 1.157140 and 1.073443. These are PPC 32 versus paired PPC 1024,
`v_b=.80`, `t=[.5,4]`; they are not particle-force errors or same-history
projected-rate measurements.

Evidence: `results/e2_interior_noise_replay_20260911/`.

### 6. Decide the practical claim — COMPLETE with narrowed scope

The intended strong claim—one end-to-end full-`f` E2 campaign simultaneously
showing rate fidelity and force-noise benefit—is not available. It is also not
necessary if the paper is narrowed transparently:

- projected linear E2 histories establish the protected-mode mechanism and
  measured rate preservation at signal-resolved settings;
- a compatible but separate short full-`f` replay establishes a modest
  density/grid-field noise benefit at the exact `.80` interior width; and
- the weak-Landau v2 campaign supplies a genuine same-campaign comparison,
  but promotion fails because CIC has only 1.721 fitted e-foldings even though
  C4 has 2.076 and C4/CIC is 0.989415 `[0.967884,1.013827]`.

A new E2 full-`f` comparison is required only if coauthors retain an
unrestricted, same-campaign practical claim. It is not required for the
narrowed paper and was not launched.

## Other evidence questions closed

- Historical job 4867515, which supplies Table-4 C4/Silverman density errors,
  used the rejected repeated 32-node-per-cell longitudinal velocity
  construction. Those density metrics remain valid for that constructed
  ensemble but cannot be joined to global-loading rates as a same-campaign
  physical validation. No recovered historical Landau rate job is identified
  as using that per-cell construction.
- The previously disputed E2 C4 signal evidence is the recovered
  pre-emergence floor with fit-start SNR 11.213 and a shared-window CIC
  baseline spanning 4.484 e-foldings. Older “baseline duration not applicable”
  entries came from a retrospective candidate-only schema, not automatic
  baseline passes. New confirmations explicitly require both arms.
- Compensated E2 model discrepancies are reported independently of the 10%
  preservation gate: 4.428%, 0.932%, 6.486% and 0.545% for `p1n1`, `p2n2`,
  `p2n4`, and `p3n4`, and all predictions lie outside the narrow registered
  intervals.

## Commands and verification

One command rebuilds the affected tables and figures from cached raw data and
submits no simulation:

```bash
MPLBACKEND=Agg python3 \
  paper1_source_package/reproducibility/rebuild_research_completion.py \
  --output-root /new/nonexistent/rebuild-directory
```

Production simulation commands remain separately documented in the frozen
MeluXina launchers and were not invoked. Representative regression commands:

```bash
python3 -m unittest discover \
  -s paper1_source_package/reproducibility/tests -p 'test_*.py'
python3 -m json.tool \
  paper1_source_package/reproducibility/CAMPAIGN_MANIFEST.json >/dev/null
```

Final verification: the current-code end-to-end rebuild completed with 35
hashed products and `simulation_submitted=false` in
`results/research_completion_rebuild_check_20260912_r3/`; five key scientific
CSVs are byte-identical to the canonical artifacts. The reproducibility suite
passes 84/84 tests, the secondary-claim suite passes 26/26 tests, and
`SHA256SUMS.txt` records 7,184 files. The first packaging attempt, which failed
only while writing its final manifest because of a lowercase Python boolean,
is retained; the defect and correction are recorded in `AUDIT_CHANGELOG.md`.

## Strongest supported quantitative findings

1. In the controlled projected E2 scan, the precomputed `R=.95` interior
   settings pass paired 10% rate, signal and duration gates for all four
   resolved drifts `.60--.80`; at `.80` the registered ratio is 0.969862
   `[0.95376,0.97301]`.
2. In the fresh weak-Landau same-campaign experiment, C4 gives 3.25865x density
   and 1.61280x pre-gather grid-field error-power reductions while C4/CIC is
   0.989415 `[0.967884,1.013827]`; the complete claim still fails the CIC
   duration rule.
3. At the direct E2 corner, exact compensated `p=3,n=4` gives a measured rate
   ratio 0.973416 `[0.970473,0.975293]`, and its `(12,0)` controlled transfer
   is `W^2=0.997137575`; rate preservation is established even though model
   calibration to the narrow interval is not.

## Final handoff

1. **Problem and user.** The method helps PIC practitioners choose smoothing
   that reduces selected grid errors without silently erasing declared linear
   plasma modes.
2. **What is new.** Relative to the checked primary literature, the defensible
   distinction is the implementation-audited protected-mode bandwidth rule,
   explicit interior margin, and controlled equilibrium scan—not KDE,
   smoothing-induced dispersion or ensemble averaging themselves.
3. **Where it outperforms.** It rejects density-only widths that badly distort
   rates and identifies rate-safe Gaussian/compensated settings. The measured
   cost is reduced noise benefit compared with the density optimum; no
   universal or end-to-end performance ranking is supported.
4. **Strongest numbers.** The three results immediately above.
5. **Conditional/failed/outside scope.** Weak Landau fails baseline duration;
   `.818` is signal/linearity-unresolved; exact compensated model calibration
   is imperfect; nonlinear full-`f`, particle-force noise, continuous global
   optimality and end-to-end speedup are outside scope.
6. **Further runs.** None are necessary for the narrowed paper. The evidence
   is ready for coauthor review and then targeted manuscript editing using
   `MANUSCRIPT_CORRECTIONS.md`.
