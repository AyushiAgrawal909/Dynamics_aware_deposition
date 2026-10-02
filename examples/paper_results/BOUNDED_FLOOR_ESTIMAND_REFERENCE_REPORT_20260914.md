# Paper 1 bounded floor, estimand and finite-reference correction

This report uses archived histories only. No simulation was launched and no
manuscript source was edited. The executable analysis is
`analyze_bounded_floor_estimand_reference_20260914.py`; final versioned output
is `results/bounded_floor_estimand_reference_correction_20260914_r12/`.

## Direct weak-Landau CIC floor comparison

Both ensembles contain 40 histories and use the same `64 x 16` mesh,
normalisation, Fourier convention, global inverse-normal loading protocol,
solver, `dt=0.05`, `t=90`, diagnostic cadence and frozen reducer. Particle
sets with the same seed identifier were generated independently and are not
nested.

| quantity | PPC 1024 | PPC 4096 |
|---|---:|---:|
| coherent floor, `abs(rfft2(Ex)[1,0]/(Nx Ny))` | 2.53969e-4 | 4.06958e-4 |
| coherent amplitude at `t=0` | 1.248979e-2 | 1.248981e-2 |
| raw coherent amplitude at `t=5` | 6.72396e-3 | 6.53571e-3 |
| 49-sample envelope at `t=5` | 7.88963e-3 | 7.86602e-3 |
| fit-start signal/floor | 31.065 | 19.329 |
| persistent `5 x floor` crossing | 32.6 | 25.4 |
| accepted interval | [5, 30.1] | [5, 22.9] |
| fitted damping rate | -0.0685811 | -0.0670563 |
| fitted amplitude e-foldings | 1.7214 | 1.2003 |
| selected floor interval | [85, 90] | [85, 90] |

The envelope is the edge-padded, centred 49-sample geometric mean of the
magnitude of the coherent complex seed mean. The floor is the minimum rolling
median over complete five-time-unit windows stepped by one in `[70,90]`.

The floor difference is measured directly, not inferred from the crossings.
Initial amplitudes agree to `1.0e-6` relatively and fit-start envelopes to
0.30%, so neither normalisation nor initial coherence explains it. In the
selected late interval, PPC 4096 has only 0.5605 times the PPC-1024 mean
individual-seed magnitude, and its incoherent standard-error amplitude is
0.4353 times as large. However, its amplitude-weighted coherence is 2.726
times larger (0.742 versus 0.272). The coherent floor is consequently 1.602
times higher. Early seed-frequency spread is also smaller at PPC 4096
(`0.00605` versus `0.00979`), consistent with less dephasing. Recurrence is
detected at 64.8 and 69.2 respectively, after both accepted fits.

This demonstrates that coherent cancellation/retained phase coherence, not
larger incoherent sampling noise, explains the reducer-level ordering. The
archives do not isolate why the independently generated PPC-4096 loading
retains more phase coherence; particle count, realised loading and late-time
dynamics remain confounded.

## E2 estimands and initialization-only prediction

The main physics comparison should be the seed-specific initialization-only
finite-marker rate against that seed's fitted rate. The predictor is the
principal represented-mode, diagonal `+k` response of the explicit-midpoint
update using the actual current marker velocities, beam signs, score weights,
CIC factors and exact round-trip transfer. It omits spatial aliases in the
finite-marker sum, finite-sample `+k/-k` mixing and nonlinear full-f response.
It is approximate and contains no fitted rate or amplitude.

| drift | arm | predicted ratio of means | measured ratio of individual-rate means | registered ensemble-amplitude ratio |
|---:|---|---:|---:|---:|
| .60 | fixed boundary | .99275 | .99275 [.99267,.99284] | .99278 [.99268,.99287] |
| .60 | R=.95 interior | .95035 | .95038 [.94979,.95103] | .95059 [.94991,.95122] |
| .70 | fixed boundary | .98570 | .98567 [.98525,.98611] | .98595 [.98537,.98636] |
| .70 | R=.95 interior | .95085 | .95074 [.94927,.95228] | .95173 [.94970,.95317] |
| .76 | fixed boundary | .97048 | .97057 [.96860,.97254] | .97269 [.96971,.97426] |
| .76 | R=.95 interior | .95179 | .95193 [.94868,.95518] | .95546 [.95052,.95804] |
| .80 | fixed boundary | .90702 | .90891 [.88743,.92666] | .94064 [.90797,.94692] |
| .80 | R=.95 interior | .95360 | .95446 [.94407,.96310] | .96986 [.95376,.97301] |

The registered ensemble-amplitude results and original verdicts remain
prominent as a distinct observable. The individual-rate presentation is a
scientifically motivated post-hoc sensitivity. Because asymptotic eigenmode
amplitudes cannot be predicted from equal initial field amplitudes alone, no
independent predicted ensemble-amplitude slope is reported. At `v_b=.80`, the
fixed-arm individual interval crosses 0.90: preservation is not established;
this does not prove distortion exceeding 10%. The fixed arm is a
theory-boundary control, not a density-optimal selector.

No saved artifact specifies Claude's claimed “18-fold” averaging convention.
Recalculation gives RMSE-improvement factors of 35.34 when 120 current
arm/seed rows are weighted equally, 22.16 after first forming 12 arm/drift
means, and 19.68 for the three `v_b=.80` arm means. Historical nominal rows
give 7.80 and current-plus-historical rows 10.23. Thus 18-fold is not
reproducible as a unique number. What is established is a large descriptive
residual reduction for current audited initializations. Arms share a loading
within seed and drift templates are paired, so rows are dependent and are not
120 independent validations. Historical reconstructed predictions remain
nominal.

## Finite-reference correction

Population variance and covariance are used consistently, so

`Q_H = MSE(X_H-R) - Var(R) + 2 Cov(X_H,R)`

equals candidate variance plus the squared candidate/reference mean
difference. Complete candidate/reference seed pairs are resampled and every
term is recomputed. Reported uncertainty below is a log-scale bootstrap-SE
interval centred on the plug-in ratio. Raw percentile ranges are retained in
the CSV because resampling a population-variance functional shrinks its
variance and visibly biases the adjusted-ratio bootstrap, especially for the
wide comparator.

| metric/arm | original CIC/arm | adjusted CIC/arm | `+2 Cov` contribution |
|---|---:|---:|---:|
| density, C4 sigma=1 | 3.25865 [3.25215,3.26516] | 7.218 [7.101,7.338] | -2.401e-7 (-0.153% of pair MSE) |
| density, fixed CV-derived 7.2443 | 4.61153 [4.59803,4.62507] | 36.858 [34.063,39.883] | -1.454e-8 (-0.013% of pair MSE) |
| pre-return grid field, C4 sigma=1 | 1.61280 [1.60371,1.62194] | 1.8951 [1.8790,1.9114] | -1.712e-8 (-0.145% of pair MSE) |
| pre-return grid field, fixed CV-derived 7.2443 | 0.63238 [0.61847,0.64661] | 0.5803 [0.5659,0.5950] | -3.009e-8 (-0.100% of pair MSE) |

The central 7.218 and 36.858 calculations are verified. Their raw percentile
bootstrap distributions are `[6.828,7.055]` and `[27.77,32.50]`, shifted below
the plug-in values by the variance-functional bootstrap bias. The wide-arm
adjustment is therefore statistically fragile as well as reference-dependent.
Neither number is a continuum-error reduction unless the PPC-4096 reference
mean is independently shown unbiased relative to the continuum.

## Corrections and run decision

The supplement now renders delta weights as `w_j^{delta}`, uses explicit
midpoint fractions, correct spacing and natural logarithms, and includes the
finite-reference identity. All nine equation-bearing pages were rendered and
visually inspected. Proposed replacement I now states: one-way field amplitude
has `W`, matched particle-field amplitude has `W^2`, and its power has
`|W|^4`; the pre-return grid-field power remains a distinct `|W|^2` metric.

The corrected benefit figure shows full intervals, identifies the green band
only as the rate-ratio tolerance, separates provenance from validation status,
and explicitly marks the Landau CIC duration limitation.

No new run is recommended from this task. For a possible `k lambda_D=0.5`
benchmark, the continuum root is
`omega=1.415662-0.153359 i`, the period is 4.4383 and two damping e-foldings
take 13.041. With a fixed start at `t=5`, the nominal two-e-fold endpoint is
18.041, and a centred 45-sample envelope at cadence 0.1 requires about 2.2
additional units of look-ahead. More importantly, a two-e-fold fit above a
five-floor threshold needs fit-start SNR at least `5 exp(2)=36.95`. A
conditional extrapolation from the current amplitude, including the `1/k`
field scaling and stronger damping before `t=5`, gives an envelope near
`4.1e-3` and would require a floor below roughly `1.1e-4`. Existing coherent
floors do not support that assumption, and their PPC ordering is not monotone.
This estimate also omits altered transients and late dynamics. Therefore
`2/|gamma|` alone does not justify `k lambda_D=0.5`; it remains a possible
benchmark only after a separately reviewed signal/transient preflight.

The supported measurement improvement is to archive and report the coherent
residual, individual-seed magnitude, amplitude-weighted coherence and
incoherent standard-error amplitude together. For rate physics, retain the
registered coherent observable but add individual complex-mode fits as the
predeclared companion estimand in any future campaign.

## Writing handoff

The rewrite should follow this argument:

1. State the physical mode information absent from density-only optimisation.
2. Introduce the fixed-density, fixed-transfer E2 equilibrium scan as the
   controlled mechanism test.
3. Compare initialization-only finite-marker predictions with individual
   measurements, then show the registered ensemble observable separately.
4. Introduce the protected-mode constraint and the R=.95 interior margin,
   keeping the boundary as a control rather than a measured optimum.
5. End with measured error benefits, campaign provenance, duration/signal
   limitations and the distinction between finite-reference and continuum
   error.

Remaining uncertainties are bounded: the origin of the PPC-4096 coherent
late residual, bias of the finite reference relative to continuum, omitted
couplings in the finite-marker predictor, and unavailable exact historical
initializations. These narrow claims; they do not require a simulation in the
present task.

Commands used:

```text
MPLBACKEND=Agg python3 analyze_bounded_floor_estimand_reference_20260914.py \
  --output results/bounded_floor_estimand_reference_correction_20260914_r12
tectonic -X compile DISCRETE_FINITE_MARKER_TECHNICAL_SUPPLEMENT.tex \
  --outdir results/bounded_floor_estimand_reference_correction_20260914_r12 \
  --keep-intermediates
python3 -m py_compile analyze_bounded_floor_estimand_reference_20260914.py
```

Manuscript hashes remain unchanged:

* `Dynamics_constrained_Deposition/introduction.tex`: `44a76c...26be6`
* `Paper1_Rewritten/main.tex`: `59bc6e...cdb9`
* `paper1_source_package/paper1.tex`: `4c4004...1827`
