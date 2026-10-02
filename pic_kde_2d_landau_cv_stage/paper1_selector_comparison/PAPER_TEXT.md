# Preregistered Methods Text

## Delta-f Scope

Delta-f is shown as a variance-reduction reference rather than as a competing
bandwidth selector. Unlike CIC, density-based bandwidth rules, and C4, delta-f
changes the particle formulation and assumes a small perturbation about a known
background distribution. Ranking it as a selector would therefore compare
incommensurable methods. Its role in the comparison is to show the near-noiseless
limit against which the measurability achieved by the selector methods can be
interpreted.

The validity boundary is measured rather than assumed. In the frozen Landau
test, `alpha=0.001` remains valid with maximum weight ratio
`0.1138 [0.1098, 0.1240]`. At `alpha=0.01`, every seed breaches the `0.5`
validity threshold, with median first crossing at `t=6.1`; at `alpha=0.1`, the
initial perturbation is already outside the delta-f validity range. Delta-f is
therefore a high-precision small-perturbation reference, not a full-f-general
replacement for C4.

The ratios scale exactly by ten under each tenfold increase in `alpha`, as
required because the linearized weight equation gives `w_delta proportional
to alpha`. This is both an implementation check and a validity-boundary
measurement; by itself it is not evidence that the resulting field trajectory
is dynamically wrong after the threshold is crossed.

The delta-f rate reference uses `alpha=0.001`, whereas the frozen full-f Landau
matrix benchmark uses `alpha=0.01`. This cross-amplitude comparison is valid
for the linear damping rate because `gamma` is amplitude-independent in linear
theory. It is not a valid delta-f reference for the benchmark's noise floor,
signal-to-noise ratio, or measurability. At the matrix operating amplitude,
every tested delta-f seed breaches the standard `|w_delta|/|w_full| < 0.5`
criterion by `t=6.1`; delta-f is therefore inapplicable as a self-consistent
reference at the operating point where C4 runs natively.

The large-amplitude comparison also shows a dynamics-level failure rather than
only a validity-indicator breach. At `alpha=0.1`, full-f and linearized delta-f
separate persistently by 25% at `t=9.5`. The full-f mode then arrests, rebounds,
and exhibits nonlinear bounce oscillations, while linearized delta-f continues
smooth Landau decay; over `t in [40, 60]`, the full-f mean mode amplitude is
`6.73` times the linearized value. Linearized delta-f therefore cannot represent
the trapping physics present in the full-f benchmark outside its validity
range.

## Preregistered Outcome Interpretation

A selector is successful only when its paired 95% confidence interval satisfies
the rate-fidelity tolerance and the paired lower confidence bound on noise
reduction exceeds unity. If C4 satisfies the rate constraint in the
near-marginal two-stream case but its noise-reduction interval does not exclude
unity, we report the result as a closed safe window: the dynamics-safe width is
identified, but it permits no statistically demonstrated smoothing benefit.
This outcome does not support the claim that C4 wins the cell, although it is
consistent with the predicted collapse of admissible smoothing near marginal
stability.

## Rule-of-thumb Contrast

The periodic Silverman rule selects a width 1.83 times the C4 width for weak
Landau damping but 31.55 times the C4 width for near-marginal two-stream. The
matrix reports these ratios directly. This preregistered comparison tests
whether a density rule that is approximately adequate for a robust damped mode
becomes dynamically unsafe as marginality narrows the admissible smoothing
window.

## Unified Coupling Mechanism

The matched transfer reduces the plasma coupling by
`sinc(pi*m/N)^4*exp[-(2*pi*m*sigma/N)^2]`. The resulting dispersion shift has
one physical interpretation in both benchmark classes: smoothing moves each
mode toward greater stability. A growing two-stream mode approaches marginal
stability and its positive growth rate falls, whereas a damped Langmuir mode
acquires a larger damping magnitude. At still larger widths the coherent mode
is erased into the estimator floor and a fitted rate no longer measures the
dispersion root.

The zero-parameter warm+CIC calculation predicts 10% rate ceilings of
`1.062 dx` for isolated two-stream `m=8` and `0.199 dx` for the protected
`m=12` edge. The corresponding kinetic Landau calculation predicts `2.453 dx`,
inside the measured `2-3 dx` central-value bracket.

At the measured `m=8` 10% crossing, `0.99 dx`, the closed form predicts a rate
ratio of `0.913842`, or 8.616% damage. Calibrating the damage to the measured
10% gives a bandwidth divisor
`sqrt(0.10/0.086158)=1.0773`. This maps the raw closed-form `m=12` ceiling
`0.198720 dx` to a bias-corrected prior of `0.1845 dx`. This one-point prior was
then tested, rather than assumed transferable, in the independent `m=12`
sweep described below.

The projected linearized-delta-f sweep resolves the binding `m=12` response.
Every adjacent rate-ratio decrease is positive at 95% confidence, and the
paired four-point comparison favors the raw closed form: RMSE `0.01493` versus
`0.03071` for the transferred `m=8` bias correction, with the paired RMSE
difference in `[0.01548, 0.01578]`. The `m=8` correction therefore does not
transfer to the protected edge.

The independent `m=1-14` linearized-delta-f band sweep resolves the upper edge.
Modes `m=2-12` pass the strict `R^2>0.98`, two-e-folding growth gate. Mode
`m=1` has a positive net trend but remains unresolved by that strict linearity
gate and is not labeled stable. Modes `m=13-14` show no resolved exponential
growth: their long-window log-amplitude fits have `R^2<0.01`, and their final
amplitudes are below their initial amplitudes. The highest clean unstable mode
is therefore `m=12`; it remains the binding protected mode. At `m=12`, the
measured rate is `0.15291 [0.14166, 0.16025]`, compared with the warm prediction
`0.14467`.

The selected width is `0.1845 dx`. Its measured protected-edge fidelity is
`0.9273 [0.9143, 0.9334]`, satisfying the all-mode 10% rate gate with margin.
The raw crossing `0.1987 dx` narrowly fails the confidence-supported gate
because its lower bound is `0.89984`. At the exact selected width, the measured
noise reduction is `1.2005 [1.1977, 1.2034]`; therefore the near-marginal safe
window is narrow but not closed.

## Right-censored CV Widths

The unguarded LOO-CV row is retained as a right-censored characterization of
the density objective. Its continued decrease as the perturbation approaches a
uniform density is correctly divergent behavior, not a pathological optimizer:
in the uniform limit the bias vanishes and density MISE/LSCV is minimized by an
infinitely wide kernel. Wu and Qin explicitly identify this limit and do not
re-optimize the width when their Anderson-Darling test cannot reject
uniformity. We therefore report their AD-gated update policy as a separate row
rather than using it to conceal the divergence of unguarded CV.

The right-censored response is qualitatively different across the two
benchmarks. At the Landau search boundary, `sigma >= 60 dx` gives a protected
mode response `W^2 <= 8.52e-16`, already below any physically useful signal in
the calculation. For isolated E2 `m=8`, the same boundary gives
`W^2 <= 7.85e-242`. The latter is representable as a normal IEEE binary64
scalar, but it is about 225 decades below binary64 relative roundoff in an
order-one field calculation and it underflows binary32. It is therefore
operationally zero in the FFT-field chain.

## Wu-Qin Update Policy

The Wu-Qin row minimizes the same least-squares leave-one-out objective,
`CV(h) = integral(f_hat_h^2) - (2/n) sum_i f_hat_{h,-i}(X_i)`, but it does not
solve that objective at every step. Re-optimization is enabled only when the
Anderson-Darling p-value first reaches `p <= 0.01` and then changes by another
factor of two. A finite target bandwidth is mixed into the live bandwidth with
adjustment rate `0.05`. Particle-wise widths use
`lambda_i = (f_tilde(X_i)/g)^(-alpha)` with the paper's default
`alpha = 0.5`, where `g` is the geometric mean pilot density.
