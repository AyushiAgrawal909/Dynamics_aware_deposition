# E2 Monotonic-Erasure Gate Result

## Result Statement

The E2 density selector does not identify a finite dynamically safe smoothing
scale. Both high-ppc-reference MISE and particle-only leave-one-out CV improve
monotonically as `sigma/dx` increases across the scanned range. The curve has
the expected shallow tail toward the uniform-density limit.

This is the relevant statement, not "`sigma_MISE = 12 dx`". The correct
interpretation is:

```text
MISE and CV keep improving through the largest scanned value, sigma = 12 dx.
The density criterion prefers erasing structure rather than preserving the
two-stream mode.
```

## E2 Dynamic Safety References

From the completed E2 growth-rate scan:

```text
Internal selector reference, gamma0 = gamma(sigma=0.25 dx):
sigma = 0.35 dx: gamma/gamma0 = 0.94177  safe under eps=0.06
sigma = 0.45 dx: gamma/gamma0 = 0.86267  boundary
sigma = 0.55 dx: gamma/gamma0 = 0.76238  degraded
sigma = 0.68 dx: gamma/gamma0 = 0.59987  degraded
```

For paper-facing figures normalized to unsmoothed CIC, use the CIC reference:

```text
gamma_CIC = 0.14594390066996013
gamma(0.25 dx) / gamma_CIC = 0.9441
gamma(0.35 dx) / gamma_CIC = 0.8891
```

Therefore, a 10% CIC-referenced damage tolerance requires
`gamma_sigma/gamma_CIC >= 0.90`, and the corrected visual safe marker is
`sigma_safe ~= 0.30 dx` on the current MISE/gamma grid. The old
`sigma_safe ~= 0.35 dx` marker is only valid under the internal
`gamma/gamma(0.25 dx)` convention.

## MISE Curve Shape

MISE against the high-ppc E2 reference is strictly monotone decreasing over the
scanned range:

```text
monotonic_decreasing = True
MISE(12 dx) / MISE(0 dx) = 0.03065
```

Tail checks:

```text
0.35 -> 0.68 dx: MISE ratio = 0.416928, drop = 58.307%
0.68 -> 3.0  dx: MISE ratio = 0.146926, drop = 85.307%
3.0  -> 12.0 dx: MISE ratio = 0.894395, drop = 10.561%
6.0  -> 12.0 dx: MISE ratio = 0.990562, drop = 0.944%
8.0  -> 12.0 dx: MISE ratio = 0.997127, drop = 0.287%
```

This is the shallow-tail-to-uniform behavior: most of the apparent density
benefit comes from erasing particle noise and eventually physical structure;
after `sigma ~= 6-8 dx`, the curve is nearly flat.

## Particle CV Selector

The particle-only selector is least-squares leave-one-out CV on the actual E2
binned estimator:

```text
CIC particle binning -> spectral Gaussian W -> LOO-CV self-term correction
```

It also selects the largest scanned value:

```text
sigma_CV = 12 dx
classification = cv_selects_dynamically_degraded_sigma
```

Thus the CV selector does not rescue the dynamically safe bandwidth. It fails
in the same direction as high-ppc-reference MISE.

## Headline Figure

The redesigned paper figure is:

```text
/mnt/tier2/users/u103576/logs/pic_kde_2d_figures/E2_mise_gamma_damage_logx_with_cv.png
```

It plots:

- log x-axis: `sigma/dx`, so the `sigma <= 1 dx` crossing region is visible
- left y-axis: normalized density MISE, log scale
- right y-axis: growth-rate damage `1 - gamma_sigma/gamma0`
- vertical marker: `sigma_safe ~= 0.30 dx` for CIC-referenced 10% damage
- vertical marker: `sigma_CV = 12 dx`
- shaded region: dynamically degraded bandwidths,
  `gamma_sigma/gamma_CIC < 0.90`

The figure shows the core result directly: density error keeps sliding down
while growth-rate damage rises, and both density selectors point into the
dynamically degraded region.
