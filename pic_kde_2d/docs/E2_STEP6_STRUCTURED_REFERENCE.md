# E2 Step 6 Option A: Nonlinear Structured Reference

## Purpose

This is the direct flat-field objection test. The E2 MISE/CV selectors were
rerun on the same near-marginal two-stream setup, but the reference density was
taken from a later nonlinear-growth window:

```text
time window: t in [20, 30]
live dynamics: sigma_live = 0.25 dx
low ppc: 32
reference ppc: 512
seeds: 5
diagnostic samples: 101 times per seed
```

The reference is no longer the nearly-flat linear-window density. It contains
finite-amplitude structure at the physical two-stream mode.

## Gate Ratios

Using the corrected CIC-referenced E2 dynamic safety marker:

```text
sigma_safe ~= 0.30 dx
```

The structured-reference selectors give:

```text
sigma_MISE = 12.0 dx
sigma_MISE / sigma_safe = 40.00

sigma_CV = 0.55 dx
sigma_CV / sigma_safe = 1.83
```

## Interpretation

The flat-field objection is strongly defeated for high-ppc-reference MISE:
even with nonlinear density structure present, MISE continues to improve
monotonically through the largest scanned bandwidth. The density error is still
dominated by noise reduction enough that the selector sacrifices physical
two-stream structure.

Particle-only CV is more sensitive to the nonlinear structure. It no longer
chooses the extreme tail; it selects `0.55 dx`. However, `0.55 dx` is already
in the dynamically degraded region from the E2 growth-rate scan:

```text
sigma = 0.55 dx: gamma_sigma/gamma0 = 0.76238
sigma = 0.55 dx: gamma_sigma/gamma_CIC = 0.7200
```

So CV does not fully pass the structured-reference defense. It narrows the gap,
but still points beyond the dynamically safe value.

## Result For The Paper

The precise statement should be:

```text
On a nonlinear structured reference, reference-MISE still monotonically rewards
mode erasure. Particle CV becomes less extreme but remains dynamically unsafe.
```

This is stronger and more nuanced than the linear-window result:

- Linear window: both MISE and CV select the extreme erasure tail.
- Nonlinear window: MISE still selects the erasure tail; CV shifts near the
  boundary but remains outside the safe region.

## Figure

```text
/mnt/tier2/users/u103576/logs/pic_kde_2d_figures/E2_nonlinear_mise_gamma_damage_logx_with_cv.png
```
