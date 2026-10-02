# CV Strawman Audit

## Question

Does `scripts/run_E2_cv_selector.py` implement the same kind of selector as the
Wu-Qin KDE PIC method?

## Answer

It implements the same *least-squares leave-one-out cross-validation objective*
used by Wu and Qin to estimate/minimize MISE, but adapted to the estimator used
in this project:

```text
CIC particle binning -> spectral Gaussian W post-filter
```

It is therefore a faithful "MISE/LOO-CV selector" strawman for the deposition
chain we are testing, but it is not a literal reproduction of every Wu-Qin
implementation detail.

The unguarded loss and the Wu-Qin temporal controller are now reported
separately. A minimum at the upper search boundary is emitted as
`right_censored`; it is never promoted to a finite optimum.

## Match To Wu-Qin

Wu and Qin define the MISE and then replace the unknown density term with a
leave-one-out estimator. Their CV objective has the standard least-squares form:

```text
CV(h) = integral f_hat_h(x)^2 dx - 2/n * sum_i f_hat_{h,-i}(X_i)
```

This is exactly the structure implemented in:

```text
scripts/run_E2_cv_selector.py
```

Code mapping:

```text
term_integrated_square = integral rho_smooth^2 dx dy
raw_at_particles_mean  = mean_i rho_smooth(X_i)
loo_mean               = self-term-corrected mean_i rho_{-i}(X_i)
score                  = term_integrated_square - 2 * loo_mean
```

Relevant code:

```text
cv_score_for_density(...):
    rho_smooth = rho_prob_cic if W is None else apply_W_to_grid(...)
    term_integrated_square = sum(rho_smooth^2) * dx * dy
    raw_at_particles_mean = sum(rho_smooth * rho_prob_cic) * dx * dy
    loo_mean = (raw_at_particles_mean - k_self / n) * n / (n - 1)
    return term_integrated_square - 2 * loo_mean
```

This is a binned/grid form of least-squares LOO-CV. The self-term subtraction is
the leave-one-out correction.

## Differences From Wu-Qin

The current strawman differs in these implementation details:

1. Wu-Qin use a periodic KDE kernel, described through wrapped normal/von Mises
   kernels. We use the actual project estimator: CIC binning followed by a
   spectral Gaussian grid filter.

2. Wu-Qin discuss adaptive particle-wise widths and a width-update strategy. The
   unguarded strawman uses one fixed global sigma, because the paper question
   here is whether density-selected smoothing is dynamically safe. The separate
   `Wu-Qin AD-gated CV` row implements their p-value update policy.

3. Wu-Qin's presentation is 1D circular KDE. Our test is 2D periodic binned KDE.
   The least-squares CV objective is the same; the kernel/operator differs.

4. For a binned CIC estimator, the self kernel depends weakly on sub-cell
   particle offset. The script averages a small set of offsets to estimate the
   self term. This is a controlled approximation specific to CIC binning.

## How To Phrase This In The Paper

Use:

```text
We compare against a Wu-Qin-style least-squares leave-one-out CV selector,
implemented for the same CIC+W estimator used in our simulations.
```

Avoid:

```text
We exactly reproduce Wu and Qin's full KDE algorithm.
```

The stronger and defensible claim is:

```text
The same MISE/leave-one-out-CV principle used by KDE-PIC selects dynamically
unsafe smoothing on the near-marginal two-stream benchmark.
```

## Consequence For The Result

The strawman is close enough to test the scientific principle: selecting sigma
by density MISE/LOO-CV alone. It is not close enough to claim that every
implementation detail of Wu-Qin's adaptive KDE code fails.

If a reviewer demands a literal reproduction, the next step is a separate
direct von-Mises LOO-CV implementation. That would be a robustness check, not a
replacement for the current result, because the current result already applies
the MISE/LOO-CV objective to the exact deposition chain evaluated here.

## Exact Wu-Qin Controls

- Kernel parameter: bandwidth `h`; for the periodic von Mises kernel,
  concentration `kappa` is used and `1/kappa` is analogous to squared wrapped
  Gaussian width.
- Loss: least-squares leave-one-out CV,
  `integral(f_hat_h^2) - 2 mean_i(f_hat_{h,-i}(X_i))`.
- Local adaptation:
  `lambda_i = (f_tilde(X_i)/g)^(-alpha)`, with default `alpha=0.5`.
- Time update: do not optimize near uniformity; enable optimization at
  Anderson-Darling `p <= 0.01`, require another factor-of-two p-value decrease
  before the next target, and relax with adjustment rate `0.05`.

The implementation lives in
`pic_kde_2d/methods/wu_qin_bandwidth.py`. The amplitude-halving audit is
`scripts/run_wu_qin_alpha_halving.py`.
