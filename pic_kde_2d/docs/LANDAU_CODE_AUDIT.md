# Landau code audit for the sigma-refinement run

## Frozen chain

The refinement uses `Nx=64`, `Ny=16`, `Lx=2*pi/0.4`, `Ly=8`, `alpha=0.01`,
`dt=0.05`, and 10 paired seeds at PPC 32 and 1024. Each width is expressed in
cell units in its own direction: `sigma_x=sigma_dx*dx` and
`sigma_y=sigma_dx*dy`. The protected mode is `(m_x,m_y)=(1,0)`.

The force chain is:

1. CIC particle deposition.
2. Gaussian `W(k)` applied to charge density.
3. Spectral periodic Poisson solve, `-laplacian(phi)=rho` and `E=-grad(phi)`.
4. The identical real, zero-phase `W(k)` applied to both field components.
5. CIC gather and a velocity-Verlet electron push with `q/m=-1`.

Thus the sigma-dependent force response at the protected mode contains `W^2`.
The numerical audit measures the expected response to within `3.2e-15`.

## Initialization

The displacement is `xi=alpha*k*sin(k.x)/|k|^2`, so the electron charge
perturbation is `+alpha*cos(k.x)` to first order. The velocity perturbation is
chosen so the charge/current moments satisfy continuity for complex frequency
`omega=omega_r+i*gamma`:

`u_parallel = alpha*(gamma*sin(k.x)-omega_r*cos(k.x))/|k|`.

The measured current/density relation agrees with `omega/k` to `1.2e-5` in the
finite-amplitude CIC audit. The Poisson mode relation agrees with `-i/k` to
`1.8e-16`.

This initialization is moment-matched, but it is not the exact
velocity-dependent Vlasov eigenfunction. Calling it an exact kinetic eigenmode
would be an overclaim. Its non-eigenfunction component produces an early
projection transient; damping fits therefore begin at `t=5`.

## Fitting correction

The historical `fit_gamma_ensemble` is unsuitable for this refinement because
it can select disjoint above-floor points after recurrence and can silently
fall back to an uncensored fixed window. The refinement instead:

- coherently averages the complex Fourier coefficient over paired seeds;
- forms a geometric envelope over one Langmuir period, removing projection
  beating without changing an exponential damping slope;
- estimates the floor from the quietest rolling-median window in `t=[70,90]`;
- fits one contiguous interval beginning at `t=5`;
- stops at the first five-point persistent crossing below `5*floor`;
- never permits a recurrence point to re-enter the fit;
- reports R-squared, fitted e-foldings, fit duration, SNR, and bootstrap quality;
- bootstraps `gamma_sigma/gamma_CIC` using common seed indices.

A fit is publication-eligible only if it is damped, spans at least five time
units and 30 diagnostics, resolves at least 0.5 fitted e-foldings, and has
`R^2>=0.75`. Open/failed points remain in the output as censored diagnostics but
cannot set `sigma_R`. In addition, a PPC can set `sigma_R` only if its CIC
baseline agrees with kinetic theory within 15%; this prevents a ratio of two
noise-biased fits from masquerading as a valid rate calibration.

For each width the reducer records the matched protected-mode transfer
`W^2=exp[-(2*pi*m*sigma_cells/Nx)^2]`. On a theory-valid PPC it calibrates the
local rate-sensitivity coefficient through the origin,
`|gamma_sigma/gamma_CIC-1| = C_R*(1-W^2)`, using quality-passing points with
less than 25% force attenuation. This coefficient is deliberately unavailable
when the CIC baseline fails the measurability/theory gate.

That local coefficient is retained as an empirical diagnostic, not as the C4
rate model. The zero-parameter kinetic dispersion with the matched CIC and
Gaussian coupling predicts a 10% crossing at `2.453 dx`. Its predictions
`1.066` at `2 dx` and `1.151` at `3 dx` both lie inside the paired measured
intervals. The earlier restricted estimate `C_R=0.640` is underpowered in this
range and must not replace the dispersion calculation; the unrestricted linear
fit is also invalid once the response leaves its linear regime.

## Audit verdict

The signs, normalization, Poisson operator, matched filter, and charge/current
phase are correct for the stated normalized model. The remaining limitation is
initial-condition projection, not a force-chain bug. The new censored fit is
required to prevent that limitation and the low-PPC noise floor from being
misreported as a physical smoothing response.
