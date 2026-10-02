# E2 Parameter Block

## Source

This block records the configuration that produced the completed revised E2
two-stream scan in Meluxina job `4622204`, from:

- `scripts/submit_E2_revised_paired.sbatch`
- `scripts/run_E2_revised.py`
- `E2_revised_result_e2paired_extended30_v2.json`

## Executed Configuration

| Quantity | Value |
| --- | ---: |
| Problem | 2D two-stream, mode `(m_x, m_y) = (12, 0)` |
| Grid | `Nx = 128`, `Ny = 16` |
| Domain | `Lx = 64.0`, `Ly = 16.0` |
| Cell spacing | `dx = 0.5`, `dy = 1.0` |
| Physical wavenumber | `k_x = 2*pi*12/64 = 1.1780972451` |
| Grid-scaled wavenumber | `k_x*dx = 2*pi*12/128 = 0.5890486225` |
| Beam drift magnitude | `v0 = 0.8` for beams at `+/- v0` |
| Thermal velocity | `v_th = 0.05` |
| Thermal character | warm two-stream, but nearly cold: `v0/v_th = 16` |
| Initial displacement amplitude | `alpha = 0.001` |
| Low-resolution particles per cell | `ppc = 32` (`65,536` particles) |
| Seeds | `0..29` (`30` paired seeds) |
| Time step | `dt = 0.05` |
| Run length | `n_steps = 300`, `t_end = 15.0` |
| Diagnostic stride | `2` steps (`delta t_diag = 0.1`) |
| Growth fit window | `t in [0.5, 4.0]`, `36` diagnostic points |

## Method And Reference Definition

E2 used `CIC deposit -> Gaussian FFT grid post-filter W on rho -> spectral
Poisson/gradient -> same W on E -> CIC gather`. It did not use a direct
Gaussian particle shape and did not use `ml_deposition_apply`.

The original E2 dynamic growth reference was the smallest tested filter,
`sigma_ref = 0.25 dx`, evaluated by fitting the ensemble-averaged
`|E_x(m_x=12,m_y=0,t)|` over `t in [0.5, 4.0]`. For job `4622204`:

```text
gamma_ref(sigma=0.25 dx) = 0.13778740644710036
bootstrap 95% CI          = [0.11454282576133261, 0.16105930583726985]
```

The already established dynamic safety boundary is:

```text
Internal selector convention:
sigma = 0.35 dx: gamma/gamma_ref = 0.94177, CI [0.92439, 0.95411]  safe
sigma = 0.45 dx: gamma/gamma_ref = 0.86267, CI [0.82192, 0.89173]  boundary
sigma = 0.55 dx: gamma/gamma_ref = 0.76238, CI [0.69275, 0.81226]  degraded
sigma = 0.68 dx: gamma/gamma_ref = 0.59987, CI [0.48266, 0.68237]  degraded
```

Paper-facing figures use the unsmoothed CIC reference instead:

```text
gamma_CIC = 0.14594390066996013
gamma(0.25 dx) / gamma_CIC = 0.9441
gamma(0.35 dx) / gamma_CIC = 0.8891
```

Thus `sigma=0.35 dx` does not pass a central 10% CIC-referenced damage
tolerance. The corrected CIC-referenced selector chooses `sigma=0.30 dx`, with
`gamma_sigma/gamma_CIC = 0.9166` and noise reduction `1.56x`.

## Density-MISE Gate Protocol

The density selector must be tested independently of the live-force damage.
For the MISE scan:

1. Evolve both the `ppc=32` sample and a high-ppc reference using the executed
   E2 reference dynamics, `sigma_live = 0.25 dx`.
2. At diagnostic times in the growth-fit interval `t in [0.5, 4.0]`, save
   `rho_ref` as the unsmoothed CIC density from the high-ppc trajectory.
3. At the same low-ppc particle states, apply each candidate Gaussian `W` only
   as a diagnostic density post-filter, not as the live force method.
4. Measure `MISE(sigma) = mean((rho_sigma - rho_ref)^2)` over cells, times,
   and paired seeds.

This protocol asks the gate question directly: does a density-optimal
post-filter choose a sigma that E2 dynamics has already shown to be unsafe?
