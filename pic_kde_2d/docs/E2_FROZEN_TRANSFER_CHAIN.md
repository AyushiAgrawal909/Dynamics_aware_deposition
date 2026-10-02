# E2 Frozen Transfer Chain

## Scope

This document freezes the method used for the completed revised E2 two-stream
results, including jobs `4622108` and `4622204`. Those jobs launched
`scripts/run_E2_revised.py` from the `pic_kde_2d` Python benchmark tree. They
did not launch the HEXAPIC production executable.

For each `sigma_dx` and seed, the runner constructs:

```python
grid = Grid2D(Nx=128, Ny=16, Lx=64.0, Ly=16.0)
sigma_x = sigma_dx * grid.dx
sigma_y = sigma_dx * grid.dy
deposit, gather = make_spectral_W_gaussian_methods(grid, sigma_x, sigma_y)
history = run_simulation(..., deposit, gather, ...)
```

## Executed Chain

One force evaluation in the E2 velocity-Verlet loop is:

```text
particle charge
  -> bilinear CIC deposit onto periodic nodes
  -> FFT Gaussian post-filter on rho
  -> FFT Poisson solve and spectral gradient
  -> FFT Gaussian post-filter on Ex and Ey
  -> bilinear CIC gather to particles
  -> velocity kick
```

The second half of each velocity-Verlet step repeats the same chain after the
particle drift.

## Deposit

Deposition is **CIC**, not direct Gaussian particle deposition.
`deposit_cic_2d` places each particle on four adjacent 2D grid nodes with
bilinear weights. The Gaussian operation is applied only after that deposit:

```python
rho_cic = deposit_cic_2d(...)
rho = irfft2(rfft2(rho_cic) * W)
```

The filter is a fixed, spatially uniform **grid post-filter**:

```text
W(kx, ky) = exp[-0.5 * (kx^2 sigma_x^2 + ky^2 sigma_y^2)]
```

It is not a Gaussian particle shape and not an adaptive or learned deposition
operator. `ml_deposition_apply` is not called by this E2 code path.

## Field Solve

The E2 Poisson operator is **not HYPRE/PFMG** and is **not a five-point
finite-difference solve**. `Grid2D.poisson_solve` uses NumPy FFTs:

```text
rho_hat = rfft2(rho)
phi_hat = rho_hat / (KX^2 + KY^2), with the zero mode set to zero
Ex_hat  = -i KX phi_hat
Ey_hat  = -i KY phi_hat
```

Thus E2 uses the continuous-symbol periodic spectral Green function and
spectral gradient on the resolved FFT grid:

```text
G_E(k) = -i k / |k|^2,  k != 0.
```

No finite-difference Laplacian symbol or finite-difference gradient symbol
appears in the executed force chain.

## Gather And Matching

Gather is matched to the executed deposited operator in two parts:

1. The particle-to-grid and grid-to-particle shapes are both bilinear CIC.
2. The same fixed Gaussian grid filter `W` is applied once to deposited
   density and once to the solved grid electric field before CIC gather.

The gather closure is:

```python
Ex_gather = irfft2(rfft2(Ex) * W)
Ey_gather = irfft2(rfft2(Ey) * W)
E_particle = gather_cic_2d(Ex_gather, Ey_gather, ...)
```

This is matched **CIC plus symmetric grid filtering**, not matched direct
Gaussian particle deposit/gather.

## Fourier Transfer Statement

Ignoring sampling aliases for the moment, the force response for a resolved
mode contains:

```text
CIC deposit shape * W * spectral Poisson/gradient * W * CIC gather shape
```

or schematically:

```text
H_E2(k) = S_CIC(k)^2 * W(k)^2 * [-i k / |k|^2].
```

Because CIC deposition occurs before the post-filter, the complete sampled
analysis can include CIC alias branches subsequently multiplied by the grid
filter evaluated on resolved modes. Alias analysis appropriate to a CIC plus
grid-post-filter method is therefore relevant.

## Gate For Subsequent Theory

E2 did use **CIC plus a grid post-filter**, so it must not be analyzed as a
direct Gaussian particle-shape experiment. However, it did **not** use
`ml_deposition_apply`, HYPRE/PFMG, a five-point Laplacian, or a
finite-difference field gradient.

Any Step 5 theory intended to reproduce these E2 results must use:

```text
CIC deposit -> Gaussian spectral grid filter -> spectral FFT Poisson/gradient
-> same Gaussian spectral grid filter -> CIC gather.
```

If a separate production HEXAPIC experiment uses
`CIC -> ml_deposition_apply -> HYPRE/PFMG -> finite-difference gradient/gather`,
that is a different transfer chain and requires a separate frozen document.
