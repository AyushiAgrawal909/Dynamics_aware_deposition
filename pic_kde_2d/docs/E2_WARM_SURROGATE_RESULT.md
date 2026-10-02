# E2 Warm Surrogate Result

## Inputs Checked

The warm two-stream surrogate used the frozen E2 chain and parameters:

- `Nx = 128`, `Lx = 64`, `dx = 0.5`
- `kx_mode = 12`, `k = 1.1780972450961724`, `k dx = 0.5890486225480862`
- `v0 = 0.8`, `v_th = 0.05`
- `S_CIC = sinc(k dx / 2)^2 = 0.971417506770955`
- matched deposit/gather CIC factor `S_CIC^2 = sinc(k dx / 2)^4 = 0.9436519724610983`
- `kx_mode / Nx = 0.09375`, so the resolved mode is below the alias-danger range.

The cold-equivalent near-threshold coupling is:

```text
A0 = (omega_p / (k v0))^2 * sinc(k dx / 2)^4 = 1.0623548310687245
```

This confirms that the earlier cold-surrogate inputs were read correctly.

## Warm Model

The tested warm dispersion was:

```text
1 + T_eff^2 omega_p^2 / (2 k^2 v_th^2)
    * sum_b [1 + zeta_b Z(zeta_b)] = 0

zeta_+ = (omega - k v0) / (sqrt(2) k v_th)
zeta_- = (omega + k v0) / (sqrt(2) k v_th)
T_eff^2 = sinc(k dx / 2)^4 * exp(-(k sigma)^2)
```

The script solves the symmetric purely growing branch `omega = i gamma`.

## Result

Output directory:

```text
/mnt/tier2/users/u103576/logs/pic_kde_2d_e2warm_surrogate
```

Local copy:

```text
results/part2/warm_surrogate/
```

Key numbers:

```text
gamma_ref_pred                  = 0.12099094557044046
prekill_rmse (sigma <= 0.8 dx)  = 0.5302644444994805
predicted stabilization sigma   = 0.4556985331214416 dx
measured stabilization sigma    ~= 1.0 dx
stabilization relative error    = 54.4%
```

Measured vs predicted growth-ratio curve:

```text
sigma/dx   measured gamma/gamma0   warm prediction
0.25       1.0000                  1.0000
0.35       0.9418                  0.7671
0.45       0.8627                  0.2036
0.55       0.7624                  0.0000
0.68       0.5999                  0.0000
1.00       0.0439                  0.0000
```

## Decision

The warm surrogate fails both validation checks:

- pre-stabilization RMSE is far above the 15% target;
- predicted stabilization is still near `0.46 dx`, not the measured `~1.0 dx`.

Because the executed E2 setup has `v0/v_th = 16`, it is effectively close to
the cold limit for the instability branch. The warm correction does not explain
the gradual measured roll-off.

Do not add more analytic layers before the paper result. Use the measured
`gamma(sigma)` curve as the dynamics constraint for the selector. The core
result remains intact: density selectors choose sigmas that erase the unstable
mode, while the dynamics-safe sigma must be read from a growth-rate constraint.
