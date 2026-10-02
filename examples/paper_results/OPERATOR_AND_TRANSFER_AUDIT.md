# Exact filter and operator correction audit

Audit date: 2026-09-12. No particle history was rerun for this audit, and the
authoritative manuscript was not edited.

## Result

The implemented filter transfers are internally consistent. The Python
production chain applies the requested filter once after CIC deposition and
once at CIC gather, so a mode sees `W` in the saved pre-gather grid field and
`W^2` in the particle force. The 24 controlled single-mode cases in
`results/filter_transfer_audit/` agree with the independently evaluated
analytic transfers to round-off.

The remaining defect is representational, not a particle-history defect:
`sigma_eq=sqrt(n/2)` cells matches an `n`-pass raw binomial filter only in the
small-`k*dx` expansion. It is not a global identity and must not be used in
place of the exact binomial transfer at a protected mode.

## Exact definitions

With `theta=k*dx`, `s=sin(theta/2)^2`, and `n` repeated passes,

```text
W_raw(theta) = (1-s)^n = cos(theta/2)^(2n)
W_gaussian(theta; sigma_cells) = exp[-(theta*sigma_cells)^2/2]
sigma_eq = sqrt(n/2)       (leading-order correspondence only)
```

For the compensated family,

```text
W_p,n(theta) = (1-s)^n * sum_{j=0}^p C(n+j-1,j) s^j .
```

Every quantitative protected-mode calculation must use these exact `W`
values and the actual filtered axes. `W^2` denotes the matched
deposit--gather force response; `1-W^2` is the protected attenuation.

## Quantified exact-versus-Gaussian differences

Source:
`results/operator_transfer_correction_20260911/raw_binomial_vs_low_k_gaussian_transfer.csv`.

| Mode/filter comparison | exact raw-binomial `W` | low-k-matched Gaussian `W` | relative one-way error | relative round-trip error |
|---|---:|---:|---:|---:|
| Landau `(1,0)`, `Nx=64`, `n=12` | 0.9714878893 | 0.9714991776 | -0.00116% | -0.00232% |
| E2 `(12,0)`, `Nx=128`, `n=4` | 0.7032000394 | 0.7068206804 | -0.51224% | -1.02186% |
| E2 `(12,1)`, isotropic `n=4` | 0.6021024593 | 0.6058088035 | -0.61180% | -1.21986% |

The difference is negligible for the weak-Landau low mode but material enough
at E2 that “equivalent to five or six digits” is false. The exact pass-budget
helper was already used for the integer decisions: the Landau ceiling remains
12 passes and the E2 longitudinal ceiling remains zero. No rate/noise history
or stored pass count changes.

The compensated E2 `p=3,n=4` transfer is also confirmed directly:
`W^2=0.997137575124` at `(12,0)` and `0.997004072097` at `(12,1)` when both axes
are filtered. These are different modes, not inconsistent measurements.

## Spectral versus finite-difference operators

The audited Python campaigns use

```text
phi_hat(k) = rho_hat(k)/(epsilon0*|k|^2),  k != 0
E_hat(k)   = -i*k*phi_hat(k),
```

with the neutral zero mode removed. Therefore their field/noise predictions
must use exact Fourier `k` and `|k|^2`. The five-point positive Laplacian
`Lambda_h` and centered derivative `D_h` describe the separate C++/HYPRE
finite-difference path only. The size of the otherwise introduced operator
difference is documented in
`results/operator_transfer_correction_20260911/spectral_vs_finite_difference_symbols.csv`:
the finite-difference/spectral field-amplitude ratios are 0.99920 for weak
Landau `(1,0)`, 0.97092 for E2 `(12,0)`, and 0.97261 for E2 `(12,1)`.

The production E2 dispersion/root reducer was inspected and already uses the
spectral Green function and spectral derivative. Consequently the frozen
widths, predicted scan roots and rate reductions do not require recomputation.
Only the manuscript force-noise equation and the language surrounding the
binomial/Gaussian comparison require correction.

## Impact classification

| Issue | Classification | Required action |
|---|---|---|
| Global equality of raw binomial and `sigma_eq` Gaussian | wording/table metadata | call it a low-`k` match and retain exact `W` values |
| Existing raw-binomial and compensated histories | no simulation effect | retain; implementation used exact transfers |
| Landau/E2 integer pass budgets | no numerical change | retain 12 and 0; helper already exact |
| Python force-noise equation written with `D_h,Lambda_h` | equation wording | replace with spectral `k/|k|^2` symbols |
| E2 dispersion roots and frozen widths | no recomputation | audited reducer was already spectral |
| C++ HYPRE conservation tests | no change | keep as a separately labelled finite-difference chain |

## Reproduction

```bash
python3 paper1_source_package/reproducibility/audit_exact_filter_and_spectral_operators.py \
  --output /new/nonexistent/operator-audit-directory
python3 paper1_source_package/reproducibility/measure_filter_transfers.py \
  --subcells 8 --output /new/nonexistent/transfer-directory
```

The machine-readable audit is
`results/operator_transfer_correction_20260911/operator_and_transfer_audit.json`.
