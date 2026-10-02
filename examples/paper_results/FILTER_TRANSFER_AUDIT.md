# Implemented filter-transfer audit

Audit date: 2026-09-09 (Europe/Luxembourg)

## Scope and source recovered

The August 2026 raw-binomial and compensated-binomial functions are **not** in
the newer top-level `pic_kde_2d/pic_kde_2d/methods/spectral_W.py`. They are
present in the preserved campaign copy
`pic_kde_2d_landau_cv_stage/pic_kde_2d/methods/spectral_W.py`, together with the
matching run and submission scripts. The audit therefore imports that campaign
copy directly and hashes every imported source file in its JSON output. This
locates the implementation but does not recover a historical Git revision;
that remains unknown.

No plasma was advanced. The test uses a deterministic, periodic subcell
quadrature carrying one cosine charge mode. It measures the target Fourier
coefficient at these points in the actual Python chain:

1. CIC deposit followed by the production `W` implementation;
2. the spectral Poisson electric field returned from that density; and
3. production field-side `W` followed by CIC gather to the same particles.

The first two ratios must equal `W`; the complete force ratio relative to CIC
must equal `W^2`. An independent scalar implementation of every analytic
formula is used as the prediction.

Commands:

```text
cd /Users/ayushi.agrawal/Downloads/paper1_source_package/reproducibility
python3 measure_filter_transfers.py --subcells 8
python3 -m unittest -v tests/test_filter_transfer_audit.py
```

Results: 24 filter/mode combinations passed the stage assertions; 3/3
regression tests passed. The largest absolute discrepancies were `4.44e-16`
for the constructed `W`, `6.66e-16` after deposition, `6.66e-16` after the
Poisson field calculation, and `1.78e-15` through the complete gathered force.
The maximum residual after scaling one force trace onto its CIC counterpart was
`1.83e-15`.

Machine-readable results:

* [`filter_transfer_audit.json`](results/filter_transfer_audit/filter_transfer_audit.json)
* [`filter_transfer_audit.csv`](results/filter_transfer_audit/filter_transfer_audit.csv)
* [`measure_filter_transfers.py`](measure_filter_transfers.py)
* [`test_filter_transfer_audit.py`](tests/test_filter_transfer_audit.py)

## Predicted versus measured transfers

The columns `rho` and `E_grid` are ratios to the CIC result at the same mode.
`F_particle` is the complete filtered-chain/CIC particle-force ratio.

### Weak Landau protected mode `(1,0)` on `64 x 16`

| Filter | predicted `W` | measured rho | measured `E_grid` | predicted `W^2` | measured `F_particle` |
|---|---:|---:|---:|---:|---:|
| Gaussian, sigma=1 cell | 0.995192450664 | 0.995192450664 | 0.995192450664 | 0.990408013858 | 0.990408013858 |
| raw binomial, n=4 | 0.990404177838 | 0.990404177838 | 0.990404177838 | 0.980900435480 | 0.980900435480 |
| compensated p=1, n=4 | 0.999942311481 | 0.999942311481 | 0.999942311481 | 0.999884626290 | 0.999884626290 |
| compensated p=2, n=4 | 0.999999722382 | 0.999999722382 | 0.999999722382 | 0.999999444763 | 0.999999444763 |
| compensated p=3, n=4 | 0.999999998831 | 0.999999998831 | 0.999999998831 | 0.999999997661 | 0.999999997661 |
| compensated p=1, n=16 | 0.999230389607 | 0.999230389607 | 0.999230389607 | 0.998461371514 | 0.998461371514 |

### E2 protected longitudinal mode `(12,0)` on `128 x 16`

| Filter | predicted `W` | measured rho | measured `E_grid` | predicted `W^2` | measured `F_particle` |
|---|---:|---:|---:|---:|---:|
| Gaussian, sigma=1 cell | 0.840726281515 | 0.840726281515 | 0.840726281515 | 0.706820680430 | 0.706820680430 |
| raw binomial, n=4 | 0.703200039358 | 0.703200039358 | 0.703200039358 | 0.494490295353 | 0.494490295353 |
| compensated p=1, n=1 | 0.992899377106 | 0.992899377106 | 0.992899377106 | 0.985849173057 | 0.985849173057 |
| compensated p=2, n=2 | 0.997757915078 | 0.997757915078 | 0.997757915078 | 0.995520857101 | 0.995520857101 |
| compensated p=1, n=4 | 0.940221189882 | 0.940221189882 | 0.940221189882 | 0.884015885903 | 0.884015885903 |
| compensated p=2, n=4 | 0.990152772870 | 0.990152772870 | 0.990152772870 | 0.980402513622 | 0.980402513622 |
| compensated p=3, n=4 | 0.998567761909 | 0.998567761909 | 0.998567761909 | 0.997137575124 | 0.997137575124 |

For the disputed E2 `p=3,n=4` case, the measured force attenuation is
`1-W^2 = 0.002862424876`. It differs from the reported protected attenuation
`0.0028624` by `8.69e-6` relatively, well inside the fixed 0.3% acceptance
criterion. This is a transfer-function pass; it does not by itself validate the
separately fitted growth-rate ratio.

### E2 oblique check `(12,1)` on `128 x 16`

| Filter | predicted `W` | measured rho | measured `E_grid` | predicted `W^2` | measured `F_particle` |
|---|---:|---:|---:|---:|---:|
| Gaussian, sigma=1 cell | 0.778337204229 | 0.778337204229 | 0.778337204229 | 0.605808803487 | 0.605808803487 |
| raw binomial, n=4 | 0.602102459345 | 0.602102459345 | 0.602102459345 | 0.362527371550 | 0.362527371550 |
| compensated p=1, n=1 | 0.991461081543 | 0.991461081543 | 0.991461081543 | 0.982995076215 | 0.982995076215 |
| compensated p=2, n=2 | 0.997544157197 | 0.997544157197 | 0.997544157197 | 0.995094345557 | 0.995094345557 |
| compensated p=2, n=4 | 0.989151645032 | 0.989151645032 | 0.989151645032 | 0.978420976869 | 0.978420976869 |
| compensated p=3, n=4 | 0.998500912417 | 0.998500912417 | 0.998500912417 | 0.997004072097 | 0.997004072097 |

This check proves that the production constructor filters both axes with the
requested pass count. For example, raw `n=4` at `(12,1)` has `W=0.60210`, not
the x-only value `0.70320`. The E2 rate launchers explicitly supplied equal x
and y pass counts, although their seeded target mode was `(12,0)`; the y factor
is then exactly one for the target rate but remains active in the density-noise
metric.

## CIC, axes, passes, and units

The absolute CIC deposition and gather factors agreed with each other for all
three modes. With 8-by-8 deterministic subcell quadrature they were:

| Mode | continuum `sinc^2_x sinc^2_y` | deposit measured | gather measured |
|---|---:|---:|---:|
| Landau `(1,0)` | 0.999197067539 | 0.999192069126 | 0.999192069126 |
| E2 `(12,0)` | 0.971417506771 | 0.971242493749 | 0.971242493749 |
| E2 `(12,1)` | 0.958997769551 | 0.958742690685 | 0.958742690685 |

The small difference from the continuum sinc expression is the finite
subcell-quadrature error; it is below the predeclared `3e-4` relative test
tolerance and is not folded into the machine-precision relative-filter test.

Checks established:

* one raw binomial pass is `cos^2(k Delta/2)=1-s`;
* `n` means exactly `n` repetitions in each requested axis;
* every nonzero binomial/compensated case is zero at both grid Nyquist modes;
* Gaussian CLI widths in cells are multiplied by the corresponding physical
  `dx` and `dy` before calling `make_W_gaussian`; and
* the CIC factors occur once at deposit and once at gather, while the extra
  filter occurs once at each location. No additional square was found in the
  implementation and no filter factor was omitted.

## Status and unresolved issue

The recovered Python transfer implementation passes. The large E2 differences
between nearly unit `W^2` and some measured growth-rate ratios cannot be blamed
on an accidental extra/missing filter pass. They must be tested by re-solving
the actual discrete dispersion relation and auditing the E2 equilibrium and
fit protocol, which is the next task.

The C++ post-deposition filters are a different implementation: they apply a
density-side filter only and contain no p=1/2/3 compensated family. They are
therefore not substituted for these Paper 1 Python measurements.

No manuscript text was changed at this stage. The equations already state a
matched `W^2` Python chain; captions/tables will be altered only after the
dispersion and validation audits decide whether the rate claims survive.

