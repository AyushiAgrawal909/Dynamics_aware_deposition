# Recovered CIC per-seed quality verification

Frozen fit window: `[24.32,69.16]`. No seed was removed. `N_e` below is each
seed's own fitted rate multiplied by the fixed window duration 44.84. The
registered duration and fit-start SNR gates are ensemble-amplitude gates; the
per-seed `N_e` and SNR columns are additional diagnostics using the same
numerical thresholds, not replacements for the registered ensemble gates.

| Seed | Individual gamma | R-squared | Individual N_e | Individual SNR diagnostic | max delta-f ratio | Complete | R-squared gate | delta-f gate | positive rate | N_e >= 2 diagnostic | SNR >= 5 diagnostic |
|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 200 | 0.116688 | 0.997772 | 5.2323 | 11.6876 | 0.0005168 | pass | pass | pass | pass | pass | pass |
| 201 | 0.096461 | 0.991869 | 4.3253 | 5.3430 | 0.0001297 | pass | pass | pass | pass | pass | pass |
| 202 | 0.122042 | 0.999138 | 5.4724 | 22.9502 | 0.0009992 | pass | pass | pass | pass | pass | pass |
| 203 | 0.093838 | 0.994787 | 4.2077 | 12.1893 | 0.0001985 | pass | pass | pass | pass | pass | pass |
| 204 | 0.149143 | 0.999616 | 6.6876 | 32.3791 | 0.0066153 | pass | pass | pass | pass | pass | pass |
| 205 | 0.123425 | 0.999009 | 5.5344 | 18.6075 | 0.0012525 | pass | pass | pass | pass | pass | pass |
| 206 | 0.163825 | 0.999847 | 7.3459 | 48.0570 | 0.0159612 | pass | pass | pass | pass | pass | pass |
| 207 | 0.148642 | 0.999762 | 6.6651 | 33.0022 | 0.0061166 | pass | pass | pass | pass | pass | pass |
| 208 | 0.125085 | 0.999547 | 5.6088 | 21.2464 | 0.0014135 | pass | pass | pass | pass | pass | pass |
| 209 | 0.105359 | 0.995554 | 4.7243 | 10.2135 | 0.0002886 | pass | pass | pass | pass | pass | pass |

Registered ensemble-amplitude results:

- fitted rate: `0.1447458990`;
- fitted e-foldings: `6.4904061`;
- fit R-squared: `0.9994114`;
- fit-start signal/floor ratio: `22.2489`;
- all ten individual R-squared gates pass;
- all ten individual delta-f gates pass;
- mean individual rate: `0.1244508093`, paired-seed bootstrap 95% interval
  `[0.1111166540,0.1383865170]`.

The `6.4904` value is only the ensemble-amplitude result. Individual fitted
durations range from `4.2077` to `7.3459` e-foldings and are reported by seed
above and at full precision in `reuse_audit.csv`.
