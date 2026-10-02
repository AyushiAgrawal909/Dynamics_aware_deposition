# Per-seed quality checks

All values use the frozen window [24.32, 69.16]. The e-folding and single-seed SNR columns are diagnostics; the registered arm-level duration and SNR gates use the ensemble mean-magnitude histories. A failed diagnostic is reported but does not create an unregistered seed-rejection rule.

| Seed | Arm | gamma | R-squared | E-foldings | Start SNR | max delta-f ratio | Init arrays | Required gates | Diagnostics |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 200 | cic | 0.116688 | 0.997772 | 5.2323 | 11.688 | 0.000516822 | MATCH | PASS | PASS |
| 200 | q050_x000 | 0.111075 | 0.996547 | 4.9806 | 8.798 | 0.000338332 | MATCH | PASS | PASS |
| 201 | cic | 0.096461 | 0.991869 | 4.3253 | 5.343 | 0.00012974 | MATCH | PASS | PASS |
| 201 | q050_x000 | 0.089434 | 0.986270 | 4.0102 | 4.740 | 7.71156e-05 | MATCH | PASS | FLAG |
| 202 | cic | 0.122042 | 0.999138 | 5.4724 | 22.950 | 0.000999192 | MATCH | PASS | PASS |
| 202 | q050_x000 | 0.116563 | 0.998799 | 5.2267 | 22.473 | 0.000677627 | MATCH | PASS | PASS |
| 203 | cic | 0.093838 | 0.994787 | 4.2077 | 12.189 | 0.000198454 | MATCH | PASS | PASS |
| 203 | q050_x000 | 0.086995 | 0.991676 | 3.9009 | 9.229 | 0.000124895 | MATCH | PASS | PASS |
| 204 | cic | 0.149143 | 0.999616 | 6.6876 | 32.379 | 0.00661533 | MATCH | PASS | PASS |
| 204 | q050_x000 | 0.144595 | 0.999528 | 6.4836 | 28.567 | 0.00475346 | MATCH | PASS | PASS |
| 205 | cic | 0.123425 | 0.999009 | 5.5344 | 18.608 | 0.00125254 | MATCH | PASS | PASS |
| 205 | q050_x000 | 0.117995 | 0.998682 | 5.2909 | 17.666 | 0.000858574 | MATCH | PASS | PASS |
| 206 | cic | 0.163825 | 0.999847 | 7.3459 | 48.057 | 0.0159612 | MATCH | PASS | PASS |
| 206 | q050_x000 | 0.159643 | 0.999817 | 7.1584 | 41.427 | 0.0117833 | MATCH | PASS | PASS |
| 207 | cic | 0.148642 | 0.999762 | 6.6651 | 33.002 | 0.00611657 | MATCH | PASS | PASS |
| 207 | q050_x000 | 0.144081 | 0.999704 | 6.4606 | 29.131 | 0.00437966 | MATCH | PASS | PASS |
| 208 | cic | 0.125085 | 0.999547 | 5.6088 | 21.246 | 0.00141354 | MATCH | PASS | PASS |
| 208 | q050_x000 | 0.119738 | 0.999361 | 5.3690 | 18.092 | 0.000973069 | MATCH | PASS | PASS |
| 209 | cic | 0.105359 | 0.995554 | 4.7243 | 10.213 | 0.000288623 | MATCH | PASS | PASS |
| 209 | q050_x000 | 0.099189 | 0.993086 | 4.4476 | 7.742 | 0.000183911 | MATCH | PASS | PASS |
