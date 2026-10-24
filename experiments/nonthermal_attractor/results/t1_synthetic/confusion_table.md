# T1 synthetic suite: expected vs returned verdicts

19/25 match; outcomes: **wrong**: 1, conservative miss: 5, match: 19.

5 seeds per case, 64 cells, quick=False. 'Expected' is the verdict for data that resolve the relevant feature; a *conservative miss* (INCONCLUSIVE, DEGENERATE or no claim where a positive verdict was expected) is the designed behaviour when the feature is not supported by counts. A FALSE POSITIVE would be a failure of the analysis.

| Case | ppc (finest) | Expected (resolved data) | Returned | Outcome |
|---|---|---|---|---|
| S1 | 512 | `NO_TAIL` | `NO_TAIL` | match |
| S2 | 512 | `IDENTIFIED_AND_FORECAST_WINS` | `S2_UNIDENTIFIABLE_DEGENERATE` | conservative miss |
| S3 | 512 | `UNIDENTIFIABLE_DEGENERATE` | `UNIDENTIFIABLE_DEGENERATE` | match |
| S4 | 512 | `NO_CLAIM` | `NO_CLAIM` | match |
| S4b | 512 | `NO_CLAIM` | `NO_CLAIM` | match |
| S2-gate | 512 (3 levels to 128) | `PASS` | `INCONCLUSIVE` | conservative miss |
| S5 | 512 (3 levels to 128) | `FAIL_REGULATOR_DEPENDENT` | `INCONCLUSIVE` | conservative miss |
| S6 | 512 (3 levels to 128) | `PASS_AFTER_ONSET_ALIGNMENT+lnN+IDENTIFIED` | `INCONCLUSIVE+lnN+IDENTIFIED` | conservative miss |
| S1 | 2048 | `NO_TAIL` | `NO_TAIL` | match |
| S2 | 2048 | `IDENTIFIED_AND_FORECAST_WINS` | `IDENTIFIED_AND_FORECAST_WINS` | match |
| S3 | 2048 | `UNIDENTIFIABLE_DEGENERATE` | `UNIDENTIFIABLE_DEGENERATE` | match |
| S4 | 2048 | `NO_CLAIM` | `NO_CLAIM` | match |
| S4b | 2048 | `NO_CLAIM` | `NO_CLAIM` | match |
| S2-gate | 2048 (3 levels to 512) | `PASS` | `PASS` | match |
| S5 | 2048 (3 levels to 512) | `FAIL_REGULATOR_DEPENDENT` | `INCONCLUSIVE` | conservative miss |
| S6 | 2048 (3 levels to 512) | `PASS_AFTER_ONSET_ALIGNMENT+lnN+IDENTIFIED` | `PASS+lnN+IDENTIFIED` | **wrong** |
| S1 | 8192 | `NO_TAIL` | `NO_TAIL` | match |
| S2 | 8192 | `IDENTIFIED_AND_FORECAST_WINS` | `IDENTIFIED_AND_FORECAST_WINS` | match |
| S3 | 8192 | `UNIDENTIFIABLE_DEGENERATE` | `UNIDENTIFIABLE_DEGENERATE` | match |
| S4 | 8192 | `NO_CLAIM` | `NO_CLAIM` | match |
| S4b | 8192 | `NO_CLAIM` | `NO_CLAIM` | match |
| S2-gate | 8192 (5 levels to 512) | `PASS` | `PASS` | match |
| S5 | 8192 (5 levels to 512) | `FAIL_REGULATOR_DEPENDENT` | `FAIL_REGULATOR_DEPENDENT` | match |
| S6 | 8192 (5 levels to 512) | `PASS_AFTER_ONSET_ALIGNMENT+lnN+IDENTIFIED` | `PASS_AFTER_ONSET_ALIGNMENT+lnN+IDENTIFIED` | match |
| S7 | 512, 2048, 8192 | `ETA_MIN_RECOVERED_OR_FLAGGED` | `ETA_MIN_RECOVERED_OR_FLAGGED` | match |
