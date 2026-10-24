# T2 validations (V1-V4)

Command: `python3 experiments/nonthermal_attractor/scripts/t2_validate.py` (59 s, 4 threads).

| Check | Backend | Result | Criterion | Pass |
|---|---|---|---|---|
| V1 charge neutrality | PIC | total charge 3.6e-15 | round-off | True |
| V1 momentum (400 steps) | PIC | max drift 2.2e-16 (scale 10.0) | round-off | True |
| V1 energy drift (stable Maxwellian, t=40) | PIC | 3.2e-05 of K | reported | - |
| V2 Landau k lambda_D = 0.5 | PIC | gamma = -0.1540 vs -0.1534 (0.4%) | 5% | True |
| V2 Landau k lambda_D = 0.5 | Vlasov | gamma = -0.1540 vs -0.1534 (0.4%); recurrence T_R = 201 | 5% | True |
| V3 waterbag growth, k = k* | PIC | gamma = 0.3497 vs 0.3522 (0.7%) | 5% | True |
| V3 waterbag growth, k = k* | Vlasov | gamma = 0.3481 vs 0.3522 (1.2%) | 5% | True |
| V4 stable control, dK/K at t=20 | PIC | dx/lambda_Db=1.08: -2.4e-06; dx/lambda_Db=1.08: -2.5e-06; dx/lambda_Db=0.54: 2.8e-06; dx/lambda_Db=0.54: 2.8e-06 | << 0.1 f_tail | see note |
| V4 nonlinear (t=30-60) dx halving | PIC | field energy 0.1%, dK 2.6% | reported | - |
| V4 nonlinear (t=30-60) dt halving | PIC | field energy 0.1%, dK 2.1% | reported | - |

V3 seeds a single mode at k = k* in a box of 4 wavelengths and fits from t = 3/gamma until the mode reaches 5% of its saturated amplitude; the noise-free Vlasov run starts from a 1e-7 seed so that the linear phase is long compared with the beat of the two neutral roots.

V4 note: the short stable-control runs only show that finite-grid heating is not present at t = 20; the decisive measurement is the stable-control arm of the scan at production length, which enters the frozen quality gate. The nonlinear halving runs use the same seed spectrum on a 4-wavelength box at dx = 0.0400 (dx/lambda_Db = 0.69); single seed, so differences include realisation noise after saturation.

```json
{
 "V1_pic": {
  "charge": 3.552713678800501e-15,
  "momentum_drift_max": 2.1597307275911248e-16,
  "momentum_scale": 10.026785960961087,
  "energy_drift_rel": 3.203757806591412e-05
 },
 "V2_pic": {
  "gamma_measured": -0.1540342414102004,
  "gamma_theory": -0.15335946690960478
 },
 "V2_vlasov": {
  "gamma_measured": -0.1539950745304895,
  "gamma_theory": -0.15335946690960478,
  "recurrence_time": 201.06192982974676,
  "mass_drift": 0.0
 },
 "V3_pic": {
  "backend": "pic",
  "k": 0.6141700378894744,
  "gamma_measured": 0.3497098716323394,
  "gamma_theory": 0.3522228436816047,
  "rel_err": 0.00713460837178676,
  "fit_window": [
   8.517335129779031,
   15.65
  ]
 },
 "V3_vlasov": {
  "backend": "vlasov",
  "k": 0.6141700378894744,
  "gamma_measured": 0.3480990157313418,
  "gamma_theory": 0.3522228436816047,
  "rel_err": 0.011708008223313087,
  "fit_window": [
   8.517335129779031,
   40.300000000000004
  ]
 },
 "V4_control": [
  {
   "dx_over_lambda_Db": 1.0825317547305482,
   "dK_over_K": -2.4487137428679653e-06,
   "t": 20.0
  },
  {
   "dx_over_lambda_Db": 1.0825317547305482,
   "dK_over_K": -2.4862436362256376e-06,
   "t": 20.0
  },
  {
   "dx_over_lambda_Db": 0.5412658773652741,
   "dK_over_K": 2.80850487993401e-06,
   "t": 20.0
  },
  {
   "dx_over_lambda_Db": 0.5412658773652741,
   "dK_over_K": 2.83083845177192e-06,
   "t": 20.0
  }
 ],
 "V4_nonlinear": {
  "dx": 0.03996237375973577,
  "dx_over_lambda_Db": 0.6921686174291964,
  "base": {
   "fe_mean": 2.1620433044960956,
   "dK": -2.09311325617875,
   "E_drift_rel": 1.8028823358577384e-05
  },
  "dx_half": {
   "fe_mean": 2.164821643264896,
   "dK": -2.148796474460429,
   "E_drift_rel": 1.5783479050800562e-05
  },
  "dt_half": {
   "fe_mean": 2.1638232108160484,
   "dK": -2.1377167658598495,
   "E_drift_rel": 2.7791896845193855e-05
  },
  "fe_rel_change_dx": 0.0012834030819324751,
  "fe_rel_change_dt": 0.0008225747422690687,
  "dK_rel_change_dx": 0.025913677234444165,
  "dK_rel_change_dt": 0.020865023090726763
 }
}
```
