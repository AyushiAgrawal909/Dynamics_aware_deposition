# T2 cost estimate

Measured on this machine (4 threads, numba=True) at the production grid: nx = 4096, L = 163.69 v_b/omega_p, dt = 0.05, t_end = 4800.0 (96000 steps).

* PIC throughput: 7.4e+07 particle-steps/s (timed at 256 ppc).
* Vlasov throughput: 4.83e+06 cell-steps/s (timed at nv = 512).

| Arm | Size | Work | Wall time on this machine |
|---|---|---|---|
| PIC nested ladder | ppc [16384, 8192, 4096, 2048, 1024, 512, 256, 128], 4 seeds | 5.13e+13 particle-steps | 192.7 h |
| Stable control | same ladder, 2 seeds | 2.57e+13 particle-steps | 96.4 h |
| Vlasov dv ladder | nv [12800, 6400, 3200, 1600] | 9.44e+12 cell-steps | 543.3 h |
| **Total** | | | **832.3 h** (budget 8.0 node-h) |

Largest single runs: PIC at 16384 ppc ~ 24.2 h; Vlasov at nv = 12800 ~ 289.7 h.
Delta v_b = 0.15 v_b (cost estimate only): Vlasov arm ~ 724.4 h; the PIC arm is unchanged at fixed nx because Delta x = 0.040 is still below lambda_Db(0.15) = 0.043.
Doubling L (finite-size check at one middle level) doubles that level's cost.

These timings are from a 4-core cloud container and are pessimistic for a Meluxina CPU node (128 cores): PIC scales with threads until memory bandwidth saturates, and independent (level, seed) runs are embarrassingly parallel across nodes.

**Decision:** the matrix does NOT fit the local budget. Per the T2 spec, SLURM scripts are generated in `slurm/` and no underpowered run is launched or interpreted here.
