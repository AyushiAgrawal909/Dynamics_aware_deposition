# Exact-width q050_x000 validation plan

## Compatibility audit

| Feature | Statistical selection experiment | Prepared dynamics validation | Assessment |
|---|---|---|---|
| Equilibrium | symmetric beams `+/-0.80`, thermal spread `0.05` | same nominal equilibrium | Compatible equilibrium parameters. |
| Wavevector | candidate budget protects `(12,1)`, but saved full-f snapshots were initialized with perturbation `(12,0)` and the objective is full-grid | actual projected conjugate pair `(12,1)` | The protected transfer is the same; the evolved perturbation is not the same. |
| Grid/domain | `128 x 16`, `Lx=64`, `Ly=16`, `dx=0.5`, `dy=1` | identical | Exact match. |
| Time step | `0.05` | `0.02` | Different numerical time response. |
| Duration/window | snapshots `0.5--4.0` | rates on `[24.32,69.16]`, stop `71.16` | Different observables and time scales. |
| Filter | identical Gaussian amplitude definition; candidate W applied once offline to deposited density | W once after CIC deposition and once to the nodal return field | Same one-way W, but selector metrics are pre-return; particle force contains `W^2`. |
| PPC/reference | candidate PPC 32; CIC reference PPC 1024 | both dynamics arms PPC 512 | Different finite-marker systems. |
| Loading | regular marker lattice, global quiet-start thermal nodes, random balanced beam signs; low and reference are separate initializations using the same numeric seed | same initializer family and regular lattice; q050 actual arrays must match the production-runtime CIC reconstruction | Same algorithm. Historical CIC arrays were not saved, so byte identity to the archive is not claimed. |
| Perturbation | full-f, `alpha=0.001` | linearized delta-f, `alpha=1e-8` | Different simulation formulation. |
| Field solve | full spectral solve | periodic spectral solve after projection to the target conjugate pair | Same spectral operator on the protected pair; different mode content. |
| Live selection runs | both low and reference evolved with an isotropic `0.25`-cell live filter; candidate widths were evaluated offline | exact q050 width is in the live matched coupling | Candidate performance and dynamics were not measured in one run. |

The research trees have identical grid, CIC and conventions files. Their
`particles2d.py` files differ only by later optional position-jitter support
whose default is false; `init_two_stream_2d` is identical. The later
`spectral_W.py` adds binomial filters but retains the same Gaussian function.

Conclusion: this is a valid cross-experiment test of whether the archived
selected width preserves the targeted linear mechanism under the audited
dynamics benchmark. It is not a same-configuration joint error-benefit and
dynamics result.

## Exact transfer at the protected mode

With `Ny=16`, `m_y=1` and `sigma_y/dy=0.1716392815470568`,

```text
q = (2*pi*m_y*sigma_y/(Ny*dy))^2 = 0.004543108902388957
W = exp(-q/2)                         = 0.9977310235762052
W^2 in the matched particle force    = 0.9954671954064221
|W|^4 in particle-force power        = 0.9909549371303280
```

The audited grid/time-discrete model gives
`gamma_CIC=0.11746393707505787`, `gamma_q050=0.11174354321424793` and
ratio `0.9513008502587932`, satisfying the fixed 5% operating rule.

## Measurement and decision

For each seed, fit the slope of `ln(abs(Ehat_12,1))` on the frozen window. The
main realised-mode estimand is the ratio of the two arithmetic means of these
individual rates. Its uncertainty is a paired seed bootstrap: each draw uses
the same resampled seed indices in both arms. No temporal samples are treated
as independent observations.

Also report the slope of `ln(mean_seed(abs(Ehat_12,1)))` for each arm and its
paired ratio. This ensemble-mixture slope is a distinct observable and can
differ from the mean individual slope because growing seeds acquire different
amplitude weights. Neither result replaces the other.

Success for the selected-width dynamics test requires the quality gates in the
README plus the entire primary paired interval inside `[0.90,1.10]`. The
secondary estimator gets its own verdict. The source-locked reducer writes all
bootstrap draws, individual rates and quality values. Because the individual
rate mean is primary, every individual log-amplitude fit must also satisfy the
same frozen `R^2 >= 0.98` threshold; no seed can be dropped to pass it.

## Recovered CIC arm and pairing qualification

The Mac search found selector records and unrelated or incompatible histories,
but no local exact-mode CIC set. A read-only MeluXina search recovered all ten
CIC histories from job 5201231 at
`e2_beam_drift_scan_5201231/runs/vb0800/cic/seed200--209`. Every saved result
matches the frozen dynamics configuration: `(12,1)`, PPC 512, `dt=0.02`,
`t_end=71.16`, `alpha=1e-8`, zero filter width and projected conjugate target
mode. Each history has 1,780 samples at cadence 0.04. The largest saved
delta-f validity ratio among them is `0.0159613`, below the frozen `0.5` gate.

Historical particle arrays and their hashes were not saved. Equal seed numbers
are therefore insufficient by themselves. The reuse audit instead reconstructs
each state on MeluXina under the same Python 3.11.10/NumPy 2.3.5 runtime from
the archived wrapper, frozen arguments, explicitly hashed production script,
`particles2d.py`, `delta_f.py`, `grid2d.py` and seed. It records all six
reconstructed array hashes and an aggregate hash. The reconstructed time-zero
target-mode value agrees exactly with the archived history in this production
runtime.

The new runner reconstructs the q050 state before time integration and refuses
to run unless its aggregate hash equals the frozen production-runtime value.
The final reducer then checks each actual q050 array hash (`x`, `y`, `vx`,
`vy`, `w_full`, `w_delta`) against the reuse audit. This is a deterministic
pairing construction, not recovery of a saved historical array hash.

The historical input manifest omitted `conventions.py` and `cic.py`. Their
observed mtimes predate job 5201231, but mtimes are not cryptographic evidence;
the recorded git revision cannot fill this gap because the research Python tree
was untracked. `conventions.py` controls marker-grid factorization, while
`cic.py` does not create the initial particle arrays. This residual limitation
is explicit in the audit. On balance, the exact wrapper/config, hashes of the
state-construction implementation, production-runtime reconstruction, exact
time-zero agreement and mandatory q050 hash checks are adequate for
conditional reuse. Any mismatch stops the campaign as unresolved.

The original design used fresh paired CIC/q050 seeds `105--114`. It changed
before any q050 dynamics result existed because the compatible CIC histories
were recovered and passed the audit. Seeds `200--209` are disjoint from
selection seeds `100--104` and static-error seeds `105--109`, but they were
already analysed in job 5201231 and are not fresh confirmation seeds.

The registered CIC ensemble-amplitude duration is `6.4904061` e-foldings.
Individual-seed fitted durations are distinct and range from `4.2076787` to
`7.3459241`; every seed's history, `R^2`, delta-f, positive-rate, two-e-folding
diagnostic and fit-start-SNR diagnostic passes. Full values are in
`CIC_QUALITY_GATES.md` and `reuse_audit.csv`.

## Cost and stopping rule

The empirical basis is completed job 5195640: approximately `0.45` CPU
node-hour for ten concurrent histories through `t=60`. Scaling to `t=71.16`
gives roughly `0.534` node-hour for the one missing packed ten-history q050
arm. Allowing for the CIC audit and reduction gives an expected new cost of
about `0.60` node-hour. The fixed scheduler cap is `1.0` node-hour (`01:00:00`
on one CPU node with ten worker cores). The ten archived CIC histories cost no
new allocation.

The run ends at the fixed duration or at the scheduler cap regardless of the
result. A timeout, missing history, pairing mismatch or failed signal gate is
reported as unresolved. There is no automatic resubmission, extension or seed
addition.
