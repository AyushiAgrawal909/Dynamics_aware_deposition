# Selector Comparison: Preregistered Centerpiece

This directory freezes the selector-by-benchmark comparison before the missing
production runs are built. The figure is a 7 x 2 matrix:

| Method | Weak Landau | Near-marginal two-stream |
|---|---:|---:|
| CIC | sigma = 0 | sigma = 0 |
| Silverman rule of thumb | measured width | measured width |
| MISE oracle | noise-theory optimum | noise-theory optimum |
| LOO-CV (unguarded) | right-censored if it reaches the search boundary | right-censored if it reaches the search boundary |
| Wu-Qin AD-gated CV | hold prior width near uniformity; otherwise finite CV target | hold prior width near uniformity; otherwise finite CV target |
| C4 | sigma = 1 dx | sigma = 0.1845 dx |
| delta-f | variance-reduction reference | variance-reduction reference |

The delta-f row is not eligible to win the selector comparison. It is a
different formulation and is shown as a variance-reduction reference.

## Frozen Benchmarks

### Weak Landau

- `Nx=64`, `Ny=16`, `k_mode=1`, `k lambda_D=0.4`, `alpha=0.01`.
- Rate and noise metrics use PPC 1024.
- The CIC rate must first agree with kinetic theory within 15%.
- Damping fits use the validated envelope and floor-censored Landau fitter.

### Near-marginal two-stream

- Frozen E2 grid and plasma parameters: `Nx=128`, `Ny=16`, `Lx=64`,
  `Ly=16`, `v0=0.8`, `v_th=0.05`, `dt=0.05`.
- The realistic-noise objective uses PPC 32 on the frozen E2 trajectory.
- Rate calibration uses the projected linearized-delta-f `m=12` mode at
  PPC 512 with 10 paired seeds.
- The historical full-f `m=12` trajectory remains excluded because `m=8`
  overtakes it; projection removes that contamination without changing the
  protected mode's linear response.
- Growth fits use the validated projection, competitor, and saturation guards.

The split protocol is deliberate: projected delta-f identifies the uncontaminated
protected-edge rate response, while the PPC-32 full-f run measures the noise
benefit where smoothing is operationally relevant. The exact selected width is
evaluated in both harnesses.

The Silverman row uses a periodic uniform-background rule fixed before the
runs. In two spatial dimensions,
`h_j = (L_j/sqrt(12))*N_particles^(-1/6)`, evaluated at the benchmark's noise
PPC. The anisotropic `(h_x,h_y)` pair is applied; the matrix reports `h_x/dx`
because both protected modes are x-directed. This avoids an origin-dependent
sample variance on a nearly uniform torus.

## Frozen Gate

Let `R_gamma = gamma_sigma / gamma_CIC`. A selector cell passes only if:

1. The CIC baseline is theory-valid within 15%.
2. The rate fit is not censored and passes the benchmark-specific guarded-fit
   quality checks.
3. The entire paired 95% confidence interval for `R_gamma` lies in `[0.9, 1.1]`.
4. The paired 95% lower confidence bound for noise reduction is greater than 1.

`PASS` requires all four conditions. `CIC` is the fidelity reference but cannot
win because its noise reduction is exactly one. `delta-f` is reported as
`REFERENCE`, not ranked. MISE and CV selections at the largest searched width
must be written as right-censored inequalities, never as interior optima.

The unguarded LOO-CV and Wu-Qin controller are intentionally separate rows.
The former characterizes the density objective without a temporal update
guard. The latter uses the same loss but only re-optimizes after an
Anderson-Darling uniformity test reaches `p <= 0.01` and subsequently falls by
another factor of two. A finite target is relaxed into the live width with the
paper's `0.05` adjustment rate. If the density is near uniform, holding the
previous width is the prescribed algorithmic result; it is not a finite CV
optimum.

One disconfirming outcome is named in advance. If C4 passes the rate gate but
the paired lower confidence bound on noise reduction does not exceed one, the
cell is labeled `SAFE_WINDOW_CLOSED`, not `PASS` and not a generic algorithm
failure. Its interpretation is: C4 identified a dynamics-safe width, but the
near-marginal safe window permits no statistically demonstrated smoothing
benefit. This is the finite-sample signature expected when the admissible width
collapses toward marginality; it does not rescue the preregistered winner claim.

The paper-level claim is preregistered as:

> C4 is the only eligible selector that passes both benchmark columns.

The claim is accepted only if the completed matrix supports it. The matrix is
allowed to disprove the claim.

## Cell Contents

Each cell reports the same output schema:

- censored fitted rate error relative to the benchmark theory target;
- paired rate ratio and 95% confidence interval relative to CIC;
- resolved e-foldings and the measured late-time floor;
- noise reduction `MISE(CIC) / MISE(selector)` and its paired interval;
- gate status and censoring reason.

Every floor is estimated independently for that selector, benchmark, PPC, and
seed ensemble on the validated tail window. A benchmark-level or CIC-inherited
floor is invalid input and the gate evaluator leaves that cell pending.

The matrix also prints each selected width relative to C4. In particular, the
final Silverman ratios are `1.833x` for Landau and `31.549x` for E2. This
exposes the regime asymmetry directly rather than reducing it to pass/fail.

For a right-censored width the figure prints filter attenuation at the search
boundary as an inequality. At `sigma >= 60 dx`, the protected-mode matched-W
response is already at most about `10^-15` for Landau and `10^-241` for
isolated E2 `m=8`; wider CV choices attenuate further. The E2 value remains
representable in binary64, but is operationally zero relative to field-scale
roundoff and underflows binary32. It must not be described as binary64
underflow.

Landau and two-stream need different physical stopping guards, but both pass
through the same matrix validation and paired-inference contract.

## Build Contract

`config/comparison_matrix.json` is the source of truth. Run:

```bash
python paper1_selector_comparison/scripts/build_comparison_manifest.py
python paper1_selector_comparison/scripts/plot_comparison_matrix.py
```

The first command creates the cell template and build manifest. The second
renders the matrix and refuses to label incomplete cells as passes.
