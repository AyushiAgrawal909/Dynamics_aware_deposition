# Paper 1 reproducibility manifest

The authoritative manuscript source, as identified by the user on
2026-09-10, is
`Dynamics_constrained_Deposition/introduction.tex`.  The earlier
`paper1_source_package/paper1.tex` edits are retained for provenance but that
file is not the submission source.

Audit date: 2026-09-12 (Europe/Luxembourg)

Original manuscript checked: `Dynamics_constrained_Deposition.pdf` (29 pages,
local timestamp 2026-09-09 15:21).  Revised source checked:
`Dynamics_constrained_Deposition/introduction.tex`; the latest verified
37-page output is
`Dynamics_constrained_Deposition/output/Dynamics_constrained_Deposition_verified_20260911_r4.pdf`
(SHA-256 `be7ee1a4...5af9`).  The preceding verified PDFs are retained unchanged.
The package-local `paper1.tex` is checked only as a non-authoritative record of
earlier edits.

## Revision status

The supplied HEXAPIC repository is at commit
`ed6dd96d16d0b70a53bf9486fd78a270ab3486d6` (`origin/main`, initial commit,
2026-04-04).  The working tree contains substantial later research changes.
The Paper 1 Python package `pic_kde_2d/` is not tracked by that commit.

None of the inspected Paper 1 result JSON/CSV files records a Git commit, a
source archive, or a worktree-diff hash.  Slurm accounting retains submission
commands such as
`sbatch paper1_selector_comparison/slurm/stage1_landau_noise_packed_gpu.sbatch`,
but the historical batch scripts have expired from `scontrol`.  Consequently,
the exact source revision that generated each research result is **unknown**.
The entry points listed below are the current recovered files whose interfaces
and output schemas match the archived results; they are not represented as an
exact historical checkout.

The C++ files under `hexapic-main/source/` are later modifications of the
supplied commit.  The Python 2D2V benchmark implementation under
`pic_kde_2d/` is a later research implementation, not supplied baseline code.
The preserved `pic_kde_2d_landau_cv_stage/` copy is later still and contains
the binomial/compensated functions absent from the top-level Python copy.

## Common runtime

The recovered Slurm scripts use Python 3.11.10 from
`/apps/USE/easybuild/release/2024.1/software/Python/3.11.10-GCCcore-13.3.0`.
They set `NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and usually
`OPENBLAS_NUM_THREADS=1`.  No archived package lock, module list, compiler
version record, or NumPy/Numba version record was found.

Unless a row says otherwise, the Python benchmarks use the repository's
periodic spectral Poisson solver and matched CIC deposit/gather.  HYPRE belongs
to the separate C++ HEXAPIC path and was not used for these Python rate and
noise campaigns.

## Campaign manifest

| Manuscript result | Jobs/data | Recovered execution settings | Current matching entry points | Revision status |
|---|---|---|---|---|
| Weak-Landau CIC/C4 rate sweep; Fig. mechanism; Table 4 rate rows | simulation `4864819`, reducer `4864820`; `results/landau_sigma_refinement_4864819` | `Nx=64`, `Ny=16`, `Lx=15.707963267948966`, `Ly=8`; `k lambda_D=0.4`, `alpha=0.01`; PPC `{32,1024}`; sigma `{0,0.3,0.5,1,2,3,5,8,12}` cells; seeds `0..9`; global shuffled quiet start; `dt=0.05`; 1800 steps (`t_end=90`); diagnostic stride 2; paired analysis, 1000 bootstrap samples. PPC 32 baseline and bootstrap quality fail and are unresolved | `scripts/run_landau_sigma_refinement.py`; `scripts/submit_landau_sigma_refinement_packed_cpu.sbatch` | exact historical revision unknown; raw histories recovered and reduction reproduced |
| Weak-Landau Silverman and CV rates | `4866573`, `5007754`; `results/p1_landau_rates_4866573`, `results/landau_cv_rate_5007754` | same weak-Landau rate protocol; PPC 1024; seeds `0..9`; sigma `1.8329728493314705` and `6.61`; 1800 steps; baseline symlinked from `4864819`; 2000 paired bootstraps | `paper1_selector_comparison/slurm/stage2_landau_exact_rates_cpu.sbatch`; `scripts/submit_landau_cv_rate_exact_gpu.sbatch`; `scripts/run_landau_sigma_refinement.py` | exact historical revision unknown |
| Weak-Landau MISE-width rate | `4872322`; `results/landau_mise6_rate_4872322` | same weak-Landau rate protocol; PPC 1024; sigma 6 cells; seeds `0..9`; 1800 steps; baseline from `4864819`; 2000 paired bootstraps | `paper1_selector_comparison/slurm/stage2_landau_mise6_rate_gpu.sbatch`; `scripts/run_landau_sigma_refinement.py` | exact historical revision unknown |
| Weak-Landau Gaussian density MISE/noise reductions used by the revised figures | `4868286`; `results/p1_landau_noise_4868286` | `Nx=64`, `Ny=16`, `Lx=15.707963267948966`, `Ly=8`; PPC 1024, reference PPC 16384; seeds `0..9`; **global shuffled** velocity quadrature; `alpha=0.01`; `dt=0.05`; 600 steps; diagnostic stride 10; averages `t=[5,30]` over 51 times; live matched Gaussian sigma 0.25 cell; candidate Gaussian sigma grid `{0,.25,.5,1,1.8329728493,2,3,4,6,8,12,16,20,24,30,40,60}`. Metric: ratio of full-grid, time-averaged density squared-error powers against paired high-PPC CIC density; all resolved modes are included, with `k=0` cancelling by charge normalization | `scripts/run_landau_mise_gate.py`; `paper1_selector_comparison/scripts/reduce_noise_shards.py`; `paper1_selector_comparison/slurm/stage1_landau_noise_packed_gpu.sbatch` | exact historical revision unknown; raw tree recovered byte-for-byte and Slurm submission line recovered |
| Table-4 Gaussian C4/Silverman density-error reductions | `4867515` on MeluXina | same grid/PPC/time window/metric as `4868286`, but **randomized per-cell** velocity quadrature; C4 reduction `2.559393`; Silverman `3.079320` | same current entry points as preceding row, selected with `--quiet-start-layout per-cell` | exact historical revision unknown; raw tree recovered byte-for-byte and protocol mismatch disclosed rather than silently replacing its measurements |
| Weak-Landau raw-binomial noise and rates; Fig. 7a | `5048431` noise, `5048412` rate; corresponding local result directories. E2 raw-binomial noise is job `5048435` | Landau noise: PPC 1024/reference 16384, seeds `0..9`, `alpha=.01`, 600 steps, `dt=.05`, diagnostic stride 2, `t=[5,30]` (251 times), live Gaussian sigma .25; passes `{0,1,2,4,8,12,13,16}`. The `n=2`, low-k-equivalent `sigma_eq=1` reduction is `5.871509`; the Gaussian `5.643223` overlay comes separately from job `4868286`. Rate: PPC 1024, seeds `0..9`, 1800 steps; same pass set. All noise values are full-grid density squared-error-power ratios | recovered campaign copy `pic_kde_2d_landau_cv_stage/scripts/run_binomial_noise_seed.py`, `reduce_binomial_noise.py`, `run_landau_binomial_pass.py`, and its binomial submit/reducer scripts | exact historical revision unknown; Landau and E2 raw-noise trees recovered byte-for-byte |
| Compensated and higher-order filter rows | `results/compensated_binomial_20260811`, `results/higher_order_compensation_20260811`; individual Slurm job IDs not preserved in copied summaries | Landau rate: PPC 1024, seeds `0..9`, `alpha=.01`, 1800 steps. Historical E2 rate: PPC 512, seeds `0..9`, `alpha=1e-8`, mode `(12,0)`, `Nx=128`, `Ny=16`, `Lx=64`, `Ly=16`, `v_beam=.8`, `v_thermal=.05`, `dt=.02`, `t_end=60`, stride 2; all 140 original histories confirm `(12,0)`. First order passes `{0,1,2,4,8,16}`; orders 2/3 passes `{0,1,2,4}`. Noise follows the density-MISE protocol for each benchmark | recovered campaign copy `pic_kde_2d_landau_cv_stage/scripts/run_landau_binomial_pass.py`, `run_linearized_delta_f_mode.py`, compensated/higher-order submit and analysis scripts; filter implementation in that copy's `pic_kde_2d/methods/spectral_W.py` | numerical reductions reproduced from raw histories; exact historical revision/job IDs unknown; historical E2 analyzer needs two helpers absent from stage copy. New direct corner results are recorded separately below |
| E2 isolated mode-8 Gaussian rate sweep | `4863559`, analysis `4864701`; `results/E2_m8_sigma_sweep_4863559` | PPC 2048; modes 1--14 tracked; target mode 8; sigma `{0,.1,.2,.35,.5,1,2,3,5,8}` cells; seeds `0..9`; paired sigma-zero ensemble; competing-mode guard. Remaining runtime settings must be recovered from raw run metadata or remain unknown | `scripts/run_E2_m8_sigma_sweep_highppc.py` and its packed submit/analyzer scripts | exact historical revision unknown |
| Linearized-delta-f reference and protected E2 band | `4872599`, `4873221`; recovered raw trees `.../recovered_meluxina_20260911/df_linear_ref_4872599` and `df_band_1_14_4873221` | modes 1--14; seeds `0..9`; E2 reference PPC 512; `Nx=128`, `Ny=16`, `Lx=64`, `Ly=16`; `v_beam=.8`, `v_thermal=.05`; reference perturbation amplitudes are campaign-specific and must be read from each result (do not assume one shared alpha) | `scripts/run_linearized_delta_f_mode.py`; delta-f reference/band submit and analysis scripts | all 30 reference and 140 band histories recovered and current reducers reproduce archived tables; exact historical revision unknown |
| E2 mode-12 Gaussian response | `4872745`; recovered raw tree `.../recovered_meluxina_20260911/df_m12_sigma_4872745` | PPC 512; mode 12; seeds `0..9`; `alpha=1e-8`; sigma `{0,.175,.1844548849,.1987203604,.22}`; `Nx=128`, `Ny=16`, `Lx=64`, `Ly=16`; `v_beam=.8`, `v_thermal=.05`; `dt=.02`, `t_end=60`, stride 2; 2000 paired bootstraps | `scripts/run_linearized_delta_f_mode.py`; `scripts/analyze_linearized_delta_f_m12_sigma.py`; `scripts/submit_linearized_delta_f_m12_sigma_cpu.sbatch` | all 50 logical histories recovered; sigma-zero symlink materialized locally; reducer reproduces archived values to `3.66e-15`; exact historical revision unknown |
| E2 directional/anisotropic corner; Tables 3/5 and Pareto | `4913408`; `results/e2_joint_corner_4913408` | rate: projected mode `(12,1)`, PPC 512, seeds `0..9`, `alpha=1e-8`, same E2 grid/distribution, `dt=.02`, `t_end=60`, stride 2; fit window `[24.32,59.68]`, 5000 paired bootstraps. Noise: PPC 32/reference 1024, `alpha=.001`, `dt=.05`, 80 steps, stride 2, evaluation `[.5,4]`; configurations stored verbatim in submit script/result JSON | `scripts/run_linearized_delta_f_mode.py`; `scripts/run_E2_joint_corner_noise_seed.py`; `scripts/analyze_E2_joint_corner_sweep.py`; `scripts/submit_E2_joint_corner_sweep_gpu.sbatch` | exact historical revision unknown; all raw rate/noise histories recovered from MeluXina and reduction reproduced |
| Weak-Landau PPC regime map | snapshots `5049182`, reference reducer `5049264`, dynamics `5049265`, final reducer `5049266` | PPC `{8,32,128,512,1024,4096}`; choices `{CIC,C4,MISE,LSCV}`; seeds `0..9`; selector widths read from `5049264`, not hard-coded. Exact snapshot runtime settings are in the selector result JSON where present; unrecorded settings remain unknown. At PPC 4096, C4 has rate ratio `1.02078 [0.99763,1.03065]` but only `1.155` fitted model e-foldings; the archived `1.809` resolved-capacity value is not fitted duration | `scripts/run_landau_ppc_selector_snapshots.py`; `scripts/reduce_landau_ppc_common_reference.py`; `scripts/run_landau_ppc_dynamics.py`; corresponding submit/reducer scripts | exact historical revision unknown; dependency chain and all 240 raw histories recovered |
| Static LSCV widths and strong-Landau boundary | `5008392`, `5008393`, `5007754`, `5048278`, plus `results/strong_landau_alpha025_20260811` | grid `64x16`, `Ly=8`, `k lambda_D=.4`, PPC 1024, global shuffled quiet start, seeds `0..9`; alphas `.01`, `.25`, `.5`. The exact LOO selector self term uses every particle's actual CIC phase through `c_p^T(CWC^T)c_p`, rather than an eight-offset phase approximation. Alpha .5 rate grid `{0,1,1.10785,1.5,2,2.453}`, `dt=.05`, 900 steps, stride 2. At its CV width, fit starts 3 and 5 give ratios `1.01020` and `1.03116`, with only `1.242` and `.656` model e-foldings. Alpha .25 exact CV-width fit gives ratio `1.04791` and `.964` e-foldings. Both finite-amplitude cells fail the stated duration gate | `scripts/run_landau_lscv_aligned.py`; `scripts/run_landau_sigma_refinement.py`; strong-Landau submit/analyzer scripts | raw LOO shards and rate histories recovered and numbers reproduced; historical `FAIL` versus current `UNRESOLVED` analyzer-label difference retained, while Task 7 classifies the validation gate as `FAIL (duration)`; exact revisions/job IDs unknown |
| Strong-Landau raw-binomial pass sweep | `5048414`; recovered `landau_binomial_strong_5048414` | PPC 1024, alpha `.5`, seeds `0..9`, passes `{0,1,2,4,8,12,13,16}`, `dt=.05`, 900 steps, stride 2. At n=16 the fit-start-3 ratio is `1.06600 [1.06453,1.06723]` with `1.311` e-foldings; fit-start 5 gives `1.17925 [1.17618,1.18171]` with `.750` e-foldings | preserved-stage `run_landau_binomial_pass.py` and `analyze_landau_binomial_passes.py` | raw histories recovered and both reductions reproduced; the manuscript's former “all tested pass” statement is not supported |
| New direct compensated E2 binding-corner validation | rate array `5195640`, dependent reducer `5195641`, both completed exit 0 | projected `(12,1)`, PPC 512, alpha `1e-8`, seeds `0..9`, `128x16`, `Lx=64`, `Ly=16`, `v_beam=.8`, `v_thermal=.05`, `dt=.02`, `t_end=60`, stride 2; CIC, p1/n1, p2/n2, p2/n4, p3/n4; fixed fit window `[24.32,59.68]`, 5000 paired seed bootstraps. Direct ratios: p1/n1 `0.83807` FAIL, p2/n2 `0.95608` PASS, p2/n4 `0.78885` FAIL, p3/n4 `0.97342` PASS; every arm passes duration and signal | `reproducibility/meluxina/submit_E2_compensated_corner_cpu.sbatch`; `analyze_E2_compensated_corner.py`; local raw copy `pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/e2_compensated_corner_5195640` | current audited source hashes matched local/remote; 50 histories and reducer products copied byte-for-byte; local/remote aggregate manifest hash `1376b971...b7f` |
| New weak-Landau duration pilot | rate array `5195658`, dependent reducer `5195659` | three paired seeds, PPC 1024, alpha `.02`, `dt=.05`, 1800 steps, stride 2; CIC/C4 and CIC/compensated p1-n16. Both Gaussian arms are below the five-floor threshold at `t=5`; reducers return no finite rate, zero e-foldings and zero filtered bootstrap quality | `reproducibility/meluxina/submit_weak_landau_duration_pilot_cpu.sbatch` and dependent analysis script | completed; failed signal precondition, so no ten-seed expansion was submitted |
| New exact LOO-CV selected-width dynamics | rate array `5195682`, dependent reducer `5195683`, both completed exit 0 | paired CIC and sigma `7.244293147028428`, PPC 1024, alpha `.01`, seeds `0..9`, `dt=.05`, 1800 steps, stride 2; 2000 paired seed bootstraps. Measured ratio `1.82222 [1.48934,2.21398]`, `1.62678` fitted e-foldings; floor-censored, zero quality-passing bootstrap fraction | `reproducibility/meluxina/submit_weak_landau_lscv_exact_cpu.sbatch` and dependent analysis script; raw local copy `pic_kde_2d/results_raw_rate_audit/recovered_meluxina_20260909/weak_landau_lscv_exact_5195682` | current audited runner hash matched local/remote; all 20 histories and reducer products copied byte-for-byte from tier2 |
| Held-out weak-Landau selector comparison | selector snapshots job `5049182`; fixed audit split seeds `0..4` train, `5..9` test | PPC 1024 candidates versus paired PPC 4096 reference; globally shuffled loading; 51 snapshots over `t=[5,30]`; identical 22-width isotropic Gaussian family and 5610 width-snapshot evaluations for MISE and LOO-CV. MISE selects 7 cells and gives held-out reduction `4.99199 [4.95617,5.02573]`; LOO selects 12 and gives `4.94856 [4.91313,4.98055]`. All 252 balanced 5/5 splits preserve these widths | `reproducibility/validate_selectors_independent.py`; results under `reproducibility/results/selector_independent_audit` | 192-file snapshot tree recovered from MeluXina; remote/local aggregate SHA-256 `3e1c254f...f0f2b`; density-only discrete-family result |
| Dense held-out E2 anisotropic search | array `5197174`, reducer `5197184`, all completed exit 0 | 85 predeclared model-admissible diagonal Gaussians; binding `(12,1)` grid/time budget `q=0.009086217804777913`; PPC 32/reference 1024; seeds `100..104` train and `105..109` test; `128x16`, `L=(64,16)`, `alpha=.001`, `v_beam=.8`, `v_thermal=.05`, `dt=.05`, 80 steps, stride 2, evaluation `[.5,4]`. Both density and pre-gather field select `(0,0.2427346)`; held-out reductions `1.15837 [1.15409,1.16196]` and `1.07509 [1.07064,1.07989]` | `reproducibility/e2_anisotropic_design.py`, `run_E2_anisotropic_dense_noise_seed.py`, `analyze_E2_anisotropic_dense_noise.py`, and matching Slurm launchers | 43-file raw/result tree copied without overwrite; remote/local aggregate SHA-256 `d43ffbe5...425b`; optimality limited to the 85-point design and stated grid metrics |
| Independent new E2 protected-mode test | rate array `5197185`, reducer `5197186`, all completed exit 0 | projected new mode `(10,1)`; Gaussian `(sigma_x,sigma_y)=(0,0.2427346)` versus CIC; PPC 512; new seeds `100..109`; `alpha=1e-8`; `128x16`, `L=(64,16)`, `v_beam=.8`, `v_thermal=.05`, `dt=.02`, `t_end=60`, stride 2; fixed fit `[20,44]`, 5000 paired bootstraps. Measured ratio `0.989118 [0.988928,0.989253]`; prediction `0.989247`; 6.976 e-foldings, `R^2=.999993`, SNR 232.8; all gates pass | `reproducibility/analyze_E2_new_protected_mode.py` and `meluxina/submit_E2_new_mode_m10n1_*.sbatch`; local raw copy `.../recovered_meluxina_20260909/e2_new_mode_m10n1_5197185` | current local/remote scripts hash-matched before launch; 63-file remote/local aggregate SHA-256 `2f7d4267...f5c4`; local reducer adds verified grid/time-discrete prediction |
| Preregistered E2 symmetric-beam-drift scan | frozen config `results/e2_beam_drift_scan_design_20260911/frozen_campaign.json`; failed pre-launch attempt `5201218`; timed-out production job `5201231`; registered reduction `results/e2_beam_drift_scan_analysis_20260911_timeout_r6`; completed extension `5202227`; combined diagnostic reduction `results/e2_beam_drift_scan_analysis_20260912_complete_extension_r9` | projected `(12,1)`; symmetric drifts `{.60,.70,.76,.80,.818}`; CIC, fixed Gaussian `(0,.2427346)` with constant `W^2=.99095494`, and per-drift constrained Gaussian targeting predicted ratio `.95`; PPC 512; fresh paired seeds `200..209`; `128x16`, `L=(64,16)`, `v_thermal=.05`, `alpha=1e-8`, `dt=.02`, stride 2; fixed fit start `24.32`, model-scheduled stops `{40.76,44.12,50.80,69.16,204.08}`; 5000 paired bootstraps | `design_E2_beam_drift_scan.py`, `verify_E2_beam_drift_scan_transfer.py`, `run_E2_beam_drift_seed.py`, `analyze_E2_beam_drift_scan.py`, `complete_E2_estimator_comparison.py`, `diagnose_E2_seed_rate_variation.py`, and frozen MeluXina launchers | Original job `5201231` remains a 120/150 four-hour timeout; all eight filtered measured gates pass at `.60--.80`. Separately frozen job `5202227` completed the 30 missing `.818` histories in `02:21:40` without a budget/configuration change. The `.818` physical result is unresolved because every arm fails fit-start SNR and three seeds/arm fail the linear-`delta f` guard. Registered results remain primary. Original raw aggregate SHA `0d00fe2d...75114`; extension campaign hash `e664460d...af446`; amendment hash `fc9ae62f...15700`. |
| Erasure figure | `results/weak_landau_alias`; no Slurm job ID recovered | PPC 128; seeds `0..9`; coherent mode histories have 901 samples to approximately `t=90`; CIC and reference-MISE width 16. JSON gives common fit window and floors; per-seed arrays are the NPZ files. Other launch settings remain unknown until a matching submit log is found | `paper1_selector_comparison/figures/make_fig_erasure.py`; originating simulation entry point unknown | simulation revision/job ID unknown |
| Spectral static-width energy campaign | array job `5048437`; `results/energy_conservation_5048437` | 10 seeds per cell, `dt=.05`, diagnostic stride 2. Weak Landau: `64x16`, PPC 1024, alpha `.01`, `t_end=90`, widths `{0,1,6}`. Strong Landau: same grid/PPC, alpha `.5`, `t_end=45`, same widths. E2 longitudinal `(12,0)`: `128x16`, PPC 32, alpha `.001`, `t_end=60`, widths `{0,.1844549,12}`. Energy is `0.5 sum |w|v^2 + 0.5 sum |E|^2 dxdy`; one full-history slope is fit per run | recovered campaign copy `pic_kde_2d_landau_cv_stage/scripts/run_energy_conservation_case.py`, `reduce_energy_conservation.py`, and submit script | exact historical revision unknown; all raw NPZ histories archived locally |
| Spatially varying grad-h momentum audit | jobs `4855514` (one-step gate), `4855533` (four 1500-step cases), `4858823` (four additional phases); raw local directories under `results/hexapic_momentum_audit` | HEXAPIC `256x4`, fully periodic, electrons and ions 8192 each, `np2c=1`, random seed 1729, `dt=1.772590709316941e-11`, 1500 steps; PFMG tolerance `1e-9`, red-black GS (`relax_type=1`), cold start, zero-mean periodic RHS, unpinned gauge; prescribed smooth sigma `(0.5,3.0,beta=6,h_y=0.5)` with grad-h on/off plus uniform and discontinuous controls. The 42.8% and 1817 results are x-component ratios; norm ratios differ | current matching files under `meluxina_grad_h_patch/` and modified `hexapic-main/source/` | exact historical source/executable hash unknown; input decks and raw logs recovered |
| Grid-phase momentum fingerprint | `4858774` one step and `4858823` 1500 steps; local raw/compact results | five phase shifts `{0,1/8,1/4,3/8,1/2}` cell with physical particles and smooth width profile translated together. Remaining settings are those of the grad-h row. The one-step x-force changes sign, while the long-run norm is y-dominated | `meluxina_grad_h_patch/submit_grid_phase_one_step_cpu.sbatch`, `submit_grid_phase_full_cpu.sbatch`, `analyze_grid_phase.py` | exact historical source/build unknown |
| PFMG tolerance isolation diagnostic | new job `5195488`; `results/hexapic_momentum_audit/hypre_tolerance_one_step_5195488` | same one-step smooth-profile case and seed as the grad-h gate; tolerances `{1e-5,1e-7,1e-9,1e-11}`; achieved manual residuals `9.60e-6` to `9.89e-12`; four single-rank tasks, job elapsed 26 s. Executable and relevant source hashes embedded in output | `reproducibility/run_hypre_tolerance_scan.sbatch`; current remote HEXAPIC executable SHA-256 `5f29c47c...60d4` | exact for diagnostic job 5195488; not a production result |
| Delta-f weight-boundary audit | `4873222`; recovered `df_breakdown_4873222` | projected Landau mode `(1,0)`; alpha `{.001,.01,.1}`; PPC 128; seeds `0..4`; `64x16`, `Lx=15.707963267948966`, `Ly=8`; `dt=.05`, `t_end=60`, stride 2; free-streaming frozen-equilibrium marker characteristics; perturbation-only weights; heuristic guard `max|w_delta|/mean|w_full|<.5` | `scripts/run_linearized_delta_f_mode.py`; corrected `scripts/analyze_delta_f_breakdown.py`; `pic_kde_2d/reproducibility/audit_secondary_claims.py` | 47 raw files recovered with matching remote/local aggregate SHA-256 `70abf806...36e`; exact historical source revision unknown; old result schema predates current runner schema |
| Finite-amplitude full-f/delta-f scope check | `4873285`, reusing job `4873222` alpha-.1 delta-f histories | full-f CIC and target-mode-projected linearized delta-f; alpha .1; PPC 128; paired seeds `0..4`; same `64x16` Landau grid; `dt=.05`, 1200 steps, stride 2. Paired target-mode energy rebound ratios use fixed windows `[12,18]`, `[30,40]` and `[40,60]` | `scripts/analyze_delta_f_fullf_overlay.py`; independent `pic_kde_2d/reproducibility/audit_secondary_claims.py` | all five full-f and five reused delta-f histories recovered; full-f directory SHA-256 `ee6d8cb3...d0ebe`; no particle phase space or harmonic spectra were recorded |
| Analytic adaptive-controller H1 gate | `results/H1_analytic_local`; 20 trajectory result files | density contrasts `{10,100}`; seeds `0..9`; CIC/global C4/frozen C4-capped Abramson; 24 width bins; `64x16`, PPC 32, reference PPC 256, equilibrium PPC 1024; `alpha=.01`, mode 4, `dt=.05`, `t_end=1`, stride 2 | `scripts/run_H1_analytic_ladder.py`; `methods/adaptive_sigma.py`; independent Task-10 reducer | exact historical revision unknown; H1 fails and records `STOP_LEARNING_ANALYTIC_H1_NEGATIVE`; no learned model was trained |
| G0 locality headroom and implementation precheck | recovered `G0_locality_headroom_ceiling_baseline` and `G0_conservation_precheck` | zero-fit covariance calculation requests free width 384 cells and predicts packet-field gain 1.6223; implementation cap 20 cells predicts 1.000033; requested moving profile is unrepresentable and no dynamics is run | exact scripts/config/tests preserved in `.../recovered_meluxina_20260910/source_snapshot` | aggregate hashes `d17ee858...5fc2` and `707bd94b...b12`; decision `STOP_H2_VARIATIONAL_PROFILE_NOT_IMPLEMENTED` |
| Fixed-width component cost table | `5049224`; recovered `component_cost_5049224` | same node, five order-rotated repeats; methods CIC/C4/delta-f; PPC 1024; seeds/repeats `0..4`; 10 warm-up and 200 timed steps; 1,048,576 particles; one CPU task, 16 GiB request. Filter application is timed; bandwidth selection, filter construction and particle/grid initialization are excluded. Memory delta is explicit persistent arrays, not incremental peak RSS | exact cost generator/reducer/submit snapshots preserved under `.../recovered_meluxina_20260910/source_snapshot`; independent Task-10 audit | all 32 files copied with remote/local SHA-256 `467e3ee0...29b3`; supports fixed-width stepping only, not end-to-end selector efficiency |

## Weak-Landau confirmatory v2 (completed 2026-09-11)

The replacement confirmation is frozen in
`weak_landau_confirmatory_v2_global_velocity.json` (SHA-256
`d107eeda...d707`).  It uses 40 fresh seeds `202609300..202609339`, exact
1024-particle-per-cell regular spatial loading, and a global inverse-normal
velocity template with 1,048,576 distinct longitudinal nodes.  The positive
resonance at `v=3.2126424` is inside support: the PPC-1024 and PPC-4096
templates contain exactly 690 and 2758 nodes at or above it.  The frozen grid
is `64x16`, `L=(5 pi,8)`, `dt=.05`, `t_end=90`, with mode cadence `.1` and
noise snapshots every `.5` on `[5,30]`.  The periodic FFT field chain is CIC
deposit, one Gaussian `W` on charge, spectral Poisson, the same `W` on the
return field, then CIC gather.  Arms are CIC, cell-isotropic C4 at one cell,
and the independently selected exact LOO-CV width `7.244293147028428` cells;
the same seed initializes byte-identical low-PPC arms.  A separately generated
PPC-4096 CIC history for each seed is used only as the finite-PPC density and
pre-return-filter grid-field error-power reference.

The registered arm-specific result remains primary: C4/CIC is
`0.989415 [0.967884,1.013827]`, with 2.076 C4 e-foldings but only 1.721 CIC
e-foldings, so promotion fails.  A separate reducer takes the intersection of
the locked CIC and C4 windows, `[5,30.1]`, and resamples paired raw seed
histories.  It gives rates `-0.0685811` and `-0.0698686` and C4/CIC
`1.018773 [1.015359,1.021931]`.  Both common-window fits fail duration; the
result is sensitivity analysis only and changes no primary gate.

Local and MeluXina smoke/preflight checks pass.  On the cluster, C4 has
measured `W=.9951924506636148` and complete-coupling
`W^2=.9904080138578516`; all analytic/measured transfer discrepancies are at
most `2.22e-16`.  The archived remote preflight SHA-256 is
`f48de6f8...f5de`.  The v2 runner SHA-256 is `3c85334a...5cc1`; the cluster
preflight records the exact hashes of every reused runtime source.  Focused
regressions pass 11/11.

The launch was deliberately hybrid after the user reported about seven
CPU-node hours remaining.  CPU jobs `5200486` and `5200487` covered low seeds
`300..303` and reference seeds `300..304`.  Historical CPU maxima of 2317 and
3149 seconds per corresponding seed give a conservative total of 6.948
node-hours.  GPU jobs `5200485`, `5200488`, and retained tasks 1--3 of arrays
`5200458`, `5200457` covered the exact non-overlapping remainder.  Task 0 of
each original GPU array was canceled before it started and before it created
any output.  The exact job/seed map, launcher hashes, and scheduler state are
in
`results/weak_landau_confirmatory_v2_submission_20260911/hybrid_cpu_gpu_allocation.json`.
Slurm independently enforced the CPU-node-minute cap; CPU reference tasks 2--4
were initially held with `AssocGrpGRESMinutes` and started only when the
allocation admitted them.

No v1 or three-seed pilot history is pooled into v2.  The window-only and
comparative analyses were not submitted while simulations ran.  The
preregistered reducer first hashed a rate-free `window_decisions.json` and
reported each accepted endpoint, ending reason and fitted e-fold span before
any C4/CIC ratio was inspected.

Production has now completed.  All 17 active Slurm elements exited zero; the
nine CPU seeds consumed 6.8958 CPU-node hours.  The archived low/reference
tree has 360 files and matching remote/local aggregate SHA-256
`47019511...8a7a1`.  Stage 1 verified every low-tree file, all 40 within-seed
three-arm particle hashes, and the frozen runtime/preflight hashes.  Without
forming any between-arm ratio, it fixed the windows and outcomes below:

| Arm | Accepted window | End reason (censor time) | Gamma | Fitted amplitude e-foldings | Stage-1 quality |
|---|---:|---|---:|---:|---|
| CIC | `[5.0,30.1]` | five-floor threshold (`32.6`) | `-0.0685811` | `1.7214` | FAIL (duration) |
| C4, sigma=1 cell | `[5.0,35.6]` | five-floor threshold (`38.1`) | `-0.0678552` | `2.0764` | PASS |
| exact LOO-CV comparator | `[5.0,28.0]` | five-floor threshold (`30.5`) | `-0.129881` | `2.9873` | PASS |

The Stage-1 decision SHA-256 is `cb724768...0cbf`.  At that ordered checkpoint,
the primary promotion rule already failed because it requires both CIC and C4
to span at least two fitted amplitude e-foldings.  Stage 2 had not yet been
run, so rate-ratio and bootstrap results were deliberately uninspected until
the Stage-1 outcome was reported separately.

Following that separate Stage-1 report, the locked Stage-2 reducer verified
the unchanged decision hash and all input hashes.  With 5000 paired-seed
resamples, C4/CIC is `0.989415 [0.967884,1.013827]`; the CIC absolute-rate
ratio to the frozen fully discrete reference is
`1.033434 [0.989774,1.078710]`.  Both rate intervals pass, but the promotion
status remains FAIL because CIC has only 1.721386 fitted e-foldings and the
paired full-quality bootstrap fraction is zero.  In the same campaign and
against same-seed PPC-4096 CIC references, C4 reduces density-error power by
`3.25865 [3.25180,3.26500]` and pre-return-filter grid-field error power by
`1.61280 [1.60345,1.62155]`.  The exact LOO-CV comparator changes the rate by
about 89%, reduces density-error power by `4.61153`, and has grid-field
reduction `0.63238`, i.e. it worsens that field metric.  The result JSON hash
is `bd1f00e2...6eb4`; exact output hashes and scope notes are in
`results/weak_landau_confirmatory_v2_submission_20260911/stage2_manifest.json`.

## Source and input integrity

The manifest records current matching paths, while `SHA256SUMS.txt` records
hashes of the local manuscript, generators, recovered submit scripts, current
entry points, selected compact result files, and the conservation raw outputs
copied locally during this audit.  Other raw MeluXina trajectories that have
not been copied remain outside the local hash manifest.

## Noise-metric protocol audit

The supplied PDF mixes `4867515` per-cell noise reductions in Table 4 and Table
7 (`2.559`, `3.079`) with the global-shuffle MISE reduction from `4868286`
(`16.624`), while Fig. 7a takes its Gaussian C4 overlay (`5.643`) from the
latter campaign.  The two campaigns use the same full-grid density
squared-error-power metric, PPC, reference, time window and 51-snapshot
cadence; their evidenced difference is the particle velocity-quadrature
layout (randomized per cell versus globally shuffled).  The authoritative
manuscript now retains the measured Table 4 values while explicitly disclosing
the mixed protocols.  It does not replace them with the later measurements.

`reproducibility/NOISE_METRIC_AUDIT.md` documents the estimator, reference,
species normalization, all-mode Parseval scope, exact leave-one-out self term,
and the distinction between grid-density, pre-gather grid-field, and complete
particle-force quantities.  The strict regeneration is under
`reproducibility/results/noise_metric_audit/`.  It reproduces all 34 Gaussian,
44 filter-family and seven directional E2 archived rows, including their
bootstrap intervals.  Controlled deletion verifies the LOO quadratic-form
correction to at most `8.33e-17`.  The raw-binomial `n=2` matched-low-k-width
value is `5.871509`; it is not the separate Gaussian overlay value `5.643223`.

## Rate-analysis audit

`reproducibility/RATE_UNCERTAINTY_AUDIT.md` documents the recovered observable,
oscillatory envelopes, floor and bounce censoring, fit windows, seed pairing,
bootstrap counts and rejected-fit fractions.  The raw archives were copied
without merging into the existing summary tree.  The complete rebuild is under
`reproducibility/results/rate_uncertainty_audit_v8`; exact commands are in its
`commands.csv`.  Eighteen of nineteen reductions match archived tables
numerically, and the nineteenth differs only in the strong-Landau gate label.
Raw histories for the original E2 Gaussian corner and raw-binomial tables were
recovered and regenerated.  A final job-ID search on 2026-09-11 also recovered
all 30 delta-f reference histories from job `4872599`, all 40 nonzero-width
isotropic m=12 histories from `4872745` plus its ten-history linked baseline,
and all 140 mode-band histories from `4873221`.  Current matching reducers
reproduce the archived CSVs with maximum absolute numeric differences
`3.38e-14`, `3.66e-15`, and zero, respectively.  Raw-tree hashes and the
job-4872745 symlink materialization are recorded in
`results/recovered_delta_f_campaigns_20260911/recovery_manifest.json`.

## Validation-gate audit

`reproducibility/VALIDATION_AUDIT.md` and
`results/validation_audit_v9/validation_status.csv` apply the two-sided 10%
rate gate, two fitted e-foldings, fit-specific signal checks, protected-mode
coverage and raw traceability separately.  The final 40-row audit has 8 PASS,
31 FAIL and 1 UNRESOLVED row.  The binding-corner passes are Gaussian C4 and
the new direct p2/n2 and p3/n4 compensated runs; the independently seeded
`(10,1)` Gaussian test also passes. Historical longitudinal compensated rows
remain labelled `(12,0)` rather than being substituted for corner evidence. No
weak-Landau row passes the complete gate.  Recovery of the old per-seed arrays
adds two longitudinal/reference passes without changing the binding-corner or
weak-Landau conclusions.

The evidence-closure audit records that job `4867515`, which supplied the
Table-4 C4/Silverman density-error powers, used the same repeated per-cell
`32x32` velocity tensor as the rejected confirmation.  It has only 32 distinct
longitudinal nodes and none at the resonant phase velocity.  Those values are
therefore retained only as density-error measurements for that constructed
ensemble, not as same-campaign Landau-dynamics validation.  No recovered
Landau rate history is identified as per-cell; exact revisions of the oldest
rate jobs remain unknown.  On recovered E2 C4 corner histories, the explicit
pre-emergence-floor check gives fit-start SNR `11.213`; the shared CIC baseline
spans 4.484 e-foldings.  All four compensated E2 grid/time point predictions
lie outside their measured 95% intervals, although p2/n2 and p3/n4 still pass
the separate measured 10% rate-preservation gate.

## Secondary-claims audit

`pic_kde_2d/reproducibility/SECONDARY_CLAIMS_AUDIT.md` documents the recovered
delta-f, finite-amplitude, controller, locality and cost evidence. Machine
output is under `pic_kde_2d/reproducibility/results/secondary_claims_audit/`.
The audit establishes the exact marker-weight guard, adds a paired target-mode
energy rebound diagnostic, corrects the H1 density/field wording, records that
no learned controller was trained, and limits the cost result to fixed-width
steady-state time stepping. End-to-end selector cost, particle phase space,
harmonic spectra, and learned-controller performance remain unmeasured.

## Manuscript regeneration

The authoritative source is
`../Dynamics_constrained_Deposition/introduction.tex`; the package
`paper1.tex` is a retained historical audit artifact. Task 11 regenerates nine
figure PDFs plus 300-dpi PNGs, mirrors the PDFs into the authoritative `figs/`
directory, and compiles with `elsarticle` (`final,1p,times`). The verified build
has 37 pages. `FIGURE_PROVENANCE.md`,
`reproducibility/TABLE_PROVENANCE.md`, and
`reproducibility/CITATION_AUDIT.md` map the rendered content to evidence.
`reproducibility/audit_manuscript.py` fails on missing/unequal figure assets,
non-300-dpi PNGs, undefined citations/references, duplicate labels, rotated
tables, placeholders, stale E2 values, and selected audited numeric anchors.
Exact commands are in `reproducibility/REPRODUCIBILITY.md`.

## Latest evidence closure

- Added the paired common-window sensitivity interval while preserving the
  registered arm-specific result as primary.
- Narrowed the historical job-4867515 density-error claims after confirming
  that its per-cell velocity support misses the Landau resonant region.
- Filled the E2 C4 signal field from recovered histories (SNR 11.213) and made
  the shared CIC baseline-duration pass explicit.
- Separated compensated-filter prediction calibration from measured rate
  preservation; all four corner predictions lie outside their measured 95%
  intervals even though p2/n2 and p3/n4 remain preservation passes.
