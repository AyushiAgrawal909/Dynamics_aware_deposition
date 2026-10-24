# T2 production scan: Meluxina submission record

Submitted 2026-10-24 from commit `43ca045` (branch `claude/t2-meluxina-submit`, on top of
merge `afb2d94` of PR #4). `attractor_sim/`, `attractor_analysis/`, `scripts/` and
`configs/t2_waterbag_dv020.yaml` are identical to `frozen_commit` 594332a (empty `git diff`).

* Account `p201103`, partition `gpu`, QOS `default`, 1 node, 64 cores per task. The code is CPU-only (numba), so the GPUs stay idle.
  The CPU allocation was 100% used this month and the `nocredit` account allows 0 running jobs.
* Mail: `ayushi.ph@sric.iitr.ac.in`, `--mail-type=ALL` (one mail per array start and end).
* Paths on Meluxina: repo `/project/home/p201103/u103576/Dynamics_aware_deposition`, venv `/project/home/p201103/u103576/venvs/nta`
  (env/release/2024.1 + Python 3.12.3, numpy 2.5.4, numba 0.68.0), bundles `/project/home/p201103/u103576/nta_t2/bundles`,
  logs `/project/home/p201103/u103576/nta_t2/logs`.
* The job list `slurm/t2_jobs.txt` (52 runs) matches `scripts/t2_scan.py --list` output on Meluxina line for line.

## Timing basis (64-core GPU node, test jobs 5382027 and 5382049)

| Run | Measured | Full run (96 000 steps, 480 snapshots) |
|---|---|---|
| Vlasov L0 (nv = 12800) | 0.437 s/step, about 13 s per diagnostic snapshot, 26 s start-up | about 13.4 h |
| PIC L0 (16384 ppc) | 0.071 s/step including diagnostics, 117 s start-up | about 1.9 h |

The other levels are scaled by their size: each level halves the work. Expected total is about 50 GPU node-hours.
Before submission, myquota showed 261 of 339 GPU node-hours used (about 78 left).

## Submitted

| Group | Runs | Tasks | --time | Job ID |
|---|---|---|---|---|
| g1_vlasov_L0 | Vlasov dv = dv_b/128 | 1 | 20:00:00 | 5382069 |
| g2_vlasov_L1 | Vlasov dv = dv_b/64 | 1 | 10:00:00 | 5382070 |
| g3_vlasov_L2-3 | Vlasov dv_b/32, dv_b/16 | 2 | 05:00:00 | 5382071 |
| g4_pic_L0 | PIC 16384 ppc, seeds 0-3, and control seeds 0-1 | 6 | 03:00:00 | 5382072 |
| g5_pic_L1 | PIC 8192 ppc, the same seeds | 6 | 01:45:00 | 5382073 |
| g6_pic_L2 | PIC 4096 ppc, the same seeds | 6 | 01:00:00 | 5382074 |
| g7_pic_L3-7 | PIC 2048 to 128 ppc, the same seeds | 30 | 00:45:00 | 5382075 |
| L-doubling check | PIC 1024 ppc (level 4), seed 0, 32 wavelengths, `configs/t2_waterbag_dv020_L32.yaml` | 1 | 01:00:00 | 5382085 |

That covers the whole `t2_scan.py` matrix (52 runs) and the protocol's L-doubling check.

The L-doubling check runs from a separate worktree at commit `9d48383`
(`/project/home/p201103/u103576/nta_t2/repo_L32`) and writes to `nta_t2/bundles_L32`, so its bundle doesn't collide with
`pic1d1v_L4_s0.npz` from the main scan. Its config only doubles the box and nx (same dx), and keeps the seeded wavenumbers,
amplitudes and phases, renumbered as modes 2, 4, ..., 52. The extra modes in the doubled box are not seeded.

## GPU budget

| | node-hours |
|---|---|
| Left at submission (myquota: 261 of 339 used) | about 78 |
| Test jobs (5382021, 5382027, 5382049) | about 0.3 |
| Expected for the scan and the L-doubling check | about 50.5 |
| Worst case if every task ran to its wall limit | about 98 |

The worst case is above the quota, but the wall limits are 1.3-1.6x the measured estimates. Only a job that hangs would get close.
Check with `myquota` (it lags by about a day) and `sacct -u $USER -S 2026-10-24 -o jobid,jobname,elapsed,state`.

## Not submitted (for November or a later decision)

1. **Conditional seeds 4-7** (`conditional_rules.onset_scatter`): if sigma(t_on) from the 4 finest-PIC seeds is > 0.1 of
   the S6 effect size, run seeds 4-7 at every PIC level before the analysis. To do that, add the seeds to a job list and submit with `t2_array.sbatch`.
2. **Analysis**, after every bundle is in: `sbatch slurm/t2_analyze.sbatch`.

## Re-submitting a group (for example after a failure, or in November)

From `experiments/nonthermal_attractor` on Meluxina:

    bash slurm/t2_submit.sh g4_pic_L0            # one group
    bash slurm/t2_submit.sh                       # every group, finest Vlasov first

Wall times are in `slurm/t2_submit.sh`. If the GPU quota is used up and CPU hours are back, switch `t2_array.sbatch` to
`--partition=cpu --cpus-per-task=128`.
