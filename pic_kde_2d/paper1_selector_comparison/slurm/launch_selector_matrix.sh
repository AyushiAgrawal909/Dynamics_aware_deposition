#!/bin/bash
set -euo pipefail

cd /home/users/u103576/hexapic-main/pic_kde_2d

e2_job=$(sbatch --parsable paper1_selector_comparison/slurm/stage1_e2_noise_array_cpu.sbatch)
e2_root=/mnt/tier2/users/u103576/logs/p1_e2_noise_${e2_job}
e2_reduce=$(sbatch --parsable --dependency=afterok:${e2_job} \
  --export=ALL,INPUT_ROOT=${e2_root},OUTPUT_ROOT=${e2_root}/reduced,BENCHMARK=e2 \
  paper1_selector_comparison/slurm/reduce_noise_cpu.sbatch)

landau_job=$(sbatch --parsable paper1_selector_comparison/slurm/stage1_landau_noise_array_cpu.sbatch)
landau_root=/mnt/tier2/users/u103576/logs/p1_landau_noise_${landau_job}
landau_reduce=$(sbatch --parsable --dependency=afterok:${landau_job} \
  --export=ALL,INPUT_ROOT=${landau_root},OUTPUT_ROOT=${landau_root}/reduced,BENCHMARK=landau \
  paper1_selector_comparison/slurm/reduce_noise_cpu.sbatch)

landau_rates=$(sbatch --parsable paper1_selector_comparison/slurm/stage2_landau_exact_rates_cpu.sbatch)
e2_rates=$(sbatch --parsable paper1_selector_comparison/slurm/stage2_e2_exact_rates_cpu.sbatch)

printf 'e2_noise=%s e2_reduce=%s e2_root=%s\n' "${e2_job}" "${e2_reduce}" "${e2_root}"
printf 'landau_noise=%s landau_reduce=%s landau_root=%s\n' "${landau_job}" "${landau_reduce}" "${landau_root}"
printf 'landau_exact_rates=%s e2_exact_rates=%s\n' "${landau_rates}" "${e2_rates}"
