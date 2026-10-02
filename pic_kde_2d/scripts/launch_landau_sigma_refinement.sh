#!/bin/bash
set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
cd "${PROJECT}"

simulation_job=$(sbatch --parsable scripts/submit_landau_sigma_refinement_packed_cpu.sbatch)
output_root="${PROJECT}/results/landau_sigma_refinement_${simulation_job}"
reduce_job=$(sbatch --parsable \
  --dependency="afterok:${simulation_job}" \
  --export="ALL,INPUT_ROOT=${output_root}" \
  scripts/submit_landau_sigma_refinement_reduce_cpu.sbatch)

printf 'simulation_job=%s\nreduce_job=%s\noutput_root=%s\n' \
  "${simulation_job}" "${reduce_job}" "${output_root}"
