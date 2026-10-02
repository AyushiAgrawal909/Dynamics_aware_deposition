#!/bin/bash
set -euo pipefail

PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
TAG=${1:-$(date +%Y%m%d_%H%M%S)}
ROOT=${PROJECT}/results/compensated_binomial_${TAG}
RATE_ROOT=${ROOT}/rates
NOISE_ROOT=${ROOT}/noise
ANALYSIS_ROOT=${ROOT}/analysis
cd "${PROJECT}"

RATE_JOB=$(sbatch --parsable \
  --export=ALL,OUTPUT_ROOT="${RATE_ROOT}" \
  scripts/submit_compensated_binomial_rates_cpu.sbatch)
NOISE_JOB=$(sbatch --parsable \
  --export=ALL,OUTPUT_ROOT="${NOISE_ROOT}" \
  scripts/submit_compensated_binomial_noise_cpu.sbatch)
ANALYSIS_JOB=$(sbatch --parsable \
  --dependency=afterok:${RATE_JOB}:${NOISE_JOB} \
  --export=ALL,RATE_ROOT="${RATE_ROOT}",NOISE_ROOT="${NOISE_ROOT}",OUTPUT_ROOT="${ANALYSIS_ROOT}" \
  scripts/submit_compensated_binomial_analysis_cpu.sbatch)

printf 'ROOT=%s\nRATE_JOB=%s\nNOISE_JOB=%s\nANALYSIS_JOB=%s\n' \
  "${ROOT}" "${RATE_JOB}" "${NOISE_JOB}" "${ANALYSIS_JOB}"
