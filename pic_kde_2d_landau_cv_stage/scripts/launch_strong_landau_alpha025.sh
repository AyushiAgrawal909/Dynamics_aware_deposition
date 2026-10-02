#!/bin/bash
set -euo pipefail
PROJECT=/home/users/u103576/hexapic-main/pic_kde_2d
TAG=${1:-$(date +%Y%m%d_%H%M%S)}
ROOT=${PROJECT}/results/strong_landau_alpha025_${TAG}
CV_ROOT=${ROOT}/cv_shards
RATE_ROOT=${ROOT}/rate_shards
ANALYSIS_ROOT=${ROOT}/analysis
cd "${PROJECT}"
CV_JOB=$(sbatch --parsable --export=ALL,OUTPUT_ROOT="${CV_ROOT}" \
  scripts/submit_strong_landau_alpha025_cv_cpu.sbatch)
RATE_JOB=$(sbatch --parsable --export=ALL,OUTPUT_ROOT="${RATE_ROOT}" \
  scripts/submit_strong_landau_alpha025_rates_cpu.sbatch)
ANALYSIS_JOB=$(sbatch --parsable \
  --dependency=afterok:${CV_JOB}:${RATE_JOB} \
  --export=ALL,CV_ROOT="${CV_ROOT}",RATE_ROOT="${RATE_ROOT}",OUTPUT_ROOT="${ANALYSIS_ROOT}" \
  scripts/submit_strong_landau_alpha025_analysis_cpu.sbatch)
printf 'ROOT=%s\nCV_JOB=%s\nRATE_JOB=%s\nANALYSIS_JOB=%s\n' \
  "${ROOT}" "${CV_JOB}" "${RATE_JOB}" "${ANALYSIS_JOB}"
