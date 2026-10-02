#!/bin/bash -l
#SBATCH --job-name=kde-momentum-full
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --array=0-11%6
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/kde_momentum_full_%A_%a.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/kde_momentum_full_%A_%a.err

set -euo pipefail

ROOT=/home/users/u103576/hexapic-main
SRC="${ROOT}/source"
AUDIT="${ROOT}/pic_kde_2d/hexapic_momentum_audit"
INPUT_SOURCE="${ROOT}/tests/landau-1d-python/hexapic_run/landau_kde_ny4.inp"
OUT="${AUDIT}/full_results_${SLURM_ARRAY_JOB_ID}"
NSTEPS=1500

module purge
module load env/release/2024.1
module load foss/2024a
module load Python/3.11.10-GCCcore-13.3.0
module load Hypre/2.31.0-foss-2024a
module load bzip2/1.0.8-GCCcore-13.3.0
module load libpciaccess/0.18.1-GCCcore-13.3.0

export OPENPMD_ROOT=/project/home/p201103/software/openpmd
export ADIOS2_ROOT=/project/home/p201103/software/adios2
export LD_LIBRARY_PATH="${OPENPMD_ROOT}/lib:${OPENPMD_ROOT}/lib64:${ADIOS2_ROOT}/lib:${ADIOS2_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${OPENPMD_ROOT}/lib/pkgconfig:${OPENPMD_ROOT}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export OMP_NUM_THREADS=1
export HEXAPIC_RANDOM_SEED=1729
export HEXAPIC_MOMENTUM_AUDIT=1
export HEXAPIC_HYPRE_SPD_FORM=1
export HEXAPIC_HYPRE_PIN_GAUGE=0
export HEXAPIC_HYPRE_DIAGONAL_SHIFT=1e-6
export HEXAPIC_HYPRE_RELAX_TYPE=1

shapes=(equal_0p5 equal_3p0 unequal_0p5_3p0)
splits=("0.5,0.5" "3.0,3.0" "0.5,3.0")
tolerances=(1e-5 1e-9)
starts=(warm cold)

shape_index=$((SLURM_ARRAY_TASK_ID % 3))
tol_index=$(((SLURM_ARRAY_TASK_ID / 3) % 2))
start_index=$((SLURM_ARRAY_TASK_ID / 6))
shape="${shapes[$shape_index]}"
split="${splits[$shape_index]}"
tol="${tolerances[$tol_index]}"
start="${starts[$start_index]}"
case_dir="${OUT}/${shape}__tol${tol}__${start}"

mkdir -p "${case_dir}"
cp "${INPUT_SOURCE}" "${case_dir}/audit.inp"
# Reduce to 8 ppc while retaining the same quiet-start benchmark and seed.
sed -i 's/ParticleWeight = .*/ParticleWeight = 2.080645823188331e+05/' "${case_dir}/audit.inp"

export HEXAPIC_KDE_SPLIT_SIGMA_CELLS="${split}"
export HEXAPIC_HYPRE_TOL="${tol}"
if [ "${start}" = cold ]; then
    export HEXAPIC_HYPRE_COLD_START=1
else
    export HEXAPIC_HYPRE_COLD_START=0
fi

cd "${case_dir}"
srun -n 1 "${SRC}/HEXAPIC" audit.inp -steps "${NSTEPS}" -dsteps "${NSTEPS}" -quiet > run.log 2>&1
echo "CASE_DIR=${case_dir}"
