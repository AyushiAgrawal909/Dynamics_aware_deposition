#!/bin/bash -l
#SBATCH --job-name=kde-hypre-tol
#SBATCH --account=p201103
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/kde_hypre_tol_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/kde_hypre_tol_%j.err

set -euo pipefail

ROOT=/home/users/u103576/hexapic-main
SRC="${ROOT}/source"
AUDIT="${ROOT}/pic_kde_2d/hexapic_momentum_audit"
INPUT_SOURCE="${ROOT}/tests/landau-1d-python/hexapic_run/landau_kde_ny4.inp"
OUT="${AUDIT}/hypre_tolerance_one_step_${SLURM_JOB_ID}"

module purge
module load env/release/2024.1
module load foss/2024a
module load Hypre/2.31.0-foss-2024a
module load bzip2/1.0.8-GCCcore-13.3.0
module load libpciaccess/0.18.1-GCCcore-13.3.0

export OPENPMD_ROOT=/project/home/p201103/software/openpmd
export ADIOS2_ROOT=/project/home/p201103/software/adios2
export LD_LIBRARY_PATH="${OPENPMD_ROOT}/lib:${OPENPMD_ROOT}/lib64:${ADIOS2_ROOT}/lib:${ADIOS2_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${OPENPMD_ROOT}/lib/pkgconfig:${OPENPMD_ROOT}/lib64/pkgconfig:${ADIOS2_ROOT}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export OMP_NUM_THREADS=1
export HEXAPIC_RANDOM_SEED=1729
export HEXAPIC_MOMENTUM_AUDIT=1
export HEXAPIC_KDE_SMOOTH_SIGMA_CELLS="0.5,3.0,6.0,0.5"
export HEXAPIC_KDE_GRAD_H=1
export HEXAPIC_KDE_DERIVATIVE_AUDIT=1
export HEXAPIC_HYPRE_SPD_FORM=1
export HEXAPIC_HYPRE_PIN_GAUGE=0
export HEXAPIC_HYPRE_DIAGONAL_SHIFT=0
export HEXAPIC_HYPRE_RELAX_TYPE=1
export HEXAPIC_HYPRE_COLD_START=1

mkdir -p "${OUT}"
git -C "${ROOT}" rev-parse HEAD > "${OUT}/code_revision.txt"
git -C "${ROOT}" status --short > "${OUT}/code_status.txt"
sha256sum \
  "${SRC}/HEXAPIC" \
  "${SRC}/field_solver_hypre.cpp" \
  "${SRC}/kde_deposition.cpp" \
  "${SRC}/kde_deposition.hpp" \
  "${SRC}/particles.cpp" > "${OUT}/code_sha256.txt"

tolerances=(1e-5 1e-7 1e-9 1e-11)

run_tolerance() (
  local tolerance="$1"
  local tag="${tolerance/-/m}"
  local case_dir="${OUT}/tol_${tag}"
  mkdir -p "${case_dir}"
  cp "${INPUT_SOURCE}" "${case_dir}/audit.inp"
  sed -i 's/ParticleWeight = .*/ParticleWeight = 2.080645823188331e+05/' "${case_dir}/audit.inp"
  export HEXAPIC_HYPRE_TOL="${tolerance}"
  cd "${case_dir}"
  srun --exclusive -N 1 -n 1 -c 1 "${SRC}/HEXAPIC" audit.inp \
    -steps 1 -dsteps 1 -quiet > run.log 2>&1
  grep -E "KDE_CHARGE_AUDIT|RHS_AUDIT|HYPRE_AUDIT step|KDE_DERIVATIVE_AUDIT|MOMENTUM_AUDIT step" \
    run.log > audit_extract.log
)

for tolerance in "${tolerances[@]}"; do
  run_tolerance "${tolerance}" &
done
wait
