#!/bin/bash -l
#SBATCH --job-name=kde-momentum-audit
#SBATCH --account=p201103
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/tier2/users/u103576/logs/kde_momentum_audit_%j.out
#SBATCH --error=/mnt/tier2/users/u103576/logs/kde_momentum_audit_%j.err

set -euo pipefail

ROOT=/home/users/u103576/hexapic-main
SRC="${ROOT}/source"
AUDIT="${ROOT}/pic_kde_2d/hexapic_momentum_audit"
INPUT_SOURCE="${ROOT}/tests/landau-1d-python/hexapic_run/landau_kde_ny4.inp"
OUT="${AUDIT}/results_${SLURM_JOB_ID}"

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

mkdir -p "${OUT}"
cd "${SRC}"
if [ ! -e HEXAPIC.pre_momentum_audit ]; then
    cp HEXAPIC HEXAPIC.pre_momentum_audit
fi
make -f makefile_HPCFS -j "${SLURM_CPUS_PER_TASK}"

run_case() {
    local shape="$1"
    local split="$2"
    local tol="$3"
    local case_dir="${OUT}/${shape}__tol${tol}"
    mkdir -p "${case_dir}"
    cp "${INPUT_SOURCE}" "${case_dir}/audit.inp"
    cd "${case_dir}"
    export HEXAPIC_KDE_SPLIT_SIGMA_CELLS="${split}"
    export HEXAPIC_HYPRE_TOL="${tol}"
    srun -n 1 "${SRC}/HEXAPIC" audit.inp -steps 1 -dsteps 1 -quiet > run.log 2>&1
}

for tol in 1e-5 1e-9; do
    run_case equal_0p5 "0.5,0.5" "${tol}"
    run_case equal_3p0 "3.0,3.0" "${tol}"
    run_case unequal_0p5_3p0 "0.5,3.0" "${tol}"
done

python3 "${AUDIT}/analyze_audit.py" "${OUT}"
echo "RESULT_DIR=${OUT}"
