#!/bin/bash
# Submit the T2 matrix on Meluxina in wall-time groups (timings: slurm/T2_SUBMISSION.md).
# Usage, from experiments/nonthermal_attractor:  bash slurm/t2_submit.sh [group ...]
# With no argument every group is submitted, finest Vlasov first.
set -euo pipefail
declare -A WALL=( [g1_vlasov_L0]=20:00:00 [g2_vlasov_L1]=10:00:00 [g3_vlasov_L2-3]=05:00:00
                  [g4_pic_L0]=03:00:00 [g5_pic_L1]=01:45:00 [g6_pic_L2]=01:00:00 [g7_pic_L3-7]=00:45:00 )
GROUPS_ALL="g1_vlasov_L0 g2_vlasov_L1 g3_vlasov_L2-3 g4_pic_L0 g5_pic_L1 g6_pic_L2 g7_pic_L3-7"
for g in ${@:-$GROUPS_ALL}; do
  f=slurm/t2_jobs_$g.txt
  id=$(JOBS=$f sbatch --parsable --export=ALL,JOBS=$f --job-name=nta-t2-$g \
       --array=1-$(wc -l < "$f") --time=${WALL[$g]} slurm/t2_array.sbatch)
  echo "$g $id $(wc -l < "$f") tasks, --time=${WALL[$g]}"
done
