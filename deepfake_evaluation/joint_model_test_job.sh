#!/bin/bash
#
#SBATCH --job-name=joint_model_eval
#SBATCH --out="hslurm-%A_%a.out"
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
##SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --array=0
#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --partition=normal

jobs_per_source_file=10
offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))

echo $(hostname)

source activate /om2/user/annesyab/summer2023/anaconda/envs/metamer_310
python -u joint_model_test.py \
-idx_job "$job_idx" \