#!/bin/sh
#use-everything
#SBATCH --time=2-00:00:00  # -- first number is days requested, second number is hours.  After this time the job is cancelled.
#SBATCH --partition=mcdermott
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=salavill@mit.edu # -- use this to send an automated email when:
#SBATCH --out=ecapa/outlogs/testing_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=any-A100

echo $(hostname)
job_idx="$SLURM_ARRAY_TASK_ID"

source activate metamer310

sound_path=/om2/user/amagaro/voice-speech-metamers/metamers_pipeline/kell2018/metamers/psychophysics_wsj400_jsintest_inversion_loss_layer_RS0_I3000_N8/0_SOUND_million/orig.wav


python -u ./make_metamer_joint.py "$job_idx" "$sound_path"
