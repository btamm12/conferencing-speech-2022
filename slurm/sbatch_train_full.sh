#!/bin/bash
#
#SBATCH --output=out/train_full_%A_%a.out
#SBATCH --error=out/train_full_%A_%a.err
#SBATCH --nice=900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=116000M
#SBATCH --gres=gpu_mem:21000
#SBATCH --array=0-29%1

# Run:
# > slurm/submit_array_job.sh train_example

# array 0-29%1 => 30 stages (full training, only 300M, 100% data: requires 176 GB memory so spills into swap)
# array 0-10%1 => 11 stages (300m feature extraction + training for layers 0:9)
srun python_executable.sh "src/train/make_train_full.py --stage ${SLURM_ARRAY_TASK_ID} --use_caching"
