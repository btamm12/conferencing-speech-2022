#!/bin/bash
#
#SBATCH --output=out/train_example_%A_%a.out
#SBATCH --error=out/train_example_%A_%a.err
#SBATCH --nice=900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=20G
#SBATCH --gres=gpu_mem:22000
#SBATCH --array=0-138%1

# Run:
# > slurm/submit_array_job.sh train_example

# array 0-138%1 => 139 stages (full training)
# array 0-3%1 => 4 stages (300m feature extraction layers 0:25 + training for layers 0 and 1)
srun python_executable.sh "src/train/make_train.py --stage ${SLURM_ARRAY_TASK_ID} --example --use_caching"
