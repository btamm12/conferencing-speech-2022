#!/bin/bash
#
#SBATCH --nice=900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1-12:00:00
#SBATCH --mem=12000M
#SBATCH --gres=gpu_mem:18000

# Run:
# > slurm/submit_job.sh extract_val
srun python_executable.sh "src/train_lf_mfcc_4ds/extract_features_parallel.py"
