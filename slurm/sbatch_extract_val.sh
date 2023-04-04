#!/bin/bash
#
#SBATCH --nice=900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:00:00
#SBATCH --mem=8000M
#SBATCH --gres=gpu_mem:18000

# Run:
# > slurm/submit_job.sh extract_val
srun python_executable.sh "src/predict_layer_fusion_corrupted/make_extract_val.py"
