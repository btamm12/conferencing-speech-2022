#!/bin/bash
#
#SBATCH --nice=900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=4-12:00:00
#SBATCH --mem=8000M
#SBATCH --gres=gpu_mem:4000

# Run:
# > slurm/submit_job.sh extract_val
srun python_executable.sh "src/predict_all_layer_fusion_41/make_predict_val.py"
