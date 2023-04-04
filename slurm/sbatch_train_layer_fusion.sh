#!/bin/bash
#
#SBATCH --nice=900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=10-8:00:00
#SBATCH --mem=8000M
#SBATCH --gres=gpu_mem:10000

# Run:
# > slurm/submit_job.sh train_example
srun python_executable.sh "src/train_layer_fusion/make_train_layer_fusion.py"
