#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=6G
#SBATCH --gres=gpu_mem:24000

srun make_executable.sh download