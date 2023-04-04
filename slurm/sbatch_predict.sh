#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu_mem:6000

srun make_executable.sh predict