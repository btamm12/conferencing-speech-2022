#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:05:00
#SBATCH --mem=4G

srun cp_executable.sh