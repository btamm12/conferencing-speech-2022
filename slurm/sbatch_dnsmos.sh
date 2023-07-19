#!/bin/bash
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --mem=12G

srun dnsmos_executable.sh