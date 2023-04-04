#!/bin/bash
SLURM_DIR=$(realpath $(dirname $0))
OUT_DIR=${SLURM_DIR}/out
mkdir -p $OUT_DIR
echo "Running sbatch_${1}.sh from slurm directory."
sbatch \
--chdir $SLURM_DIR \
--job-name ${1} \
${SLURM_DIR}/sbatch_${1}.sh 
