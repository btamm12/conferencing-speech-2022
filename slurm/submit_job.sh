#!/bin/bash
SLURM_DIR=$(realpath $(dirname $0))
OUT_DIR=${SLURM_DIR}/out
mkdir -p $OUT_DIR
echo "Running sbatch_${1}.sh from slurm directory."
sbatch \
--chdir $SLURM_DIR \
--job-name ${1} \
--error ${OUT_DIR}/${1}_%A.err \
--output ${OUT_DIR}/${1}_%A.out \
${SLURM_DIR}/sbatch_${1}.sh 
