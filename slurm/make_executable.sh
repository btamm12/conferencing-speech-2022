#!/bin/bash
SLURM_DIR=$(dirname $0)
ROOT_DIR=$(realpath ${SLURM_DIR}/..)
cd $ROOT_DIR
source venv/bin/activate
echo "Running make $1."
make $1
