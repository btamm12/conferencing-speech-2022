#!/bin/bash
SLURM_DIR=$(dirname $0)
ROOT_DIR=$(realpath ${SLURM_DIR}/..)
cd $ROOT_DIR
source venv/bin/activate
cd src/dnsmos
echo "running dnsmos"
./run_dnsmos.sh
echo "finished"
