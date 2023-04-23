#!/usr/bin/env bash
echo "Running job_exectuable.sh"
export PYTHONPATH=""
echo "Activating virtual environment."
source /esat/audioslave/btamm/miniconda3/bin/activate py39_lightning

# Exit if we receive a non-zero exit code!
# Source: https://stackoverflow.com/a/821419
set -e


# Install packages.
# ROOT_DIR="$(realpath .)"
# VENV_DIR="$ROOT_DIR/venv"
# VENV_PATHS="$VENV_DIR:$VENV_DIR/bin"
# SPCH_DIR="/users/spraak/spch/prog/spch/Python-3.6.8"
# SPCH_PATHS="$SPCH_DIR:$SPCH_DIR/bin"
# PYTHON_PATH="$SPCH_DIR/bin/python3"
# PYTHON_INTERPRETER="$VENV_DIR/bin/python3"

# rm -rf $VENV_DIR/bin/python3
# ln -s $PYTHON_PATH $VENV_DIR/bin/python3

python3 --version

# PYTHONPATHS="$ROOT_DIR:$SPCH_DIR/lib/python3.6/site-packages:$VENV_DIR/lib/python3.6/site-packages"
# export PYTHONPATH="${PYTHONPATHS}${PYTHONPATH:+:$PYTHONPATH}"
# PATHS="$VENV_PATHS:$SPCH_PATHS"
# export PATH="${PATHS}${PATH:+:${PATH}}"
# export LD_LIBRARY_PATH="$VENV_DIR/lib64/python3.6/site-packages${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# # https://stackoverflow.com/a/37868546
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8
# # PYTHON_INTERPRETER="$VENV_DIR/bin/python3"
# PYTHON_INTERPRETER="/usr/bin/python3.6"
# $PYTHON_INTERPRETER -m pip install -U pip setuptools wheel
# $PYTHON_INTERPRETER -m pip install -r requirements_condor.txt
# # https://github.com/huggingface/transformers/issues/8638#issuecomment-790772391
# $PYTHON_INTERPRETER -m pip uninstall dataclasses -y


# # Activate virtual environment.


# Find local execute folder.
_exec_dir="$(find /usr/data/condor/execute -maxdepth 1 -user btamm | head -n 1)"

# Unzip features to execute folder.
mkdir -p "${_exec_dir}/btamm_working/train/features"
mkdir -p "${_exec_dir}/btamm_working/val/features"
echo "copying train features"
cp -p "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/train/features_real/wav2vec2-xls-r-1b.zip" "${_exec_dir}/btamm_working/train/features"
echo "copying val features"
cp -p "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/val/features_real/wav2vec2-xls-r-1b.zip" "${_exec_dir}/btamm_working/val/features"
echo "Unzipping train features"
unzip -q "${_exec_dir}/btamm_working/train/features/wav2vec2-xls-r-1b.zip" -d "${_exec_dir}/btamm_working/train/features"
echo "Unzipping val features"
unzip -q "${_exec_dir}/btamm_working/val/features/wav2vec2-xls-r-1b.zip" -d "${_exec_dir}/btamm_working/val/features"
# mv "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/train/features" "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/train/features.real"
# mv "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/val/features" "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/val/features.real"
echo "Creating symlinks"
rm -f "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/train/features"
rm -f "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/val/features"
ln -s "${_exec_dir}/btamm_working/train/features" "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/train/features"
ln -s "${_exec_dir}/btamm_working/val/features" "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/val/features"

# Run the desired make command.
echo "Running python3 $1"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA}
python3 $1