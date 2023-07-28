#!/usr/bin/env bash
echo "Running job_exectuable.sh"
export PYTHONPATH=""
echo "Activating virtual environment."
source /esat/audioslave/btamm/miniconda3/bin/activate py39_lightning

# Exit if we receive a non-zero exit code!
# Source: https://stackoverflow.com/a/821419
set -e

python3 --version

# Find local execute folder.
_exec_dir="$_CONDOR_SCRATCH_DIR"
_exec_machine="${HOSTNAME%%.*}"
echo "HOSTNAME=$HOSTNAME"
echo "_exec_machine=$_exec_machine"
echo "_CONDOR_SCRATCH_DIR=$_CONDOR_SCRATCH_DIR"
echo "_CONDOR_SLOT=$_CONDOR_SLOT"
#_exec_dir="$(find /usr/data/condor/execute -maxdepth 1 -user btamm -printf "%T@ %p\n" | sort -n | head -n 1 | cut -d " " -f 2)"

# Unzip features to execute folder.
_feat="wav2vec2-xls-r-2b"
_machine="audioslave"
_root_dir="/usr/data/btamm/data/conferencing-speech-2022/data/processed"
_src_train_dir="${_machine}:${_root_dir}/train/features/${_feat}"
_src_val_dir="${_machine}:${_root_dir}/val/features/${_feat}"
_dst_train_dir="${_exec_dir}/btamm_working/train/features"
_dst_val_dir="${_exec_dir}/btamm_working/val/features"
mkdir -p "$_dst_train_dir"
mkdir -p "$_dst_val_dir"

echo "copying train features"
echo "src: ${_src_train_dir}"
echo "dst: ${_dst_train_dir}"
_pwd=$(pwd)
cd "$_dst_train_dir" # Must cd first, otherwise rsync gives error
rsync -pr --info=progress2 -e ssh "$_src_train_dir" .
#scp -q -pr "$_src_train_dir" "$_dst_train_dir"

echo "copying val features"
echo "src: ${_src_val_dir}"
echo "dst: ${_dst_val_dir}"
cd "$_dst_val_dir"
rsync -pr --info=progress2 -e ssh "$_src_val_dir" .
#scp -q -pr "$_src_val_dir" "$_dst_val_dir"

cd "$_pwd"

echo "Creating symlinks"
_sym_id="${_exec_machine}_${_CONDOR_SLOT}"
rm -f "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/train/features_localsym_${_sym_id}"
rm -f "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/val/features_localsym_${_sym_id}"
ln -s "${_exec_dir}/btamm_working/train/features" "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/train/features_localsym_${_sym_id}"
ln -s "${_exec_dir}/btamm_working/val/features" "/users/psi/btamm/GitHub/btamm12/conferencing-speech-2022/data/processed/val/features_localsym_${_sym_id}"

# Run the desired make command.
echo "Running python3 $1"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES#CUDA}
python3 $1
