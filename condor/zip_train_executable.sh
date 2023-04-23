#!/usr/bin/env bash
echo "Running zip_train_exectuable.sh"
export PYTHONPATH=""

# Exit if we receive a non-zero exit code!
# Source: https://stackoverflow.com/a/821419
set -e


# Run the desired make command.
echo "zipping"
_root_dir="/esat/audioslave/btamm/data/conferencing-speech-2022/data/processed/train/features"
cd "$_root_dir"
zip -1 -r wav2vec2-xls-r-1b.zip wav2vec2-xls-r-1b/
