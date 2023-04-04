#!/bin/bash
#
#SBATCH --nice=900
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1-12:00:00
#SBATCH --mem=8000M
#SBATCH --gres=gpu_mem:11900

# Run:
# > slurm/submit_job.sh extract_val
srun python_executable.sh "src/train_lf_mfcc_4ds/make_train.py -i xlsr -x wav2vec2-xls-r-2b -c 3"
