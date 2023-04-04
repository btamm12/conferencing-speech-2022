#!/bin/bash
SLURM_DIR=$(dirname $0)
ROOT_DIR=$(realpath ${SLURM_DIR}/..)
cd $ROOT_DIR
echo "copying 30% HDD features -> SSD"
mkdir -p scratch/train_features/
cp -pr data/processed/train/features/NISQA_VAL_LIVE scratch/train_features/
cp -pr data/processed/train/features/NISQA_TEST_P501 scratch/train_features/
cp -pr data/processed/train/features/NISQA_TEST_FOR scratch/train_features/
cp -pr data/processed/train/features/NISQA_TEST_LIVETALK scratch/train_features/
cp -pr data/processed/train/features/NISQA_TRAIN_LIVE scratch/train_features/
cp -pr data/processed/train/features/NISQA_VAL_SIM scratch/train_features/
cp -pr data/processed/train/features/withReverberationTrainDev scratch/train_features/
cp -pr data/processed/train/features/withoutReverberationTrainDev scratch/train_features/
cp -pr data/processed/train/features/NISQA_TRAIN_SIM scratch/train_features/
cp -pr data/processed/train/features/16k_speech scratch/train_features/
echo "removing 30% HDD features"
rm -rf data/processed/train/features/NISQA_VAL_LIVE
rm -rf data/processed/train/features/NISQA_TEST_P501
rm -rf data/processed/train/features/NISQA_TEST_FOR
rm -rf data/processed/train/features/NISQA_TEST_LIVETALK
rm -rf data/processed/train/features/NISQA_TRAIN_LIVE
rm -rf data/processed/train/features/NISQA_VAL_SIM
rm -rf data/processed/train/features/withReverberationTrainDev
rm -rf data/processed/train/features/withoutReverberationTrainDev
rm -rf data/processed/train/features/NISQA_TRAIN_SIM
rm -rf data/processed/train/features/16k_speech
echo "creating symlinks to SSD"
cd data/processed/train/features/
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/NISQA_VAL_LIVE .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/NISQA_TEST_P501 .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/NISQA_TEST_FOR .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/NISQA_TEST_LIVETALK .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/NISQA_TRAIN_LIVE .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/NISQA_VAL_SIM .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/withReverberationTrainDev .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/withoutReverberationTrainDev .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/NISQA_TRAIN_SIM .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/train_features/16k_speech .
echo "finished"

cd $ROOT_DIR
echo "copying 30% HDD features -> SSD"
mkdir -p scratch/val_features/
cp -pr data/processed/val/features/NISQA_VAL_LIVE scratch/val_features/
cp -pr data/processed/val/features/NISQA_TEST_P501 scratch/val_features/
cp -pr data/processed/val/features/NISQA_TEST_FOR scratch/val_features/
cp -pr data/processed/val/features/NISQA_TEST_LIVETALK scratch/val_features/
cp -pr data/processed/val/features/NISQA_TRAIN_LIVE scratch/val_features/
cp -pr data/processed/val/features/NISQA_VAL_SIM scratch/val_features/
cp -pr data/processed/val/features/withReverberationTrainDev scratch/val_features/
cp -pr data/processed/val/features/withoutReverberationTrainDev scratch/val_features/
cp -pr data/processed/val/features/NISQA_TRAIN_SIM scratch/val_features/
cp -pr data/processed/val/features/16k_speech scratch/val_features/
echo "removing 30% HDD features"
rm -rf data/processed/val/features/NISQA_VAL_LIVE
rm -rf data/processed/val/features/NISQA_TEST_P501
rm -rf data/processed/val/features/NISQA_TEST_FOR
rm -rf data/processed/val/features/NISQA_TEST_LIVETALK
rm -rf data/processed/val/features/NISQA_TRAIN_LIVE
rm -rf data/processed/val/features/NISQA_VAL_SIM
rm -rf data/processed/val/features/withReverberationTrainDev
rm -rf data/processed/val/features/withoutReverberationTrainDev
rm -rf data/processed/val/features/NISQA_TRAIN_SIM
rm -rf data/processed/val/features/16k_speech
echo "creating symlinks to SSD"
cd data/processed/val/features/
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/NISQA_VAL_LIVE .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/NISQA_TEST_P501 .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/NISQA_TEST_FOR .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/NISQA_TEST_LIVETALK .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/NISQA_TRAIN_LIVE .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/NISQA_VAL_SIM .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/withReverberationTrainDev .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/withoutReverberationTrainDev .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/NISQA_TRAIN_SIM .
ln -s /home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/scratch/val_features/16k_speech .
echo "finished"
