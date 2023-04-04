# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.train.extract_features import extract_features
from src.train.train_model_35 import train_model
from src.utils.split import Split

# slurm/submit_array_job.sh train
# slurm/submit_array_job.sh train_example

@click.command()
@click.option('-s', '--stage', default=0)
@click.option('-u', '--use_caching', is_flag=True)
@click.option('-e', '--example', is_flag=True)
@click.option('-c', '--cpus', default=4)
def main(stage, use_caching, example, cpus):
    """Train models."""
    logger = logging.getLogger(__name__)
    logger.info('training model')

    # If use_caching: calculate XLS-R features as preprocessing step and then load
    # calculated features each training iteration.
    # => This theoretically saves (num_epochs-1) calculations, but this isn't exactly
    #    true because the saved features are huge and loading time is considerable.
    # => Normally use_caching==False takes around 30 days to complete a single XLS-R
    #    model size. Multiply this by 3 for 300M, 1B, 2B and we get 90 days training.
    # => For this reason, we only consider 35% of the data and 40 epochs, which is an
    #    equivalent number of training steps as 100% data for 10 epochs (in previous
    #    work, we found most models saturated by epoch 7-10).
    # => So use_caching==False with these reductions gives 90*0.25*(40/50) = 18 days
    #    for all 3 model sizes trained
    # => When use_caching==True, we find that if you are able to cache all features
    #    on disk (~30TB data for all hidden_states of XLS-R 2B) it would take only
    #    8.3 days to train a single XLS-R model size with 100% data and 50 epochs.
    #    Multiplying this by 3 gives 25 training days.
    # => Using the 35% and 40 epochs trick, we reduce this to 25*0.25*(40/50) = 5
    #    days! We also reduce the necessary cache size to 30TB*0.25 = 7.5TB
    # => The message is: if you have 7.5TB space available, you will be able to
    #    complete the ENTIRE 35%-40epoch TRAINING PROCEDURE (this includes feature
    #    extraction preprocessing time) in 5 days. If this is not possible, the
    #    training will take 18 days.
    # 
    # => We experimented with a middle-ground approach which is halfway between
    #    full-caching and no-caching with optimal data reuse. The required cache is
    #    ~2TB for 35% data-40epoch
    #    - calculate XLS-R 300M features over entire dataset
    #    - cache hidden state layers 0:17
    #    - train models associated with layers 0:17
    #    - repeat for layers 17:25
    #    - ... then repeat all this for XLS-R 1B and XLS-R 2B (0:17, 17:34, 34:49)
    # => Using this technique, we were able to get estimates of 33.3 days for one
    #    XLS-R size, so 100 days for all three sizes, using 100% data and 50 epochs.
    # => Using the 35% data and 40 epochs reduction, this becomes 100*0.25*(40/50) =
    #    20 days. THIS IS SLIGHTLY WORSE THAN USING NO CACHE AT ALL! (18 days)
    #    
    # => Note: cache times are calculated with 7200RPM HDD, if an SSD is used, the
    #    training time will be significantly reduced, as the feature loading times
    #    are by far the bottleneck. Since an SSD is around 2x faster with serial
    #    reads (feature files are large, so reads are mostly serial), I will use a
    #    factor 1.7x speedup, accounting for other training overhead.
    #                                     HDD   =>    SSD
    #    - cache size 0.5TB ( 7 layers): 30 days => 17.6 days
    #    - cache size   2TB (17 layers): 20 days => 11.8 days
    #    - cache size 7.5TB (49 layers):  5 days =>  2.9 days
    #
    #    - in retrospect, I think the GPU training will become a bottleneck before we
    #      get to the 5-day training mark! We are using ~33% GPU utilization with the
    #      20-day training. Once this gets to 20*.33 = 6.7 days, we will hit the GPU
    #      wall
    #    ==> say lower limit is 7 days for 1 GPU and 5 days for multi-GPU
    #
    # RETROSPECT #2:
    # => training from raw audio (use_caching==False) has much slower training than
    #    initially anticipated (100-120 days instead of 18 days)
    # => I will do timing on caching mechanism

    # STAGES
    use_35 = False
    if use_caching:
        use_ram = True
        full_and_subset_training = True
        use_subset = False # included in full_and_subset_training
        fixed_args = (example, use_35, use_caching, use_ram, full_and_subset_training, use_subset, cpus)
        # 127 stages
        stages = [
            # 300M: cache 0-9 on disk, of those cache 1 layer in RAM at a time
            ["features", ("wav2vec2-xls-r-300m", 0, 9, Split.TRAIN, example, use_35)],
            ["features", ("wav2vec2-xls-r-300m", 0, 9, Split.VAL, example, use_35)],
            ["train", ("wav2vec2-xls-r-300m", 0, 1, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 1, 2, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 2, 3, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 3, 4, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 4, 5, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 5, 6, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 6, 7, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 7, 8, *fixed_args, 0,9)],
            ["train", ("wav2vec2-xls-r-300m", 8, 9, *fixed_args, 0,9)],
            # 300M: 9-18, 1 layer in RAM
            ["features", ("wav2vec2-xls-r-300m", 9, 18, Split.TRAIN, example, use_35)],
            ["features", ("wav2vec2-xls-r-300m", 9, 18, Split.VAL, example, use_35)],
            ["train", ("wav2vec2-xls-r-300m", 9, 10, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 10, 11, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 11, 12, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 12, 13, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 13, 14, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 14, 15, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 15, 16, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 16, 17, *fixed_args, 9,18)],
            ["train", ("wav2vec2-xls-r-300m", 17, 18, *fixed_args, 9,18)],
            # 300M: 18-25, 1 layer in RAM
            ["features", ("wav2vec2-xls-r-300m", 18, 25, Split.TRAIN, example, use_35)],
            ["features", ("wav2vec2-xls-r-300m", 18, 25, Split.VAL, example, use_35)],
            ["train", ("wav2vec2-xls-r-300m", 19, 20, *fixed_args, 18,25)],
            ["train", ("wav2vec2-xls-r-300m", 20, 21, *fixed_args, 18,25)],
            ["train", ("wav2vec2-xls-r-300m", 21, 22, *fixed_args, 18,25)],
            ["train", ("wav2vec2-xls-r-300m", 22, 23, *fixed_args, 18,25)],
            ["train", ("wav2vec2-xls-r-300m", 23, 24, *fixed_args, 18,25)],
            ["train", ("wav2vec2-xls-r-300m", 24, 25, *fixed_args, 18,25)],
        ]
    else:
        use_ram = False
        stages = [
            # 300M: 0-25
            ["train", ("wav2vec2-xls-r-300m", 0, 25, example, use_35, use_caching, use_ram, False, cpus) ],
            ["train", ("wav2vec2-xls-r-300m", 0, 25, example, use_35, use_caching, use_ram, True, cpus) ], # subset
        ]

    num_stages = len(stages)
    print("==================================")
    print(f"Starting stage {stage} of {num_stages}")

    stage_info = stages[stage]
    stage_fn = stage_info[0]
    stage_args = stage_info[1]
    print(f"> stage_fn={stage_fn}")
    print(f"> args={stage_args}")
    print("==================================")

    if stage_fn == "features":
        extract_features(*stage_args)
    elif stage_fn == "train":
        train_model(*stage_args)
    else:
        raise Exception(f"Unknown stage_fn: {stage_fn}")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
