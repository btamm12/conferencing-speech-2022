# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.data.calculate_norm.calculate_norm import calculate_norm
from src.utils.split import Split


@click.command()
@click.option('-e', '--example', is_flag=True)
@click.option('-i', '--partition_idx', default=0)
@click.option('-n', '--num_partitions', default=1)
def main(example, partition_idx, num_partitions):
    """Calculate norm + variance of each channel over all inputs in a dataset."""
    logger = logging.getLogger(__name__)
    logger.info('calculating norm')

    ALL_SPLITS = [
        Split.TRAIN,
        Split.TRAIN_SUBSET,
        Split.VAL,
        Split.VAL_SUBSET,
        Split.TEST,
    ]
    N = len(ALL_SPLITS)

    start_idx = int(partition_idx*N/num_partitions)
    end_idx = int((partition_idx+1)*N/num_partitions)
    splits = ALL_SPLITS[start_idx:end_idx]


    # Calculate norm.
    for split in splits:
        calculate_norm(split, example=True)
        if not example:
            calculate_norm(split, example=False)

    # calculate_norm(Split.TRAIN, example=True)
    # calculate_norm(Split.TRAIN_SUBSET, example=True)
    # calculate_norm(Split.VAL, example=True)
    # calculate_norm(Split.VAL_SUBSET, example=True)
    # calculate_norm(Split.TRAINVAL, example=True)
    # calculate_norm(Split.TEST, example=True)
    # if not example:
    #     calculate_norm(Split.TRAIN, example=False)
    #     calculate_norm(Split.TRAIN_SUBSET, example=False)
    #     calculate_norm(Split.VAL, example=False)
    #     calculate_norm(Split.VAL_SUBSET, example=False)
    #     calculate_norm(Split.TRAINVAL, example=False)
    #     calculate_norm(Split.TEST, example=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
