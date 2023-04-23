# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.model.config import ALL_CONFIGS
from src.predict_layer_fusion_41_corrupted.predict_model import predict_model
from src.utils.split import Split


@click.command()
@click.option('-c', '--cpus', default=7)
def main(cpus):
    """Make model predictions on validation splits."""
    logger = logging.getLogger(__name__)
    logger.info('predicting model')

    predict_model(Split.VAL, cpus)
    predict_model(Split.VAL_SUBSET, cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
