# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.train_layer_fusion.extract_features import extract_features
from src.utils.split import Split

@click.command()
def main():
    """Extract validation features."""
    logger = logging.getLogger(__name__)
    logger.info('training model')

    xlsr_name = "wav2vec2-xls-r-2b"
    layers = [15,36]
    extract_features(xlsr_name, layers, Split.VAL, ignore_run_once=True)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
