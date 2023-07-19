# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.train_lf_all_layers_4ds.train_model import train_model


@click.command()
@click.option('-x', '--xlsr_name', default="wav2vec2-xls-r-2b")
@click.option('-c', '--cpus', default=1)
def main(input, xlsr_name, cpus):
    """Train models."""
    logger = logging.getLogger(__name__)
    logger.info('training model')

    print("==================================")
    print(f"Starting training {xlsr_name} all layers")
    print("==================================")
    train_model(xlsr_name, cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
