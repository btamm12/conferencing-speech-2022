# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.train_layer_10.extract_features import extract_features
from src.train_layer_10.train_model_layer_10 import train_model
from src.utils.split import Split

# slurm/submit_array_job.sh train_layer_10
# slurm/submit_array_job.sh train_example

@click.command()
@click.option('-c', '--cpus', default=5)
def main(cpus):
    """Train models."""
    logger = logging.getLogger(__name__)
    logger.info('training model')

    xlsr_name = "wav2vec2-xls-r-2b"
    layer = 10
    print("==================================")
    print(f"Starting training {xlsr_name} layer {layer}")
    print("==================================")
    extract_features(xlsr_name, layer, Split.TRAIN)
    extract_features(xlsr_name, layer, Split.VAL)
    train_model(xlsr_name, layer, cpus)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
