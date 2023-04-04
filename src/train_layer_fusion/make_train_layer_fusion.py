# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.train_layer_fusion.extract_features import extract_features
from src.train_layer_fusion.train_model_layer_fusion import train_model
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
    layers = [15,36]
    layers_str = ",".join(str(x) for x in layers)
    print("==================================")
    print(f"Starting training {xlsr_name} layers {layers_str}")
    print("==================================")
    extract_features(xlsr_name, layers, Split.TRAIN)
    extract_features(xlsr_name, layers, Split.VAL)
    train_model(xlsr_name, layers, cpus)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
