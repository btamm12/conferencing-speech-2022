# -*- coding: utf-8 -*-
import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.train_lf_mfcc_4ds.extract_features import extract_features
from src.train_lf_mfcc_4ds.train_model import train_model
from src.utils_4ds.split import Split

# slurm/submit_array_job.sh train_layer_10
# slurm/submit_array_job.sh train_example

@click.command()
@click.option('-i', '--input', default="xlsr")
@click.option('-x', '--xlsr_name', default="wav2vec2-xls-r-2b")
@click.option('-c', '--cpus', default=5)
def main(input, xlsr_name, cpus):
    """Train models."""
    logger = logging.getLogger(__name__)
    logger.info('training model')

    if input == "mfcc":
        xlsr_name = None
        layers = None
        print("==================================")
        print("Starting training MFCC")
        print("==================================")
    else:
        if "300m" in xlsr_name:
            # layers = [7,20]
            layers = [5,21]
        else:
            # layers = [15,36]
            layers = [10,41]
        layers_str = ",".join(str(x) for x in layers)
        print("==================================")
        print(f"Starting training {xlsr_name} layers {layers_str}")
        print("==================================")
    # extract_features(input, xlsr_name, layers, Split.TRAIN)
    # extract_features(input, xlsr_name, layers, Split.VAL)
    train_model(input, xlsr_name, layers, cpus)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
