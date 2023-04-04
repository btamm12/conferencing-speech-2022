import csv
from io import BytesIO
import os
import torch
import zipfile

from src import constants
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.split import Split

def _load_csv(
    split: Split,
    example: bool,
    use_35: bool,
):
    # Load CSV.
    dataset = constants.get_dataset(split, example, use_35)
    xlsr_paths = []
    with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
        csv_reader = csv.reader(in_csv)
        for idx, in_row in enumerate(csv_reader):

            # Skip header row.
            if idx == 0:
                continue

            # Save feature_path, norm_mos
            col_path = STANDARDIZED_CSV_INFO.col_xlsr_path
            xlsr_paths.append(in_row[col_path])

    return xlsr_paths


def zip_next_features(
    next_xlsr_start_layer: int,
    next_xlsr_end_layer: int,
    cache_start_layer: int,
    cache_end_layer: int,
    example: bool,
    use_35: bool,
):
    # Improve throughput by zipping and saving the features that should be loaded
    # into memory into a fast scratch drive.

    # Load CSV.
    xlsr_paths_train = _load_csv(Split.TRAIN, example, use_35)
    xlsr_paths_val = _load_csv(Split.VAL, example, use_35)

    zip_train_path = str(constants.XLSR_NEXT_TRAIN_ZIP_PATH)
    with zipfile.ZipFile(zip_train_path, "w", zipfile.ZIP_STORED) as zipf:
        for xlsr_path in xlsr_paths_train:
            abs_path = str(constants.DIR_PROJECT.joinpath(xlsr_path))

            # Load features.
            xlsr_states = torch.load(abs_path)
            features = tuple(
                x for i, x in enumerate(xlsr_states)
                if cache_start_layer + i >= next_xlsr_start_layer and cache_start_layer + i < next_xlsr_end_layer
            )

            # Create IO stream, so features stay in memory (no writing to and
            # then reading from tmp file on disk).
            feature_io = BytesIO()
            torch.save(features, feature_io)

            feature_io.seek(0)
            zipped_path = xlsr_path
            with zipf.open(zipped_path, "w") as file:
                file.write(feature_io.read())

    zip_val_path = str(constants.XLSR_NEXT_VAL_ZIP_PATH)
    with zipfile.ZipFile(zip_val_path, "w", zipfile.ZIP_STORED) as zipf:
        for xlsr_path in xlsr_paths_val:
            abs_path = str(constants.DIR_PROJECT.joinpath(xlsr_path))

            # Load features.
            xlsr_states = torch.load(abs_path)
            features = tuple(
                x for i, x in enumerate(xlsr_states)
                if cache_start_layer + i >= next_xlsr_start_layer and cache_start_layer + i < next_xlsr_end_layer
            )

            # Create IO stream, so features stay in memory (no writing to and
            # then reading from tmp file on disk).
            feature_io = BytesIO()
            torch.save(features, feature_io)

            feature_io.seek(0)
            zipped_path = xlsr_path
            with zipf.open(zipped_path, "w") as file:
                file.write(feature_io.read())



def read_xlsr_zip(split: Split, example: bool, use_35: bool):

    # Load CSV.
    xlsr_paths = _load_csv(split, example, use_35)

    # Initialize results.
    results = []

    # Read from ZIP and load as Tensor.
    if split == Split.TRAIN or split == Split.TRAIN_SUBSET:
        zip_path = str(constants.XLSR_CUR_TRAIN_ZIP_PATH)
    elif split == Split.VAL or split == Split.VAL_SUBSET:
        zip_path = str(constants.XLSR_CUR_VAL_ZIP_PATH)
    else:
        raise Exception(f"Cannot read ZIP for split: {split}")

    with zipfile.ZipFile(zip_path, "r", zipfile.ZIP_STORED) as zipf:
        for xlsr_path in xlsr_paths:
            x = torch.load(zipf.open(xlsr_path, "r"))
            results.append(x)

    return results


