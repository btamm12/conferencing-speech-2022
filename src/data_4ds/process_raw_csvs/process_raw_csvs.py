import csv
import math
from random import Random
from src import constants
from src.data_4ds.process_raw_csvs.utils.transform_csv import transform_csv
from src.utils_4ds.csv_info import STANDARDIZED_CSV_INFO, STANDARDIZED_CSV_HEADER
from src.utils_4ds.run_once import run_once
from src.utils_4ds.split import Split, ALL_SPLITS, DEV_SPLITS, TRAIN_SUBSET_SPLITS, VAL_SUBSET_SPLITS



def _process_raw_csvs(split: Split):

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, example=False, use_35=False)

    # Select appropriate CSV infos.
    if split in DEV_SPLITS:
        csv_infos = constants.TRAIN_CSVS
    elif split == Split.TEST:
        csv_infos = constants.TEST_CSVS
    else:
        raise Exception(f"Unknown split: {split}")

    # Print split name.
    split_name = str(split).lower().split(".")[1]
    print(f"Processing raw CSVs for split: {split_name}.")

    # Load all CSVs.
    rows = []
    for csv_info in csv_infos:
        csv_path = csv_info.csv_path

        # Always save features in main TRAIN/VAL features dir.
        if "train" in split.name.lower():
            _split = Split.TRAIN
        elif "val" in split.name.lower():
            _split = Split.VAL
        else:
            print("no train/val in split name")
            _split = split
        out_dir = constants.get_dataset(_split, example=False, use_35=False).features_dir

        # print(f"Processing raw CSV: {csv_path}")
        new_rows = transform_csv(
            in_path=csv_path,
            out_dir=out_dir,
            csv_info=csv_info,
        )
        new_rows.pop(0)  # Remove header
        rows.extend(new_rows)

    # Shuffle rows before making split.
    rdm = Random(42)  # Reproducible random number generation.
    rdm.shuffle(rows)

    print(f"> total rows: {len(rows)}")

    # Train/val split.
    val_rows = math.ceil(constants.VAL_SPLIT * len(rows))
    train_rows = len(rows) - val_rows
    if split == Split.TRAIN or split in TRAIN_SUBSET_SPLITS:
        rows = rows[:train_rows]
    if split == Split.VAL or split in VAL_SUBSET_SPLITS:
        rows = rows[train_rows:]

    print(f"> train/val rows: {len(rows)}")


    # Subset?
    if split in [Split.TRAIN_SUBSET, Split.VAL_SUBSET]:
        col_subset = STANDARDIZED_CSV_INFO.col_in_subset
        rows = [row for row in rows if row[col_subset] == "True"]
    if split in [Split.TRAIN_PSTN, Split.VAL_PSTN]:
        col_ds_name = STANDARDIZED_CSV_INFO.col_ds_name
        rows = [row for row in rows if row[col_ds_name] == "pstn"]
    if split in [Split.TRAIN_TENCENT, Split.VAL_TENCENT]:
        col_ds_name = STANDARDIZED_CSV_INFO.col_ds_name
        rows = [row for row in rows if row[col_ds_name] == "tencent"]
    if split in [Split.TRAIN_NISQA, Split.VAL_NISQA]:
        col_ds_name = STANDARDIZED_CSV_INFO.col_ds_name
        rows = [row for row in rows if row[col_ds_name] == "nisqa"]
    if split in [Split.TRAIN_IUB, Split.VAL_IUB]:
        col_ds_name = STANDARDIZED_CSV_INFO.col_ds_name
        rows = [row for row in rows if row[col_ds_name] == "iub"]

    print(f"> subset rows: {len(rows)}")

    # Add header row.
    rows.insert(0, STANDARDIZED_CSV_HEADER)

    # Write to output CSV.
    with open(dataset.csv_path, mode="w", encoding="utf8") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(rows)

    print(f"Finished.")


def process_raw_csvs(split: Split):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    flag_name = f"processed_csv_{split_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _process_raw_csvs(split)
        else:
            print(f"Raw CSVs already processed for {split_name} split.")


if __name__ == "__main__":
    for split in DEV_SPLITS:
        process_raw_csvs(split)
        
