# PSEUDO CODE TODO:
# take val.csv and create files val.6ds.{subset,iu_bloomington,pstn,tencent,nisqa}.csv,
# remember which rows are which dataset and use these indices to construct .6ds.csv predictions for all models!


import csv
from pathlib import Path

from src import constants
from src.constants import DatasetDir
from src.utils_4ds.split import Split
from src.utils_4ds.csv_info import STANDARDIZED_CSV_INFO, STANDARDIZED_CSV_HEADER

def _filter_and_write_rows(predictions_dir: Path, pred_rows, indices, subdir_name, out_name):
        rows = [x for idx, x in enumerate(pred_rows) if idx in indices]
        out_dir = predictions_dir.joinpath(subdir_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir.joinpath(out_name)

        # write outputs
        with open(out_path, mode="w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["mos_pred"])
            for row in rows:
                csv_writer.writerow(row)

def _filter_and_write_val(val_rows, out_path: Path, indices):
        rows = [x for idx, x in enumerate(val_rows) if idx in indices]

        # write outputs
        with open(out_path, mode="w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(STANDARDIZED_CSV_HEADER)
            for row in rows:
                csv_writer.writerow(row)


def val_predictions_to_6ds():

    val_ds: DatasetDir = constants.get_dataset(Split.VAL, example=False, use_35=False)
    script_dir = Path(__file__).parent
    ground_truths_dir = script_dir.joinpath("ground_truths")
    predictions_dir = script_dir.joinpath("predictions")
    predictions_val_dir = script_dir.joinpath("predictions", "val")

    val_rows = []
    # val_mos_only = ["mos"]
    with open(val_ds.csv_path, mode="r") as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue # skip header
            val_rows.append(row)
            # val_mos_only.append(row[STANDARDIZED_CSV_INFO.col_mos])

    # Find indices per DS.
    all_indices = list(range(len(val_rows)))
    iub_indices = []
    nisqa_indices = []
    pstn_indices = []
    tencent_indices = []
    subset_indices = []
    not_subset_indices = []
    for idx, row in enumerate(val_rows):
        _in_subset = row[STANDARDIZED_CSV_INFO.col_in_subset]
        _ds_name = row[STANDARDIZED_CSV_INFO.col_ds_name]
        in_subset = (_in_subset == "True")
        is_iub = (_ds_name == "iub")
        is_nisqa = (_ds_name == "nisqa")
        is_pstn = (_ds_name == "pstn")
        is_tencent = (_ds_name == "tencent")

        if in_subset:
            subset_indices.append(idx)
        else:
            not_subset_indices.append(idx)
        if is_iub:
            iub_indices.append(idx)
        elif is_nisqa:
            nisqa_indices.append(idx)
        elif is_pstn:
            pstn_indices.append(idx)
        elif is_tencent:
            tencent_indices.append(idx)
        else:
            raise Exception(f"unknown dsname {idx} {_ds_name}")
    

    # Filter predictions based on indices.
    prediction_paths = list(predictions_val_dir.glob("*.csv"))
    for pred_path in prediction_paths:
        print(f"Processing: {pred_path}")
        pred_rows = []
        with open(pred_path, mode="r") as f:
            csv_reader = csv.reader(f)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue # skip header
                pred_rows.append(row)
        
        _filter_and_write_rows(predictions_dir, pred_rows, all_indices, "6ds_val2val", pred_path.name)
        _filter_and_write_rows(predictions_dir, pred_rows, iub_indices, "6ds_val2iub", pred_path.name)
        _filter_and_write_rows(predictions_dir, pred_rows, nisqa_indices, "6ds_val2nisqa", pred_path.name)
        _filter_and_write_rows(predictions_dir, pred_rows, pstn_indices, "6ds_val2pstn", pred_path.name)
        _filter_and_write_rows(predictions_dir, pred_rows, tencent_indices, "6ds_val2tencent", pred_path.name)
        _filter_and_write_rows(predictions_dir, pred_rows, subset_indices, "6ds_val2subset", pred_path.name)
        _filter_and_write_rows(predictions_dir, pred_rows, not_subset_indices, "6ds_val2notsubset", pred_path.name)

    # Also copy vals.
    print("Writing vals")
    _filter_and_write_val(val_rows, ground_truths_dir.joinpath("6ds_val2val.csv"), all_indices)
    _filter_and_write_val(val_rows, ground_truths_dir.joinpath("6ds_val2iub.csv"), iub_indices)
    _filter_and_write_val(val_rows, ground_truths_dir.joinpath("6ds_val2nisqa.csv"), nisqa_indices)
    _filter_and_write_val(val_rows, ground_truths_dir.joinpath("6ds_val2pstn.csv"), pstn_indices)
    _filter_and_write_val(val_rows, ground_truths_dir.joinpath("6ds_val2tencent.csv"), tencent_indices)
    _filter_and_write_val(val_rows, ground_truths_dir.joinpath("6ds_val2subset.csv"), subset_indices)
    _filter_and_write_val(val_rows, ground_truths_dir.joinpath("6ds_val2notsubset.csv"), not_subset_indices)

    print("Done")


if __name__ == "__main__":
    val_predictions_to_6ds()