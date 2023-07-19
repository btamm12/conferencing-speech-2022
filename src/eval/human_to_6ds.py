# PSEUDO CODE TODO:
# take val.csv and create files val.6ds.{subset,iu_bloomington,pstn,tencent,nisqa}.csv,
# remember which rows are which dataset and use these indices to construct .6ds.csv predictions for all models!


import csv
from math import sqrt
from pathlib import Path

from src import constants
from src.constants import DatasetDir
from src.utils_4ds.split import Split
from src.utils_4ds.csv_info import STANDARDIZED_CSV_INFO, STANDARDIZED_CSV_HEADER

def _filter_and_print_rmse(val_rows, indices, out_name):
        filtered_rows = [x for idx, x in enumerate(val_rows) if idx in indices]
        col_std = STANDARDIZED_CSV_INFO.col_mos_std
        col_votes = STANDARDIZED_CSV_INFO.col_mos_num_votes
        # std, votes
        filtered_rows = [(0., 1.) if x[col_votes] == "1" else (float(x[col_std]), float(x[col_votes])) for x in filtered_rows] 
        filtered_rows = [y for y in filtered_rows if y[0] > -0.01] # remove rows with no info (std == -1)
        filtered_rows = [y for y in filtered_rows if y[1] > 2] # only keep 3+ votes

        # std = sqrt(SE/(N-1))
        total_se = 0.
        total_votes = 0.
        for std, nvotes in filtered_rows:
            se = pow(std,2)*(nvotes-1)
            total_se += se
            total_votes += nvotes

        # rmse
        rmse = sqrt(total_se / total_votes)

        print(out_name + ": " + "%0.6f" % rmse)



def human_to_6ds():

    val_ds: DatasetDir = constants.get_dataset(Split.VAL, example=False, use_35=False)

    val_rows = []
    with open(val_ds.csv_path, mode="r") as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue # skip header
            val_rows.append(row)

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
    
    print("RMSE:")


    _filter_and_print_rmse(val_rows, all_indices, "human 6ds_val2val")
    _filter_and_print_rmse(val_rows, iub_indices, "human 6ds_val2iub")
    _filter_and_print_rmse(val_rows, nisqa_indices, "human 6ds_val2nisqa")
    _filter_and_print_rmse(val_rows, pstn_indices, "human 6ds_val2pstn")
    print("human 6ds_val2tencent: N/A (tencent not available)")
    # _filter_and_print_rmse(val_rows, tencent_indices, "human 6ds_val2tencent")
    print("human 6ds_val2subset: N/A (tencent not available)")
    # _filter_and_print_rmse(val_rows, subset_indices, "human 6ds_val2subset")
    _filter_and_print_rmse(val_rows, not_subset_indices, "human 6ds_val2notsubset")


    print("Done")


if __name__ == "__main__":
    human_to_6ds()