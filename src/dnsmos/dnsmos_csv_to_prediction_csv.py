import csv
from pathlib import Path
import shutil
from tqdm.auto import tqdm
from src import constants
from src.utils.full_path import full_path
from src.utils.split import Split
from src.utils.csv_info import STANDARDIZED_CSV_INFO

src_dir = constants.DIR_DATA_RAW
dst_dir = constants.DIR_DATA / "raw_val"
dst_dir.mkdir(parents=True, exist_ok=True)

print("reading csvs")

val_ds = constants.get_dataset(Split.VAL, example=False, use_35=False)
val_csv_rows = []
with open(val_ds.csv_path, mode="r") as f:
    csv_reader = csv.reader(f)
    for idx, row in enumerate(csv_reader):
        if idx == 0: # Skip header row.
            continue
        col_audio_path = STANDARDIZED_CSV_INFO.col_audio_path
        audio_path = full_path(row[col_audio_path])
        dst_path = audio_path.replace("data/raw/", "data/raw_val/")
        val_csv_rows.append(dst_path)

val_subset_ds = constants.get_dataset(Split.VAL_SUBSET, example=False, use_35=False)
val_subset_csv_rows = []
with open(val_subset_ds.csv_path, mode="r") as f:
    csv_reader = csv.reader(f)
    for idx, row in enumerate(csv_reader):
        if idx == 0: # Skip header row.
            continue
        col_audio_path = STANDARDIZED_CSV_INFO.col_audio_path
        audio_path = full_path(row[col_audio_path])
        dst_path = audio_path.replace("data/raw/", "data/raw_val/")
        val_subset_csv_rows.append(dst_path)

dnsmos_rows = []
with open("dnsmos_all_predictions.csv", mode="r") as f:
    csv_reader = csv.reader(f)
    for idx, row in enumerate(csv_reader):
        if idx == 0: # Skip header row.
            continue
        col_audio_path = 1
        col_mos = -1
        audio_path = row[col_audio_path]
        mos = row[col_mos]
        dnsmos_rows.append([audio_path, mos])

pdnsmos_rows = []
with open("dnsmos_all_predictions_personalized.csv", mode="r") as f:
    csv_reader = csv.reader(f)
    for idx, row in enumerate(csv_reader):
        if idx == 0: # Skip header row.
            continue
        col_audio_path = 1
        col_mos = -1
        audio_path = row[col_audio_path]
        mos = row[col_mos]
        pdnsmos_rows.append([audio_path, mos])

print("matching csvs")

# output
out_val_dnsmos_rows = ["prediction"]
out_val_pdnsmos_rows = ["prediction"]
out_val_subset_dnsmos_rows = ["prediction"]
out_val_subset_pdnsmos_rows = ["prediction"]
for idx, val_path in tqdm(enumerate(val_csv_rows)):
    # dnsmos
    found = False
    found_mos = None
    for dnspath, mos in dnsmos_rows:
        if dnspath == val_path:
            found = True
            found_mos = mos
            break
    if not found:
        raise Exception(f"Not found val idx {idx}")
    out_val_dnsmos_rows.append(found_mos)

    # pdnsmos
    found = False
    found_mos = None
    for dnspath, mos in pdnsmos_rows:
        if dnspath == val_path:
            found = True
            found_mos = mos
            break
    if not found:
        raise Exception(f"Not found val idx {idx}")
    out_val_pdnsmos_rows.append(found_mos)

# subset
for idx, val_path in tqdm(enumerate(val_subset_csv_rows)):
    # dnsmos
    found = False
    found_mos = None
    for dnspath, mos in dnsmos_rows:
        if dnspath == val_path:
            found = True
            found_mos = mos
            break
    if not found:
        raise Exception(f"Not found val idx {idx}")
    out_val_subset_dnsmos_rows.append(found_mos)

    # pdnsmos
    found = False
    found_mos = None
    for dnspath, mos in pdnsmos_rows:
        if dnspath == val_path:
            found = True
            found_mos = mos
            break
    if not found:
        raise Exception(f"Not found val idx {idx}")
    out_val_subset_pdnsmos_rows.append(found_mos)


with open("out_val_dnsmos.csv", mode="w") as f:
    for x in out_val_dnsmos_rows:
        f.write(x + "\n")
with open("out_val_pdnsmos.csv", mode="w") as f:
    for x in out_val_pdnsmos_rows:
        f.write(x + "\n")
with open("out_val_subset_dnsmos.csv", mode="w") as f:
    for x in out_val_subset_dnsmos_rows:
        f.write(x + "\n")
with open("out_val_subset_pdnsmos.csv", mode="w") as f:
    for x in out_val_subset_pdnsmos_rows:
        f.write(x + "\n")
