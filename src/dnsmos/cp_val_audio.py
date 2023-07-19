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
val_ds = constants.get_dataset(Split.VAL, example=False, use_35=False)

with open(val_ds.csv_path, mode="r") as f:
    csv_reader = csv.reader(f)
    for idx, row in tqdm(enumerate(csv_reader)):
        if idx == 0: # Skip header row.
            continue
        col_audio_path = STANDARDIZED_CSV_INFO.col_audio_path
        audio_path = full_path(row[col_audio_path])
        dst_path = audio_path.replace("data/raw/", "data/raw_val/")
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(audio_path, dst_path)
        
