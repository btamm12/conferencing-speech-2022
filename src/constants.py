from pathlib import Path

from src.utils.csv_info import CsvInfo
from src.utils.mos_transform import MosTransform
from src.utils.split import Split
from src.utils_4ds.split import Split as Split2


# ===== #
# PATHS #
# ===== #

# Locate the root directory of the project, i.e.
# conferencing-speech-2022/
DIR_PROJECT = None
for path in Path(__file__).parents:
    if path.name == "conferencing-speech-2022":
        DIR_PROJECT = path
        break
if DIR_PROJECT is None:
    raise Exception("Unable to locate root dir.")
    
# DIR_PROJECT = Path(__file__).resolve().parents[1]

# conferencing-speech-2022/data/
DIR_DATA = DIR_PROJECT.joinpath("data")
DIR_DATA_FLAGS = DIR_DATA.joinpath("flags")
DIR_DATA_PROCESSED = DIR_DATA.joinpath("processed")
DIR_DATA_RAW = DIR_DATA.joinpath("raw")

# conferencing-speech-2022/logs/
DIR_LOGS = DIR_PROJECT.joinpath("logs")
DIR_LOGS.mkdir(parents=True, exist_ok=True)

# conferencing-speech-2022/models/
MODELS_DIR = DIR_PROJECT.joinpath("models")
XLSR_NAMES = [
    "wav2vec2-xls-r-300m",
    "wav2vec2-xls-r-1b",
    "wav2vec2-xls-r-2b",
]
XLSR_DIRS = {
    xlsr_name: MODELS_DIR.joinpath(xlsr_name)
    for xlsr_name in XLSR_NAMES
}
XLSR_NAME = "wav2vec2-xls-r-300m"
XLSR_DIR = MODELS_DIR.joinpath(XLSR_NAME)

# conferencing-speech-2022/notebooks/
NOTEBOOKS_DIR = DIR_PROJECT.joinpath("notebooks")

# conferencing-speech-2022/scratch
DIR_SCRATCH = DIR_PROJECT.joinpath("scratch")
DIR_SCRATCH.mkdir(parents=True, exist_ok=True)
XLSR_NEXT_TRAIN_ZIP_PATH = DIR_SCRATCH.joinpath("next_train_features.zip")
XLSR_NEXT_VAL_ZIP_PATH = DIR_SCRATCH.joinpath("next_val_features.zip")
XLSR_CUR_TRAIN_ZIP_PATH = DIR_SCRATCH.joinpath("cur_train_features.zip")
XLSR_CUR_VAL_ZIP_PATH = DIR_SCRATCH.joinpath("cur_val_features.zip")
XLSR_ZIPS_READY_FLAG = DIR_SCRATCH.joinpath("zips_ready.txt")

# conferencing-speech-2022/src/
SRC_DIR = DIR_PROJECT.joinpath("src")


# =============================== #
# DOWNLOAD INFORMATION & RAW DIRS #
# =============================== #

# Google Drive API credentials (see main README for details).
GDRIVE_CRED_PATH = DIR_PROJECT.joinpath("gdrive_cred.json")

# IU_BLOOMINGTON (uses Google Drive folder ID).
# Columns:
#  - AudioName, Ratings, MOS, AudioType, ScaledMOS
IU_BLOOMINGTON_TRAIN_DIR = DIR_DATA_RAW.joinpath("iu_bloomington", "train")
IU_BLOOMINGTON_TRAIN_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
IU_BLOOMINGTON_TRAIN_ID = "1wIgOqnKA1U-wZQrU8eb67yQyRVOK3SnZ"
IU_BLOOMINGTON_TRAIN_CSVS = [
    CsvInfo(
        csv_path=IU_BLOOMINGTON_TRAIN_DIR.joinpath(x),
        col_audio_path=0,  # AudioName
        col_mos=2,  # MOS
        mos_transform=MosTransform(
            in_mos_min=0,
            in_mos_max=100,
            out_mos_min=1,
            out_mos_max=5,
        ),
        in_subset=False, # Not in PSTN/Tencent subset
        ds_name="iub",
        col_ratings=1,
    ) for x in (
        "audio_scaled_mos_cosine.csv",
        "audio_scaled_mos_voices.csv",
    )
]


# NISQA (uses URL of ZIP file).
# Columns:
# - db, con, file, con_description, filename_deg, filename_ref, source, lang,
# ... votes, mos, noi, col, dis, loud, noi_std, col_std, dis_std, loud_std,
# ... mos_std, filepath_deg, filepath_ref
NISQA_TRAIN_DIR = DIR_DATA_RAW.joinpath("nisqa", "train")
NISQA_TRAIN_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
NISQA_TRAIN_URL = "https://zenodo.org/record/4728081/files/NISQA_Corpus.zip"
NISQA_TRAIN_ZIP_FOLDER = "NISQA_Corpus"  # Extracting ZIP gives this folder.
NISQA_TRAIN_CSVS = [
    CsvInfo(
        csv_path=NISQA_TRAIN_DIR.joinpath(x),
        col_audio_path=19,  # filepath_deg
        col_mos=9,  # mos
        in_subset=False, # Not in PSTN/Tencent subset
        ds_name="nisqa",
        col_mos_std=18,
        col_num_votes=8,
    ) for x in (
        "NISQA_corpus_file.csv",
    )
]

# PSTN (uses URL of ZIP file).
# Columns:
# - filename, MOS, std, 95%CI, votes
PSTN_TRAIN_DIR = DIR_DATA_RAW.joinpath("pstn", "train")
PSTN_TRAIN_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
PSTN_TRAIN_URL = "https://challenge.blob.core.windows.net/pstn/train.zip"
PSTN_TRAIN_ZIP_FOLDER = "pstn_train"  # Extracting ZIP gives this folder.
PSTN_TRAIN_CSVS = [
    CsvInfo(
        csv_path=PSTN_TRAIN_DIR.joinpath(x),
        col_audio_path=0,  # filename
        col_mos=1,  # mos
        in_subset=True, # In PSTN/Tencent subset
        ds_name="pstn",
        col_mos_std=2,
        col_num_votes=4,
    ) for x in (
        "pstn_train.csv",
    )
]

# Received this link via email!
PSTN_TEST_DIR = DIR_DATA_RAW.joinpath("pstn", "test")
PSTN_TEST_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
PSTN_TEST_URL = "https://challenge.blob.core.windows.net/pstn/test.zip"
PSTN_TEST_ZIP_FOLDER = "test"  # Extracting ZIP gives this folder.
PSTN_TEST_CSVS = [
    # Note: this CSV is not raw, it will be extracted from TEST_RAW_CSV.
    CsvInfo(
        csv_path=PSTN_TEST_DIR.joinpath(x),
        col_audio_path=0,  # deg_wav
        col_mos=1,  # mos
        in_subset=True, # In PSTN/Tencent subset
        ds_name="pstn",
    ) for x in (
        "test_data_pstn.csv",
    )
]

# TENCENT (uses URL of ZIP file).
# Columns:
# - deg_wav, mos
TENCENT_TRAIN_DIR = DIR_DATA_RAW.joinpath("tencent", "train")
TENCENT_TRAIN_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
TENCENT_TRAIN_URL = "https://www.dropbox.com/s/ocmn78uh2lu5iwg/TencentCorups.zip?dl=1"
TENCENT_TRAIN_ZIP_FOLDER = "TencentCorups"
TENCENT_TRAIN_CSVS = [
    CsvInfo(
        csv_path=TENCENT_TRAIN_DIR.joinpath(x),
        col_audio_path=0,  # deg_wav
        col_mos=1,  # mos
        in_subset=True, # In PSTN/Tencent subset
        ds_name="tencent",
        col_mos_std=None,
        col_num_votes=None,
    ) for x in (
        "withReverberationTrainDevMOS.csv",
        "withoutReverberationTrainDevMOS.csv",
    )
]

# Received this link via email!
TENCENT_TEST_DIR = DIR_DATA_RAW.joinpath("tencent", "test")
TENCENT_TEST_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
TENCENT_TEST_URL = "https://www.dropbox.com/s/xghmu9vx1shnev9/TencentCorupsVal.zip?dl=1"
TENCENT_TEST_ZIP_FOLDER = "TencentCorupsVal"
TENCENT_TEST_CSVS = [
    # Note: this CSV is not raw, it will be extracted from TEST_RAW_CSV.
    CsvInfo(
        csv_path=TENCENT_TEST_DIR.joinpath(x),
        col_audio_path=0,  # deg_wav
        col_mos=1,  # mos
        in_subset=True, # In PSTN/Tencent subset
        ds_name="tencent",
    ) for x in (
        "test_data_tencent.csv",
    )
]

# Received this link via email!
TUB_TEST_DIR = DIR_DATA_RAW.joinpath("tub", "test")
TUB_TEST_DIR.mkdir(mode=0o755, parents=True, exist_ok=True)
TUB_TEST_URL = "https://nisqa-challenge-db.s3.amazonaws.com/TUB_IS22_DB1.zip"
TUB_TEST_ZIP_FOLDER = "TUB_IS22_DB1"
TUB_TEST_CSVS = [
    # Note: this CSV is not raw, it will be extracted from TEST_RAW_CSV.
    CsvInfo(
        csv_path=TUB_TEST_DIR.joinpath(x),
        col_audio_path=0,  # deg_wav
        col_mos=1,  # mos
        in_subset=True, # In PSTN/Tencent subset
        ds_name="tub",
    ) for x in (
        "test_data_tub.csv",
    )
]

# Downloaded test CSV via email!
TEST_RAW_CSV_PATH = DIR_PROJECT.joinpath("test_data.csv")
TEST_CSV_PATH_FOR_EVAL = DIR_PROJECT.joinpath("test_data_for_eval.csv")

# For ease-of-access, concatenate the train and test csvs.
TRAIN_CSVS = sum([
    IU_BLOOMINGTON_TRAIN_CSVS,
    NISQA_TRAIN_CSVS,
    PSTN_TRAIN_CSVS,
    TENCENT_TRAIN_CSVS,
], [])
TEST_CSVS = sum([
    PSTN_TEST_CSVS,
    TENCENT_TEST_CSVS,
    TUB_TEST_CSVS,
], [])


# ============== #
# PROCESSED DIRS #
# ============== #


class DatasetDir():
    """Structure of each dataset split directory."""
    root_dir: Path

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.csv_path = root_dir.joinpath("data.csv")
        self.features_dir = root_dir.joinpath("features")
        self.predictions_dir = root_dir.joinpath("predictions")
        self.create_dirs()

    def create_dirs(self):
        self.predictions_dir.mkdir(mode=0o755, parents=True, exist_ok=True)


# TRAIN-VAL SPLIT
VAL_SPLIT = 0.15

# Full datasets.
DATASET_TRAIN = DatasetDir(DIR_DATA_PROCESSED.joinpath("train"))
DATASET_TRAIN_SUBSET = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_subset"))
DATASET_TRAIN_PSTN = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_pstn"))
DATASET_TRAIN_TENCENT = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_tencent"))
DATASET_TRAIN_NISQA = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_nisqa"))
DATASET_TRAIN_IUB = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_iub"))
DATASET_VAL = DatasetDir(DIR_DATA_PROCESSED.joinpath("val"))
DATASET_VAL_SUBSET = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_subset"))
DATASET_VAL_PSTN = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_pstn"))
DATASET_VAL_TENCENT = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_tencent"))
DATASET_VAL_NISQA = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_nisqa"))
DATASET_VAL_IUB = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_iub"))
DATASET_TEST = DatasetDir(DIR_DATA_PROCESSED.joinpath("test"))

# 35% datasets for large layer-wise analysis.
DATASET_TRAIN_35 = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_35"))
DATASET_TRAIN_35_SUBSET = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_35_subset"))
DATASET_VAL_35 = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_35"))
DATASET_VAL_35_SUBSET = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_35_subset"))

# Example datasets.
DATASET_TRAIN_EG = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_eg"))
DATASET_TRAIN_EG_SUBSET = DatasetDir(DIR_DATA_PROCESSED.joinpath("train_eg_subset"))
DATASET_VAL_EG = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_eg"))
DATASET_VAL_EG_SUBSET = DatasetDir(DIR_DATA_PROCESSED.joinpath("val_eg_subset"))
DATASET_TEST_EG = DatasetDir(DIR_DATA_PROCESSED.joinpath("test_eg"))


def get_dataset(split: Split, example: bool, use_35: bool):
    if split == Split.TRAIN:
        if example:
            return DATASET_TRAIN_EG
        else:
            return DATASET_TRAIN_35 if use_35 else DATASET_TRAIN
    if split == Split.TRAIN_SUBSET:
        if example:
            return DATASET_TRAIN_EG_SUBSET
        else:
            return DATASET_TRAIN_35_SUBSET if use_35 else DATASET_TRAIN_SUBSET
    if split == Split.VAL:
        if example:
            return DATASET_VAL_EG
        else:
            return DATASET_VAL_35 if use_35 else DATASET_VAL
    if split == Split.VAL_SUBSET:
        if example:
            return DATASET_VAL_EG_SUBSET
        else:
            return DATASET_VAL_35_SUBSET if use_35 else DATASET_VAL_SUBSET
    if split == Split.TEST:
        return DATASET_TEST_EG if example else DATASET_TEST
    
    if split == Split2.TRAIN:
        return DATASET_TRAIN
    if split == Split2.TRAIN_SUBSET:
        return DATASET_TRAIN_SUBSET
    if split == Split2.TRAIN_PSTN:
        return DATASET_TRAIN_PSTN
    if split == Split2.TRAIN_TENCENT:
        return DATASET_TRAIN_TENCENT
    if split == Split2.TRAIN_NISQA:
        return DATASET_TRAIN_NISQA
    if split == Split2.TRAIN_IUB:
        return DATASET_TRAIN_IUB
    
    if split == Split2.VAL:
        return DATASET_VAL
    if split == Split2.VAL_SUBSET:
        return DATASET_VAL_SUBSET
    if split == Split2.VAL_PSTN:
        return DATASET_VAL_PSTN
    if split == Split2.VAL_TENCENT:
        return DATASET_VAL_TENCENT
    if split == Split2.VAL_NISQA:
        return DATASET_VAL_NISQA
    if split == Split2.VAL_IUB:
        return DATASET_VAL_IUB
