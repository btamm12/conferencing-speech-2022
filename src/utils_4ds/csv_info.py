from dataclasses import dataclass
from pathlib import Path
from src.utils.mos_transform import MosTransform


@dataclass
class CsvInfo:
    """Information about the annotation CSV file.

    This includes:
    - which column correspond to the "audio_path" and "mos" fields
    - if a MOS transformation should be used

    Note that the columns are zero-indexed.
    """
    csv_path: Path
    col_audio_path: int
    col_mos: int
    mos_transform: MosTransform = None
    in_subset: bool = False  # True if this CSV should be included in val_subset (only PSTN/Tencent).
    col_mos_std: int = None
    col_num_votes: int = None
    col_ratings: int = None

# STANDARDIZED FORMAT
@dataclass
class StandardizedCsvInfo:
    """Information about the standardized CSV file.

    Note that the columns are zero-indexed.
    """
    col_audio_path: int = 0  # audio path
    col_xlsr_path: int = 1
    col_mos: int = 2
    col_norm_mos: int = 3
    col_in_subset: int = 4
    col_ds_name: int = 5
    col_mos_std: int = 6
    col_mos_num_votes: int = 7


STANDARDIZED_CSV_INFO = StandardizedCsvInfo()
STANDARDIZED_CSV_HEADER = [
    "audio_path",
    "xlsr_path",
    "mos",
    "norm_mos",
    "in_subset",
    "ds_name",
    "mos_std",
    "mos_num_votes",
]
