import ast
import csv
import numpy as np
import os
from pathlib import Path

from src import constants
from src.utils.csv_info import STANDARDIZED_CSV_HEADER, CsvInfo
from src.utils.mos_transform import MosTransform


def transform_csv(in_path: Path, out_dir: Path, csv_info: CsvInfo):
    """Transform the given CSV file to the standardized format found in
    src/utils/csv_info.py.

    Args:
        in_path (Path): Path to input CSV file.
        out_dir (Path): Where to the processed CSV will be stored.
        csv_info (CsvInfo): CSV metadata.
    """

    # Add %s for feature type.
    out_dir = out_dir.joinpath("%s")

    # Make sure input path exists.
    in_dir = in_path.parent
    if not in_path.exists():
        raise Exception("Path does not exist: %s" % str(in_path))

    # Calculate relative path from DIR_PROJECT to in_dir.
    in_dir_rel_path = os.path.relpath(in_dir, constants.DIR_PROJECT)
    out_dir_rel_path = os.path.relpath(out_dir, constants.DIR_PROJECT)

    # Object for MOS normalization from [1,5] to [0,1].
    mos_normalizer = MosTransform(
        in_mos_min=1,
        in_mos_max=5,
        out_mos_min=0,
        out_mos_max=1,
    )

    # Open file.
    with open(in_path, encoding="utf8", mode="r") as in_csv:

        # Create CSV reader/writer.
        csv_reader = csv.reader(in_csv)

        # Output rows.
        out_rows = []
        for idx, in_row in enumerate(csv_reader):

            # Write header row.
            if idx == 0:
                # Write to output file.
                out_rows.append(STANDARDIZED_CSV_HEADER)
                continue

            # Skip empty row.
            if len(in_row) == 0:
                continue

            # Process row...
            out_row = []

            # 1. Feature paths.

            # 1.1. Audio path:
            #      These files will be kept in the "raw" directory (in_dir).
            audio_path = in_row[csv_info.col_audio_path]
            audio_base, audio_ext = os.path.splitext(audio_path)
            if audio_ext != ".wav":
                msg = "Expected .wav file but got %s file!" % audio_ext
                raise Exception(msg)
            audio_path = os.path.join(in_dir_rel_path, audio_path)
            audio_path = os.path.relpath(audio_path)  # resolve path
            out_row.append(audio_path)

            # 1.2. XLS-R feature paths:
            #      These files will eventually be saved in the "processed"
            #      directory (out_dir).
            feat_path = audio_base + ".pt"
            feat_path = os.path.join(out_dir_rel_path, feat_path)
            feat_path = os.path.relpath(feat_path)  # resolve path
            out_row.append(feat_path)

            # 2. Labels.

            # 2.1. MOS
            mos = in_row[csv_info.col_mos]
            if csv_info.mos_transform is not None:
                mos = csv_info.mos_transform.transform_str(mos)
            out_row.append(mos)

            # 2.2. Normalized MOS.
            norm_mos = mos_normalizer.transform_str(mos)
            out_row.append(norm_mos)

            # 3. In PSTN/Tencent subset?
            out_row.append(str(csv_info.in_subset))

            # 4. Dataset name?
            out_row.append(str(csv_info.ds_name))

            # 5. MOS std / 6. MOS num votes?
            mos_std = None
            mos_num_votes = None
            if csv_info.col_ratings is not None:
                mos_ratings = ast.literal_eval(in_row[csv_info.col_ratings])
                # Apply mos transform.
                if csv_info.mos_transform is not None:
                    mos_ratings = [csv_info.mos_transform.transform(x) for x in mos_ratings]
                mos_std = "%0.6f" % np.std(mos_ratings, ddof=1) # divide by (N-1)
                mos_num_votes = str(len(mos_ratings))
            else:
                if csv_info.col_mos_std is not None:
                    mos_std = in_row[csv_info.col_mos_std]
                if csv_info.col_num_votes is not None:
                    mos_num_votes = in_row[csv_info.col_num_votes]
            both_none = mos_std is None and mos_num_votes is None
            both_values = mos_std is not None and mos_num_votes is not None
            assert both_none or both_values
            if both_none:
                mos_std = "-1"
                mos_num_votes = "0"
            out_row.append(mos_std)
            out_row.append(mos_num_votes)

            # 7. ACRs if available
            if csv_info.col_ratings is not None:
                acrs = str([round(x, 3) for x in mos_ratings])
            else:
                acrs = "NA"
            out_row.append(acrs)

            # Append to output rows.
            out_rows.append(out_row)

    return out_rows
