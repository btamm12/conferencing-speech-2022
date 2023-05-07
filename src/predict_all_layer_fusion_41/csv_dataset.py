import csv
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.functional import pad
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC
from transformers import Wav2Vec2FeatureExtractor
from typing import List, Tuple
import warnings


from src import constants
from src.model_lf_mfcc_4ds.config import FEAT_SEQ_LEN
from src.utils.split import Split
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.full_path import full_path


class MyCrop(torch.nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x):
        # Center crop.

        unsqueezed = False
        if x.dim() == 1:
            unsqueezed = True
            x = x.unsqueeze(1)
        assert x.dim() == 2

        if x.size(0) > self.seq_len:
            center_start_idx = int(x.size(0) / 2 - self.seq_len / 2)
            start_idx = center_start_idx
            end_idx = start_idx + self.seq_len
            x = x[start_idx:end_idx, :]
        if x.size(0) < self.seq_len:
            to_pad = self.seq_len - x.size(0)
            xT = x.transpose(0, 1)
            xT = pad(xT, (0, to_pad), mode="constant", value=0.0)
            x = xT.transpose(0, 1)

        if unsqueezed:
            x = x.squeeze(1)

        return x


class CsvDataset(Dataset):
    def __init__(
        self,
        split: Split,
        skip_first: int = 0,
    ) -> None:
        super().__init__()

        self.split = split
        self.skip_first = skip_first

        # For printing...
        split_name = str(split).lower().split(".")[1]
        print(f"Creating dataloader for {split_name} set.")

        # Select dataset.
        dataset = constants.get_dataset(split, example=False, use_35=False)

        # Type to CSV column.
        col_path = STANDARDIZED_CSV_INFO.col_xlsr_path
        col_mos = STANDARDIZED_CSV_INFO.col_norm_mos

        # Load CSV.
        self.csv_data = []  # feature_path, norm_mos
        with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):

                # Skip header row.
                if idx == 0:
                    continue

                # Save feature_path, norm_mos
                file_path: str = in_row[col_path]
                norm_mos = torch.tensor(float(in_row[col_mos]))
                self.csv_data.append([file_path, norm_mos])


        self.my_crop = MyCrop(FEAT_SEQ_LEN)

    def __len__(self):
        return len(self.csv_data)

    def _getitem_impl(self, index: int):
        # print("get_item %0.6i start... " % index, end="")

        if index < self.skip_first:
            return (None, None, None, None)

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[index][0]
        mfcc_path = full_path(file_path % "mfcc")
        xlsr_300m_path = full_path(file_path % "wav2vec2-xls-r-300m")
        xlsr_1b_path = full_path(file_path % "wav2vec2-xls-r-1b")
        xlsr_2b_path = full_path(file_path % "wav2vec2-xls-r-2b")

        pt_mfcc = torch.load(mfcc_path)
        pt_300m = torch.load(xlsr_300m_path)
        pt_1b = torch.load(xlsr_1b_path)
        pt_2b = torch.load(xlsr_2b_path)

        pt_mfcc = [self.my_crop(x).unsqueeze(0) for x in pt_mfcc]
        pt_300m = [self.my_crop(x.squeeze(0)).unsqueeze(0) for x in pt_300m]
        pt_1b = [self.my_crop(x.squeeze(0)).unsqueeze(0) for x in pt_1b]
        pt_2b = [self.my_crop(x.squeeze(0)).unsqueeze(0) for x in pt_2b]

        return (pt_mfcc, pt_300m, pt_1b, pt_2b)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self._getitem_impl(index)
