import csv
import random
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset
from typing import Tuple

from src.model import config
from src import constants
from src.utils.split import Split
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.full_path import full_path


class MyCrop(torch.nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len

    def forward(self, x):
        # Random crop.

        unsqueezed = False
        if x.dim() == 1:
            unsqueezed = True
            x = x.unsqueeze(1)
        assert x.dim() == 2

        if x.size(0) > self.seq_len:
            min_start_idx = 0
            max_start_idx = x.size(0) - self.seq_len
            start_idx = random.randint(min_start_idx, max_start_idx)
            end_idx = start_idx + self.seq_len
            x = x[start_idx:end_idx, :]
        if x.size(0) < self.seq_len:
            to_pad = self.seq_len - x.size(0)
            xT = x.transpose(0,1)
            xT = pad(xT, (0, to_pad), mode="constant", value=0.0)
            x = xT.transpose(0,1)

        if unsqueezed:
            x = x.squeeze(1)

        return x



class CsvDataset(Dataset):

    def __init__(
        self,
        split: Split,
        include_subset: bool = True,
    ) -> None:
        super().__init__()

        self.split = split
        self.include_subset = include_subset

        # For printing...
        split_name = str(split).lower().split(".")[1]
        print(f"Creating dataloader for {split_name} set.")

        # Select train, val or test dataset.
        if self.include_subset:
            if split == Split.TRAIN_SUBSET or split == Split.VAL_SUBSET or Split == Split.TEST:
                msg = "if include_subset is True, then only Split.TRAIN and Split.VAL are accepted"
                raise Exception(msg)
            if split == Split.TRAIN:
                split_subset = Split.TRAIN_SUBSET
            if split == Split.VAL:
                split_subset = Split.VAL_SUBSET
            dataset = constants.get_dataset(split, example=False, use_35=False)
            dataset_subset = constants.get_dataset(split_subset, example=False, use_35=False)

        # Type to CSV column.
        col_path = STANDARDIZED_CSV_INFO.col_xlsr_path

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
                norm_mos = torch.tensor(
                    float(in_row[STANDARDIZED_CSV_INFO.col_norm_mos]))
                self.csv_data.append([file_path, norm_mos])

        if self.include_subset:
            self.csv_data_subset = []  # feature_path, norm_mos
            with open(dataset_subset.csv_path, encoding="utf8", mode="r") as in_csv:
                csv_reader = csv.reader(in_csv)
                for idx, in_row in enumerate(csv_reader):

                    # Skip header row.
                    if idx == 0:
                        continue

                    # Save feature_path, norm_mos
                    file_path: str = in_row[col_path]
                    norm_mos = torch.tensor(
                        float(in_row[STANDARDIZED_CSV_INFO.col_norm_mos]))
                    self.csv_data_subset.append([file_path, norm_mos])
                
            # Find mapping between indices of subset and full csv. This is simple
            # since the order is the same, but the subset excludes samples of the
            # full csv.
            idx_full = 0
            idx_subset = 0
            N_full = len(self.csv_data)
            N_subset = len(self.csv_data_subset)
            self.subset_idx_to_full_idx_mapping = []
            while idx_full < N_full and idx_subset < N_subset:
                file_full = self.csv_data[idx_full][0]
                file_subset = self.csv_data_subset[idx_subset][0]
                if file_full == file_subset:
                    self.subset_idx_to_full_idx_mapping.append(idx_full)
                    idx_full += 1
                    idx_subset += 1
                else:
                    idx_full += 1
            assert len(self.subset_idx_to_full_idx_mapping) == N_subset
            self.recalculate_subset_csv_extended()


        # Create transform.
        _seq_len = config.FEAT_SEQ_LEN
        self.transform = MyCrop(_seq_len)


    def recalculate_subset_csv_extended(self):
        N_full = len(self.csv_data)
        N_subset = len(self.csv_data_subset)
        self.csv_data_subset_extended = self.csv_data_subset.copy()
        self.subset_idx_to_full_idx_mapping_entended = self.subset_idx_to_full_idx_mapping.copy()

        N_to_extend = N_full - N_subset
        extended_indices = list(range(N_full))
        random.shuffle(extended_indices)
        extended_indices = extended_indices[:N_to_extend]
        for idx in extended_indices:
            self.csv_data_subset_extended.append(self.csv_data[idx])
            self.subset_idx_to_full_idx_mapping_entended.append(idx)


    def on_epoch_start(self):
        self.recalculate_subset_csv_extended()

    def __len__(self):
        return len(self.csv_data)

    def _getitem_impl(self, index: int, is_subset=False):
        # print("get_item %0.6i start... " % index, end="")

        if is_subset:
            _index = self.subset_idx_to_full_idx_mapping_entended[index]
            _csv_subset_path = self.csv_data_subset_extended[index][0]
            _csv_full_path = self.csv_data[_index][0]
            assert _csv_subset_path == _csv_full_path
        else:
            _index = index

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[_index][0]
        xlsr_states = torch.load(full_path(file_path))
        features = [self.transform(x.squeeze(0)) for x in xlsr_states]
        norm_mos = self.csv_data[_index][1]

        return (features, norm_mos)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        result_full = self._getitem_impl(index, is_subset=False)
        result_subset = self._getitem_impl(index, is_subset=True)
        features = (result_full[0], result_subset[0])
        labels = (result_full[1], result_subset[1])
        return features, labels
