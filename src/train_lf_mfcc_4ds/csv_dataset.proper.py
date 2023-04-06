import csv
import random
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset
from typing import Tuple

from src.model import config
from src import constants
from src.utils_4ds.split import Split, TRAIN_SUBSET_SPLITS, VAL_SUBSET_SPLITS, SUBSET_SPLITS
from src.utils_4ds.csv_info import STANDARDIZED_CSV_INFO
from src.utils_4ds.full_path import full_path


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
        feat_name: str, # "mfcc" or "xls-r-[SIZE]"
        split: Split,
    ) -> None:
        super().__init__()

        self.feat_name = feat_name
        self.split = split

        # For printing...
        split_name = str(split).lower().split(".")[1]
        print(f"Creating dataloader for {split_name} set.")

        # Select train, val or test dataset.
        if split in SUBSET_SPLITS or split == Split.TEST:
            msg = "if include_subset is True, then only Split.TRAIN and Split.VAL are accepted"
            raise Exception(msg)
        if split == Split.TRAIN:
            split_subset = Split.TRAIN_SUBSET
            split_subsets = TRAIN_SUBSET_SPLITS
        if split == Split.VAL:
            split_subset = Split.VAL_SUBSET
            split_subsets = VAL_SUBSET_SPLITS
        # self.num_subsets = len(split_subsets)
        self.num_subsets = 1
        dataset = constants.get_dataset(split, example=False, use_35=False)
        # dataset_subsets = [
        #     constants.get_dataset(x, example=False, use_35=False)
        #     for x in split_subsets
        # ]
        dataset_subsets = [
            constants.get_dataset(x, example=False, use_35=False)
            for x in [split_subset]
        ]

        # Type to CSV column.
        col_path = STANDARDIZED_CSV_INFO.col_xlsr_path

        # Load CSV.
        self.csv_data = []  # feature_path, norm_mos
        with open(dataset.csv_path, encoding="utf-8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):

                # Skip header row.
                if idx == 0:
                    continue

                # Save feature_path, norm_mos
                file_path: str = in_row[col_path]
                file_path = file_path % feat_name # contains %s dir for input name

                norm_mos = torch.tensor(
                    float(in_row[STANDARDIZED_CSV_INFO.col_norm_mos]))
                self.csv_data.append([file_path, norm_mos])

        self.csv_data_per_subset = []
        self.idx_to_full_idx_mapping_per_subset = []
        self.csv_data_per_subset_extended = [[] for _ in range(self.num_subsets)]
        self.idx_to_full_idx_mapping_per_subset_extended = [[] for _ in range(self.num_subsets)]
        for subset_idx in range(self.num_subsets):
            _data = [] # feature_path, norm_mos
            _dataset = dataset_subsets[subset_idx]
            with open(_dataset.csv_path, encoding="utf-8", mode="r") as in_csv:
                csv_reader = csv.reader(in_csv)
                for idx, in_row in enumerate(csv_reader):

                    # Skip header row.
                    if idx == 0:
                        continue

                    # Save feature_path, norm_mos
                    file_path: str = in_row[col_path]
                    file_path = file_path % feat_name # contains %s dir for input name
                    norm_mos = torch.tensor(
                        float(in_row[STANDARDIZED_CSV_INFO.col_norm_mos]))
                    _data.append([file_path, norm_mos])
                
            # Find mapping between indices of subset and full csv. This is simple
            # since the order is the same, but the subset excludes samples of the
            # full csv.
            idx_full = 0
            idx_subset = 0
            N_full = len(self.csv_data)
            N_subset = len(_data)
            _idx_to_full_idx_mapping = []
            while idx_full < N_full and idx_subset < N_subset:
                file_full = self.csv_data[idx_full][0]
                file_subset = _data[idx_subset][0]
                if file_full == file_subset:
                    _idx_to_full_idx_mapping.append(idx_full)
                    idx_full += 1
                    idx_subset += 1
                else:
                    idx_full += 1
            assert len(_idx_to_full_idx_mapping) == N_subset

            self.csv_data_per_subset.append(_data)
            self.idx_to_full_idx_mapping_per_subset.append(_idx_to_full_idx_mapping)

            self.recalculate_subset_csv_extended(subset_idx)


        # Create transform.
        _seq_len = config.FEAT_SEQ_LEN
        self.transform = MyCrop(_seq_len)


    def recalculate_subset_csv_extended(self, subset_idx: int):
        _csv_data = self.csv_data_per_subset[subset_idx]
        _csv_data_extended = _csv_data.copy()
        _mapping = self.idx_to_full_idx_mapping_per_subset[subset_idx]
        _mapping_extended = _mapping.copy()
        N_full = len(self.csv_data)
        N_subset = len(_csv_data)

        new_N_subset = N_subset
        while new_N_subset < N_full:
            N_to_extend = min(N_full - new_N_subset, N_subset)
            extended_indices = list(range(N_subset))
            random.shuffle(extended_indices)
            extended_indices = extended_indices[:N_to_extend]
            for idx in extended_indices:
                _csv_data_extended.append(_csv_data[idx])
                _mapping_extended.append(_mapping[idx])
            new_N_subset += N_to_extend

        self.csv_data_per_subset_extended[subset_idx] = _csv_data_extended
        self.idx_to_full_idx_mapping_per_subset_extended[subset_idx] = _mapping_extended


    def on_epoch_start(self):
        for subset_idx in range(self.num_subsets):
            self.recalculate_subset_csv_extended(subset_idx)

    def __len__(self):
        return len(self.csv_data)

    def _getitem_impl(self, index: int, subset_idx: int = None):
        # print("get_item %0.6i start... " % index, end="")

        if subset_idx is None:
            _index = index
        else:
            _index = self.idx_to_full_idx_mapping_per_subset_extended[subset_idx][index]
            _csv_subset_path = self.csv_data_per_subset_extended[subset_idx][index][0]
            _csv_full_path = self.csv_data[_index][0]
            assert _csv_subset_path == _csv_full_path

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[_index][0]
        xlsr_states = torch.load(full_path(file_path)) # could also be MFCC, but I will [wrap this in brackets] ahead of time for consistency
        features = [self.transform(x.squeeze(0)) for x in xlsr_states]
        norm_mos = self.csv_data[_index][1]

        return (features, norm_mos)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        results = []
        results.append(self._getitem_impl(index, subset_idx=None))
        for subset_idx in range(self.num_subsets):
            results.append(self._getitem_impl(index, subset_idx))
        features = tuple(result[0] for result in results)
        labels = tuple(result[1] for result in results)
        return features, labels
