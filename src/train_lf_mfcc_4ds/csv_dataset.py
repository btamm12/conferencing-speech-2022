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
    def __init__(self, seq_len: int, fix_rnd_init: bool = False) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.fix_rnd_init = fix_rnd_init

    def rnd_init(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        assert x.dim() == 2

        if x.size(0) > self.seq_len:
            min_start_idx = 0
            max_start_idx = x.size(0) - self.seq_len
            self.start_idx = random.randint(min_start_idx, max_start_idx)
        else:
            self.start_idx = -1

    def forward(self, x):
        # Random crop.

        unsqueezed = False
        if x.dim() == 1:
            unsqueezed = True
            x = x.unsqueeze(1)
        assert x.dim() == 2

        if x.size(0) > self.seq_len:
            if self.fix_rnd_init:
                assert self.start_idx != -1, "self.rnd_init() returned start_idx == -1 when this shouldn't happen"
                start_idx = self.start_idx
            else:
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
        batch_size: int,
        fix_rnd_init: bool = False,
    ) -> None:
        super().__init__()

        self.feat_name = feat_name
        self.split = split
        self.batch_size = batch_size # for hacky caching of subset features

        # For printing...
        split_name = str(split).lower().split(".")[1]
        print(f"Creating dataloader for {split_name} set.")

        # Select train, val or test dataset.
        if split in SUBSET_SPLITS or split == Split.TEST:
            msg = "if include_subset is True, then only Split.TRAIN and Split.VAL are accepted"
            raise Exception(msg)
        if split == Split.TRAIN:
            split_subset = Split.TRAIN_SUBSET
        if split == Split.VAL:
            split_subset = Split.VAL_SUBSET
        self.num_subsets = 1
        dataset = constants.get_dataset(split, example=False, use_35=False)

        # Type to CSV column.
        col_path = STANDARDIZED_CSV_INFO.col_xlsr_path
        col_subset = STANDARDIZED_CSV_INFO.col_in_subset
        col_mos = STANDARDIZED_CSV_INFO.col_norm_mos

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
                in_subset: bool = in_row[col_subset] == "True"
                norm_mos = torch.tensor(float(in_row[col_mos]))
                self.csv_data.append([file_path, in_subset, norm_mos])

        # NOTE!!
        # This code assumes num_workers == 1 in the dataloader!

        # - listens to stream of (features, labels) from full dataset
        # - whenever a subset samples appears, it is added to the buffer
        # - when the buffer is full, it is emitted during the next batch
        self.subset_buffer = []
        # samples are transferred from buffer to emitter when the batch size is
        # reached, then emitted on every __getitem__()
        self.subset_emitter = []

        self.batch_counter = 0


        # Create transform.
        _seq_len = config.FEAT_SEQ_LEN
        self.transform = MyCrop(_seq_len, fix_rnd_init)

        if feat_name == "mfcc":
            self.feat_cache = self._create_feat_cache()

    def _create_feat_cache(self):
        print("Creating feature cache...")
        cache = []
        for index in range(len(self.csv_data)):
            cache.append(self._getitem_impl(index, use_cache=False))
        print("Done.")
        return cache


    def new_epoch(self):
        self.batch_counter = 0
        self.subset_emitter = []
        self.subset_buffer = []

    def on_epoch_start(self):
        self.new_epoch()

    def __len__(self):
        return len(self.csv_data)

    def _getitem_impl(self, index: int, use_cache: bool = True):
        if self.feat_name == "mfcc" and use_cache:
            return self.feat_cache[index]

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[index][0]
        xlsr_states = torch.load(full_path(file_path)) # could also be MFCC, but I will [wrap this in brackets] ahead of time for consistency
        self.transform.rnd_init(xlsr_states[0].squeeze(0))
        features = [self.transform(x.squeeze(0)) for x in xlsr_states]
        norm_mos = self.csv_data[index][2]

        return (features, norm_mos)

    def _clone(self, result):
        features, norm_mos = result
        return [f.clone() for f in features], norm_mos.clone()


    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        if self.batch_counter == 0:
            self.new_epoch()
        result_full = self._getitem_impl(index)

        # Append this sample to the subset buffer if it is in the subset.
        _in_subset = self.csv_data[index][1]
        if _in_subset:
            self.subset_buffer.append(self._clone(result_full))
        
        # Move n_batch samples from the buffer to the emitter at the start of a batch
        # (if the buffer has enough samples).
        _start_of_batch = self.batch_counter % self.batch_size == 0
        if _start_of_batch and len(self.subset_buffer) >= self.batch_size:
            to_emit = self.subset_buffer[:self.batch_size]
            self.subset_buffer = self.subset_buffer[self.batch_size:]
            self.subset_emitter.extend(to_emit)

        # Emit samples while the emitter has elements.
        subset_emitted = False
        if len(self.subset_emitter) > 0:
            result_subset = self.subset_emitter.pop(0)
            subset_emitted = True
        else:
            if self.feat_name == "mfcc":
                _dummy_feats = [torch.zeros((0,))]
            else:
                _dummy_feats = [torch.zeros((0,)), torch.zeros((0,))]
            _dummy_label = torch.zeros((0,))
            result_subset = (_dummy_feats, _dummy_label)

        # Combine the results and include an indicator if the subset is non-empty.
        features = (result_full[0], result_subset[0])
        labels = (result_full[1], result_subset[1])
        subset_emitted = torch.tensor(int(subset_emitted), dtype=torch.int64)

        self.batch_counter += 1
        return (features, labels, subset_emitted)
