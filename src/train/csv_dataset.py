import csv
import librosa
import numpy as np
import random
import soundfile as sf
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset
from typing import List, Tuple
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel

from src.model import config
from src import constants
from src.utils.split import Split
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.full_path import full_path
from src.train.zip_next_features.zip_next_features import read_xlsr_zip


def _decode_non_mp3_file_like(file, new_sr):
    # Source:
    # https://huggingface.co/docs/datasets/_modules/datasets/features/audio.html#Audio

    array, sampling_rate = sf.read(file)
    array = array.T
    array = librosa.to_mono(array)
    if new_sr and new_sr != sampling_rate:
        array = librosa.resample(
            array,
            orig_sr=sampling_rate,
            target_sr=new_sr,
            res_type="kaiser_best"
        )
        sampling_rate = new_sr
    return array, sampling_rate


def load_audio(file_path: str, sampling_rate: int) -> torch.Tensor:
    array, _ = _decode_non_mp3_file_like(file_path, sampling_rate)
    array = np.float32(array)
    return array


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
        xlsr_name: str,
        split: Split,
        example: bool,
        use_35: bool,
        use_caching: bool,
        use_ram: bool,
        include_subset: bool = True,
        cache_start_layer: int = None,
        cache_end_layer: int = None,
        xlsr_start_layer: int = None,
        xlsr_end_layer: int = None,
        use_zip: bool = False,
        device="cuda",
    ) -> None:
        super().__init__()

        # TODO: cache percentage... only 65% of features cached to not overload swap space (kills job)
        if use_caching and use_ram and (not example) and (not use_35):
            self.ram_percent = 0.65
        else:
            self.ram_percent = 1.0

        self.xlsr_name = xlsr_name
        self.split = split
        self.example = example
        self.use_35 = use_35
        self.use_caching = use_caching
        if self.use_caching:
            if cache_start_layer is None or cache_end_layer is None:
                raise Exception("Cached to be fetched from the cached features must be given")
            if xlsr_start_layer is None or xlsr_end_layer is None:
                raise Exception("Indices to be fetched from the cached features must be given")
        self.cache_start_layer = cache_start_layer
        self.cache_end_layer = cache_end_layer
        self.xlsr_start_layer = xlsr_start_layer
        self.xlsr_end_layer = xlsr_end_layer
        if use_ram:
            if not use_caching:
                raise Exception("using RAM cache requires feature caching (use_caching must be True)")
        self.use_ram = use_ram
        self.use_zip = use_zip
        self.device = device
        self.include_subset = include_subset

        # For printing...
        split_name = str(split).lower().split(".")[1]
        example_str = "(example) " if example else ""
        print(f"{example_str}Creating dataloader for {split_name} set (use_caching=={use_caching}).")

        # Select train, val or test dataset.
        if self.include_subset:
            if split == Split.TRAIN_SUBSET or split == Split.VAL_SUBSET or Split == Split.TEST:
                msg = "if include_subset is True, then only Split.TRAIN and Split.VAL are accepted"
                raise Exception(msg)
            if split == Split.TRAIN:
                split_subset = Split.TRAIN_SUBSET
            if split == Split.VAL:
                split_subset = Split.VAL_SUBSET
            dataset = constants.get_dataset(split, example, use_35)
            dataset_subset = constants.get_dataset(split_subset, example, use_35)

        # Type to CSV column.
        if use_caching:
            col_path = STANDARDIZED_CSV_INFO.col_xlsr_path
        else:
            col_path = STANDARDIZED_CSV_INFO.col_audio_path

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
        if self.use_caching:
            _seq_len = config.FEAT_SEQ_LEN
        else:
            _seq_len = config.AUDIO_SEQ_LEN
        self.transform = MyCrop(_seq_len)

        # Create Wav2Vec2 extractor if XLS-R feature caching is disabled.
        if not self.use_caching:
            self.helper = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True
            )
            if constants.XLSR_DIRS[xlsr_name].exists():
                _model = AutoModel.from_pretrained(str(constants.XLSR_DIRS[xlsr_name]))
            else:
                _model = Wav2Vec2Model.from_pretrained(f"facebook/{xlsr_name}")
            self.model_cpu = _model
            self.model = None

        # If using RAM as cache, then pre-load entire dataset into RAM.
        if self.use_ram:
            if self.use_zip:
                print("Loading features from ZIP. This may take some time...")
                self.ram_features = read_xlsr_zip(split, example, use_35)
                self.num_ram_features = len(self.ram_features)
                print("Finished.")
            else:
                self.ram_features = []
                print("Loading features from individual files. This may take some time...")
                print(f"Total: {len(self.csv_data)} features.")
                N = int(len(self.csv_data) * self.ram_percent)
                self.num_ram_features = N
                _ram_percent_str = "%0.2f %%" % (self.ram_percent*100)
                print(f"Sending {N} features ({_ram_percent_str}) to RAM...")
                if N > 0:
                    features, _ = self._getitem_impl(0, ignore_ram=True)
                    el_size = features[0].element_size()
                    numel = features[0].nelement()
                    num_bytes = el_size * numel * len(features) * N
                    num_gb = float(num_bytes) / (1024**3)
                else:
                    num_gb = 0.0
                str_gb = "%0.2f" % num_gb
                print(f"> Estimated RAM required: {str_gb} GB")
                for idx in range(N):
                    if (idx+1) % 100 == 0:
                        print(f"> Loading feature {idx+1} of {N}")
                    features, _ = self._getitem_impl(idx, ignore_ram=True)
                    self.ram_features.append(features)
                print("Finished.")

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

    def to_cpu(self):
        if not self.use_caching:
            _name = self.split.name.lower()
            print(f"{_name} dataset to CPU")
            del self.model
            torch.cuda.empty_cache()
            self.model = None

    def to_gpu(self):
        if not self.use_caching:
            _name = self.split.name.lower()
            print(f"{_name} dataset to GPU")
            self.model = self.model_cpu.to("cuda")


    def __len__(self):
        return len(self.csv_data)

    def _getitem_impl(self, index: int, is_subset=False, ignore_ram: bool = False):
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
        if self.use_ram and not ignore_ram and _index < self.num_ram_features:
            features = tuple(self.transform(x.squeeze(0)) for x in self.ram_features[_index])
        elif self.use_caching:
            xlsr_states = torch.load(full_path(file_path))
            features = tuple(
                self.transform(x.squeeze(0))
                for i, x in enumerate(xlsr_states)
                if self.cache_start_layer + i >= self.xlsr_start_layer and self.cache_start_layer + i < self.xlsr_end_layer
            )
        else:
            if self.model is None:
                raise Exception("self.model is None, make you call CsvDataset.to_gpu() at the beginning of each epoch")
            audio_np = load_audio(full_path(file_path), sampling_rate=16_000)
            audio_crop = self.transform(torch.from_numpy(audio_np))
            input = self.helper(
                audio_crop.numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            )["input_values"].to(self.device)
            with torch.no_grad():
                output = self.model(input, output_hidden_states=True)
            features = tuple(x.squeeze(0).cpu() for x in output.hidden_states)
        norm_mos = self.csv_data[_index][1]

        # print("done")


        return (features, norm_mos)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        result_full = self._getitem_impl(index, is_subset=False, ignore_ram=False)
        result_subset = self._getitem_impl(index, is_subset=True, ignore_ram=False)
        features = (result_full[0], result_subset[0])
        labels = (result_full[1], result_subset[1])
        return features, labels
