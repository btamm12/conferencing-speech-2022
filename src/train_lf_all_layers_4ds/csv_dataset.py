import csv
import librosa
import numpy as np
import random
import soundfile as sf
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel
from tqdm.auto import tqdm
from typing import Tuple

from src.model import config
from src import constants
from src.utils_4ds.split import Split, SUBSET_SPLITS
from src.utils_4ds.csv_info import STANDARDIZED_CSV_INFO
from src.utils_4ds.full_path import full_path


def _decode_non_mp3_file_like(file, new_sr):
    # Source:
    # https://huggingface.co/docs/datasets/_modules/datasets/features/audio.html#Audio

    array, sampling_rate = sf.read(file)
    array = array.T
    array = librosa.to_mono(array)
    if new_sr and new_sr != sampling_rate:
        array = librosa.resample(
            array, orig_sr=sampling_rate, target_sr=new_sr, res_type="kaiser_best"
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
        self.start_idx = None

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
        assert self.start_idx is not None, "self.rnd_init() not called!"

        # Random crop.
        unsqueezed = False
        if x.dim() == 1:
            unsqueezed = True
            x = x.unsqueeze(1)
        assert x.dim() == 2

        if x.size(0) > self.seq_len:
            assert (
                self.start_idx != -1
            ), "self.rnd_init() returned start_idx == -1 when this shouldn't happen"
            start_idx = self.start_idx
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
        feat_name: str,  # "mfcc" or "xls-r-[SIZE]"
        split: Split,
        batch_size: int,
        xlsr_model=None,
        device=None,
    ) -> None:
        super().__init__()

        self.feat_name = feat_name
        self.split = split
        self.batch_size = batch_size  # for hacky caching of subset features

        # For printing...
        split_name = str(split).lower().split(".")[1]
        print(f"Creating dataloader for {split_name} set.")

        # Select train, val or test dataset.
        if split in SUBSET_SPLITS or split == Split.TEST:
            msg = "if include_subset is True, then only Split.TRAIN and Split.VAL are accepted"
            raise Exception(msg)
        self.num_subsets = 1
        dataset = constants.get_dataset(split, example=False, use_35=False)

        # Type to CSV column.
        col_path = STANDARDIZED_CSV_INFO.col_audio_path
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
                audio_path: str = in_row[col_path]
                in_subset: bool = in_row[col_subset] == "True"
                norm_mos = torch.tensor(float(in_row[col_mos]))
                self.csv_data.append([audio_path, in_subset, norm_mos])

        # wav2vec2 feature extractor + model
        SAMPLING_RATE = 16000
        self.sampling_rate = SAMPLING_RATE
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=SAMPLING_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )

        if xlsr_model is None:
            print(f"> Loading XLS-R model: {feat_name}")
            if constants.XLSR_DIRS[feat_name].exists():
                _model = AutoModel.from_pretrained(str(constants.XLSR_DIRS[feat_name]))
            else:
                _model = Wav2Vec2Model.from_pretrained(f"facebook/{feat_name}")
            self.device = "cuda"
            self.xlsr_model = _model.to(self.device)
            print("Model loaded.")
        else:
            print("Reusing existing XLS-R model.")
            self.xlsr_model = xlsr_model
            self.device = device
            
        
        # self.debug_N = 200

        self.debug_N = 100

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
        self.transform = MyCrop(_seq_len)

        self.audio_cache = self._create_audio_cache()

    def _create_audio_cache(self):
        print("Creating audio cache...")
        cache = []

        _n = len(self.csv_data)
        # if len(self.csv_data) > 100000:
        #     _n = self.debug_N
        # _n = min(len(self.csv_data), self.debug_N)
        for index in tqdm(range(_n)):
            audio_path: str = self.csv_data[index][0]
            audio_np = load_audio(full_path(audio_path), sampling_rate=16_000)
            cache.append(audio_np)
        print("Done.")
        return cache

    def new_epoch(self):
        self.batch_counter = 0
        self.subset_emitter = []
        self.subset_buffer = []

    def on_epoch_start(self):
        self.new_epoch()

    def __len__(self):
        # _n = min(len(self.csv_data), self.debug_N)
        _n = len(self.csv_data)
        return _n

    def _calc_xlsr(self, audio_np):
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        xlsr_input = inputs["input_values"]
        xlsr_input_dev = xlsr_input.to(self.device)

        with torch.no_grad():
            xlsr_output = self.xlsr_model(xlsr_input_dev, output_hidden_states=True)

        return xlsr_output.hidden_states

    def _getitem_impl(self, index: int, use_cache: bool = True):
        if use_cache:
            audio_np = self.audio_cache[index]
        else:
            audio_path: str = self.csv_data[index]tter has elem[0]
            audio_np = load_audio(full_path(audio_path), sampling_rate=16_000)
        xlsr_states = self._calc_xlsr(audio_np)
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
            to_emit = self.subset_buffer[: self.batch_size]
            self.subset_buffer = self.subset_buffer[self.batch_size :]
            self.subset_emitter.extend(to_emit)

        # Emit samples while the emitter has elements.
        subset_emitted = False
        if len(self.subset_emitter) > 0:
            result_subset = self.subset_emitter.pop(0)
            subset_emitted = True
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
