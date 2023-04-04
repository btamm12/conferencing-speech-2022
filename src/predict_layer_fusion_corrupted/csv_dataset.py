from audiomentations import AddGaussianSNR, Mp3Compression, RoomSimulator, SpecFrequencyMask, TimeMask, TimeStretch, Shift
from audiomentations.core.transforms_interface import BaseTransform
from src.predict_layer_fusion_corrupted.shift_add_snr import ShiftAddSNR
from src.predict_layer_fusion_corrupted.time_mask_center import TimeMaskCenter
import pyroomacoustics as pra

import csv
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.functional import pad
from torch import Tensor
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor
from typing import List, Tuple
import warnings


from src import constants
from src.utils.split import Split
from src.utils.csv_info import STANDARDIZED_CSV_INFO
from src.utils.full_path import full_path

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
            xT = x.transpose(0,1)
            xT = pad(xT, (0, to_pad), mode="constant", value=0.0)
            x = xT.transpose(0,1)

        if unsqueezed:
            x = x.squeeze(1)

        return x

class CsvDataset(Dataset):

    def __init__(self,
                 split: Split,
                 skip_first: int = 0,
    ) -> None:
        super().__init__()

        self.split = split
        self.skip_first = skip_first

        # hopefully "point in room" test will not fail anymore
        # https://github.com/LCAV/pyroomacoustics/issues/248
        pra.constants.set("room_isinside_max_iter", 50)

        # For printing...
        split_name = str(split).lower().split(".")[1]
        print(f"Creating dataloader for {split_name} set.")

        # Select dataset.
        dataset = constants.get_dataset(split, example=False, use_35=False)

        # Type to CSV column.
        col_path = STANDARDIZED_CSV_INFO.col_audio_path
        # col_path = STANDARDIZED_CSV_INFO.col_xlsr_path

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

        # Wav2Vec2FeatureExtractor
        SAMPLING_RATE = 16000
        self.sampling_rate = SAMPLING_RATE
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=SAMPLING_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        # corruptions
        _gaussian_snrs = [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
        _mp3_bitrates = [256, 128, 64, 32, 16, 8]
        _n_rooms = 3
        _rooms = {
            "size_xy": [3.0, 4.5, 6.0],
            "size_z": [2.0, 3.0, 4.0],
            "src_xy": [1.2, 1.8, 2.4],
            "src_z": [0.8, 1.2, 1.6],
            "mic_dist": [1.0, 1.0, 1.0],
            "mic_azim": [0.0, 0.0, 0.0],
            "mic_elev": [0.0, 0.0, 0.0],
        }
        _room_absorptions = [1.0, 0.50, 0.25, 0.12, 0.0625, 0.03125]
        _room_params = [
            {
                "min_size_x": _rooms["size_xy"][i],
                "max_size_x": _rooms["size_xy"][i],
                "min_size_y": _rooms["size_xy"][i],
                "max_size_y": _rooms["size_xy"][i],
                "min_size_z": _rooms["size_z"][i],
                "max_size_z": _rooms["size_z"][i],
                "min_source_x": _rooms["src_xy"][i],
                "max_source_x": _rooms["src_xy"][i],
                "min_source_y": _rooms["src_xy"][i],
                "max_source_y": _rooms["src_xy"][i],
                "min_source_z": _rooms["src_z"][i],
                "max_source_z": _rooms["src_z"][i],
                "min_mic_distance": _rooms["mic_dist"][i],
                "max_mic_distance": _rooms["mic_dist"][i],
                "min_mic_azimuth": _rooms["mic_azim"][i],
                "max_mic_azimuth": _rooms["mic_azim"][i],
                "min_mic_elevation": _rooms["mic_elev"][i],
                "max_mic_elevation": _rooms["mic_elev"][i],
            } for i in range(_n_rooms)
        ]
        _freq_mask_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        _time_mask_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        _time_stretches = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
                           1.1, 1.2, 1.3, 1.4, 1.5]
        _shift_add_snrs = [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]

        # AddGaussianSNR, Mp3Compression, RIR (different absorptions for 3 different sizes), SpecAugment (SpecFrequencyMask), TimeMask, TimeStretch, Shift + different SNR, 
        self.corruptions: List[BaseTransform] = [
            *(AddGaussianSNR(min_snr_in_db=x, max_snr_in_db=x, p=1.0) for x in _gaussian_snrs),
            *(Mp3Compression(min_bitrate=x, max_bitrate=x, p=1.0) for x in _mp3_bitrates),
            *(
                RoomSimulator(
                    min_absorption_value=_room_absorptions[abs_idx],
                    max_absorption_value=_room_absorptions[abs_idx],
                    leave_length_unchanged=True,
                    max_order=4,
                    padding=0.1,
                    p=1.0,
                    **_room_params[room_idx],
                )
            for room_idx in range(_n_rooms)
            for abs_idx in range(len(_room_absorptions))
            ),
            # *(SpecFrequencyMask(min_mask_fraction=x, max_mask_fraction=x, p=1.0) for x in _freq_mask_fracs),
            *(TimeMaskCenter(min_band_part=x, max_band_part=x, p=1.0) for x in _time_mask_fracs),
            *(TimeStretch(min_rate=x, max_rate=x, leave_length_unchanged=False, p=1.0) for x in _time_stretches),
            *(ShiftAddSNR(min_snr_in_db=x, max_snr_in_db=x, min_fraction=0.5, max_fraction=0.5, p=1.0) for x in _shift_add_snrs),
        ]

        self.corruption_names = [
            *(f"add_gaussian_snr_{x}" for x in range(len(_gaussian_snrs))),
            *(f"mp3_compression_{x}" for x in range(len(_mp3_bitrates))),
            *(f"room_{i}_absorption_{j}"
                for i in range(_n_rooms)
                for j in range(len(_room_absorptions))
            ),
            # *(f"spec_freq_mask_{x}" for x in range(len(_freq_mask_fracs))),
            *(f"time_mask_{x}" for x in range(len(_time_mask_fracs))),
            *("time_stretch_%02d" % x for x in range(len(_time_stretches))),
            *(f"shift_add_snr_{x}" for x in range(len(_shift_add_snrs))),
        ]

        assert len(self.corruptions) == len(self.corruption_names)


    def __len__(self):
        return len(self.csv_data)

    def _getitem_impl(self, index: int):
        # print("get_item %0.6i start... " % index, end="")

        if index < self.skip_first:
            return (None, None)

        # # Load features and convert to Tensor.
        # file_path: str = self.csv_data[index][0]
        # xlsr_states = torch.load(full_path(file_path))
        # features = [self.transform(x.squeeze(0)) for x in xlsr_states]
        # norm_mos = self.csv_data[index][1]

        # Load features and convert to Tensor.
        file_path: str = self.csv_data[index][0]
        audio_np = load_audio(full_path(file_path), sampling_rate=16_000)
        corrupted_inputs = []
        for transform in self.corruptions:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_np_corrupt = transform(audio_np, sample_rate=16000)
            inputs = self.feature_extractor(
                audio_np_corrupt,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            xlsr_input = inputs["input_values"]
            corrupted_inputs.append(xlsr_input)
        norm_mos = self.csv_data[index][1]

        return (corrupted_inputs, norm_mos)


    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        return self._getitem_impl(index)
