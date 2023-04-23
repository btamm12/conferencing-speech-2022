from audiomentations import (
    AddGaussianSNR,
    HighPassFilter,
    LowPassFilter,
    Mp3Compression,
    PitchShift,
    RoomSimulator,
    SpecFrequencyMask,
    TimeMask,
    TimeStretch,
    Shift,
)
from audiomentations.core.transforms_interface import BaseTransform
from src.predict_layer_fusion_41_corrupted.shift_add_snr import ShiftAddSNR
from src.predict_layer_fusion_41_corrupted.time_mask_final import TimeMaskFinal
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
        part: int,
        num_parts: int,
        skip_first: int = 0,
    ) -> None:
        super().__init__()

        self.split = split
        self.part = part # partition splitting on ESAT
        self.num_parts = num_parts 
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
                    float(in_row[STANDARDIZED_CSV_INFO.col_norm_mos])
                )
                self.csv_data.append([file_path, norm_mos])

        # Wav2Vec2FeatureExtractor
        SAMPLING_RATE = 16000
        self.sampling_rate = SAMPLING_RATE
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=SAMPLING_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )

        # corruptions
        _gaussian_snrs = [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]
        _mp3_bitrates = [256, 128, 64, 32, 16, 8]
        _n_rooms = 3 # ROOM SCENARIOS
        _size_xy = 6.0
        _size_z = 3.0
        _src_xy = 1.0
        _src_z = 1.0
        _mic_dists = [1.0, 2.0, 4.0] # mice = close -> far from source
        _mic_azim = 0.0
        _mic_elev = 0.0
        _room_absorptions = [1.0, 0.50, 0.25, 0.12, 0.0625, 0.03125]
        _room_params = [
            {
                "min_size_x": _size_xy,
                "max_size_x": _size_xy,
                "min_size_y": _size_xy,
                "max_size_y": _size_xy,
                "min_size_z": _size_z,
                "max_size_z": _size_z,
                "min_source_x": _src_xy,
                "max_source_x": _src_xy,
                "min_source_y": _src_xy,
                "max_source_y": _src_xy,
                "min_source_z": _src_z,
                "max_source_z": _src_z,
                "min_mic_distance": _mic_dists[i],
                "max_mic_distance": _mic_dists[i],
                "min_mic_azimuth": _mic_azim,
                "max_mic_azimuth": _mic_azim,
                "min_mic_elevation": _mic_elev,
                "max_mic_elevation": _mic_elev,
            }
            for i in range(_n_rooms)
        ]
        # _freq_mask_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        # _time_mask_fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        _lowpass_rolloffs = [6, 12, 18, 24, 30] # 0 (clean) will be added as first point
        _highpass_rolloffs = [6, 12, 18, 24, 30]
        _pitch_semitones = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        _time_masks_short = [(0, 0), (200, 7800), (200, 3800), (200, 1800), (200, 800), (200, 300)]
        _time_masks_long = [(0, 0), (500, 7500), (500, 3500), (500, 1500), (500, 1000), (500, 500)]
        _time_stretches = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        _shift_add_snrs = [50.0, 40.0, 30.0, 20.0, 10.0, 0.0]

        # AddGaussianSNR, Mp3Compression, RIR (different absorptions for 3 different sizes), SpecAugment (SpecFrequencyMask), TimeMask, TimeStretch, Shift + different SNR,
        corruptions: List[BaseTransform] = [
            *(
                AddGaussianSNR(min_snr_in_db=x, max_snr_in_db=x, p=1.0)
                for x in _gaussian_snrs
            ),
            *(
                Mp3Compression(min_bitrate=x, max_bitrate=x, p=1.0)
                for x in _mp3_bitrates
            ),
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
            *(
                LowPassFilter(min_cutoff_freq=400, max_cutoff_freq=400, min_rolloff=x, max_rolloff=x, p=1.0,)
                for x in _lowpass_rolloffs
            ),
            *(
                HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=2000, min_rolloff=x, max_rolloff=x, p=1.0,)
                for x in _highpass_rolloffs
            ),
            *(
                PitchShift(min_semitones=x, max_semitones=x, p=1.0,)
                for x in _pitch_semitones
            ),
            # *(SpecFrequencyMask(min_mask_fraction=x, max_mask_fraction=x, p=1.0) for x in _freq_mask_fracs),
            *(
                TimeMaskFinal(mask_ms=mask_ms, unmask_ms=unmask_ms, jitter_ms=mask_ms, p=1.0)
                for mask_ms, unmask_ms in _time_masks_short
            ),
            *(
                TimeMaskFinal(mask_ms=mask_ms, unmask_ms=unmask_ms, jitter_ms=mask_ms, p=1.0)
                for mask_ms, unmask_ms in _time_masks_long
            ),
            *(
                ShiftAddSNR(
                    min_snr_in_db=x,
                    max_snr_in_db=x,
                    min_fraction=0.5,
                    max_fraction=0.5,
                    p=1.0,
                )
                for x in _shift_add_snrs
            ),
            *(
                TimeStretch(min_rate=x, max_rate=x, leave_length_unchanged=False, p=1.0)
                for x in _time_stretches
            ),
        ]

        corruption_names = [
            "clean",
            *(f"add_gaussian_snr_{x}" for x in range(len(_gaussian_snrs))),
            *(f"mp3_compression_{x}" for x in range(len(_mp3_bitrates))),
            *(
                f"room_{i}_absorption_{j}"
                for i in range(_n_rooms)
                for j in range(len(_room_absorptions))
            ),
            *(f"lowpass_{x}" for x in range(len(_lowpass_rolloffs))),
            *(f"highpass_{x}" for x in range(len(_highpass_rolloffs))),
            *("pitch_semitone_%02d" % x for x in range(len(_pitch_semitones))),
            # *(f"spec_freq_mask_{x}" for x in range(len(_freq_mask_fracs))),
            *(f"time_mask_short_{x}" for x in range(len(_time_masks_short))),
            *(f"time_mask_long_{x}" for x in range(len(_time_masks_long))),
            *(f"shift_add_snr_{x}" for x in range(len(_shift_add_snrs))),
            *("time_stretch_%02d" % x for x in range(len(_time_stretches))),
        ]

        assert len(corruptions) + 1 == len(corruption_names) # + clean

        self.all_corruption_names = corruption_names
        self.all_corruptions = corruptions
        num_all_corrupts = len(self.all_corruption_names)


        # partition definition:
        part_start = int(num_all_corrupts * part / num_parts)
        part_end = int(num_all_corrupts * (part+1) / num_parts)
        self.corruptions = corruptions[part_start:part_end]
        self.corruption_names = corruption_names[part_start:part_end]

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
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        clean_input = inputs["input_values"]
        corrupted_inputs.append(clean_input)
        for transform in self.corruptions:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_np_corrupt = transform(audio_np, sample_rate=16000)

                # small deviation possible with mp3 compression
                if isinstance(transform, Mp3Compression):
                    if len(audio_np_corrupt) > len(audio_np):
                        audio_np_corrupt = audio_np_corrupt[:len(audio_np)]
                    if len(audio_np_corrupt) < len(audio_np):
                        to_pad = len(audio_np) - len(audio_np_corrupt)
                        audio_np_corrupt = np.pad(audio_np_corrupt, (0,to_pad))
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
