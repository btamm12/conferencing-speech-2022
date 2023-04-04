import csv
from io import BytesIO
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel
from tqdm.auto import tqdm
from typing import List
import zipfile

from src import constants
from src.utils.run_once import run_once
from src.utils.split import Split, DEV_SPLITS
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


class SimpleCsvDataset(Dataset):

    def __init__(self, split: Split):
        super().__init__()

        # Returns a constants.DatasetDir containing information about the dataset.
        dataset = constants.get_dataset(split, example=False, use_35=False)

        # Load CSV.
        self.csv_data = []  # feature_path, norm_mos
        with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):

                # Skip header row.
                if idx == 0:
                    continue

                # Save feature_path, norm_mos
                audio_path = in_row[STANDARDIZED_CSV_INFO.col_audio_path]
                xlsr_path = in_row[STANDARDIZED_CSV_INFO.col_xlsr_path]
                self.csv_data.append([audio_path, xlsr_path])

        SAMPLING_RATE = 16000
        self.sampling_rate = SAMPLING_RATE
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=SAMPLING_RATE,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )


    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        audio_path = self.csv_data[index][0]
        xlsr_path = self.csv_data[index][1]
        audio_np = load_audio(full_path(audio_path), sampling_rate=16_000)
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        xlsr_input = inputs["input_values"]
        return xlsr_input, xlsr_path


def _extract_features(
    xlsr_name: str,
    layer: int,
    split: Split, 
):

    # Verify good layer indices.
    if xlsr_name == "wav2vec2-xls-r-300m":
        num_layers = 24 + 1
    elif xlsr_name == "wav2vec2-xls-r-1b":
        num_layers = 48 + 1
    elif xlsr_name == "wav2vec2-xls-r-2b":
        num_layers = 48 + 1
    else:
        raise Exception(f"Unknown xlsr_name: {xlsr_name}")

    if layer < 0 or layer >= num_layers:
        raise Exception(f"Layer must be in [0,{num_layers-1}]")

    # For printing...
    split_name = str(split).lower().split(".")[1]
    print(f"Extracting {xlsr_name} layer {layer} for {split_name} set.")

    # Create dataset.
    csv_dataset = SimpleCsvDataset(split)
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
    )

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using: %s" % device)

    # Create model.
    print(f"Loading model...")
    if constants.XLSR_DIRS[xlsr_name].exists():
        model = AutoModel.from_pretrained(str(constants.XLSR_DIRS[xlsr_name]))
    else:
        model = Wav2Vec2Model.from_pretrained(f"facebook/{xlsr_name}")
    model = model.to(device)


    # ======================================================================= #
    #                           CALCULATE FEATURES                            #
    # ======================================================================= #

    print(f"Calculating features for {len(csv_dataset)} audio files...")
    for xlsr_input, xlsr_path in tqdm(csv_dataloader):
        with torch.no_grad():
            output = model(xlsr_input.to(device), output_hidden_states=True)
        xlsr = output.hidden_states[layer].cpu()

        # Save results to .pt files.
        torch.save(xlsr, full_path(xlsr_path))


    print("")
    print(f"Finished.")


def extract_features(
    xlsr_name: str,
    layer: int,
    split: Split,
    ignore_run_once: bool = False,
):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    layer_name = "_layer_%02i" % layer
    flag_name = f"extracted_features_{xlsr_name}{layer_name}_{split_name}"

    # Normally we should extract TRAIN features, train on these features, and then
    # immediately after re-use these features for the TRAIN_SUBSET training.
    # ==> Raise exception if we ever try to extract _SUBSET features.
    if split == Split.TRAIN_SUBSET or split == Split.VAL_SUBSET:
        raise Exception(f"Normally feature extraction can be reused for {split_name} split. Please double-check setup.")

    if ignore_run_once:
        print("extract_features ignoring run once...")
        _extract_features(xlsr_name, layer, split)
        return

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _extract_features(xlsr_name, layer, split)
        else:
            print(f"Features already extracted for {split_name} split.")


if __name__ == "__main__":
    xlsr_name = "wav2vec2-xls-r-2b"
    layer = 10
    for split in [Split.TRAIN, Split.VAL]:
        extract_features(xlsr_name, layer, split)
