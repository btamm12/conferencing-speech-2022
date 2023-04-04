import csv
from io import BytesIO
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MFCC
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel
from tqdm.auto import tqdm
from typing import List
import zipfile

from src import constants
from src.utils_4ds.run_once import run_once
from src.utils_4ds.split import Split, DEV_SPLITS, SUBSET_SPLITS
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

    def __init__(self, input: str, split: Split):
        super().__init__()
        self.input = input

        # Returns a constants.DatasetDir containing information about the dataset.
        dataset = constants.get_dataset(split, example=False, use_35=False)

        # Load CSV.
        self.csv_data = []  # feature_path, norm_mos
        with open(dataset.csv_path, encoding="utf-8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):

                # Skip header row.
                if idx == 0:
                    continue

                # Save feature_path, norm_mos
                audio_path = in_row[STANDARDIZED_CSV_INFO.col_audio_path]
                feat_path = in_row[STANDARDIZED_CSV_INFO.col_xlsr_path]
                self.csv_data.append([audio_path, feat_path])

        if input == "xlsr":
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
        out_path = self.csv_data[index][1]
        audio_np = load_audio(full_path(audio_path), sampling_rate=16_000)
        if self.input == "mfcc":
            return audio_np, out_path
        else:
            inputs = self.feature_extractor(
                audio_np,
                sampling_rate=self.sampling_rate,
                return_tensors="pt",
            )
            xlsr_input = inputs["input_values"]
            return xlsr_input, out_path


def _extract_features(
    input: str,
    xlsr_name: str,
    layers: List[int],
    split: Split, 
):

    if input == "xlsr":
        # Verify good layer indices.
        if xlsr_name == "wav2vec2-xls-r-300m":
            num_layers = 24 + 1
        elif xlsr_name == "wav2vec2-xls-r-1b":
            num_layers = 48 + 1
        elif xlsr_name == "wav2vec2-xls-r-2b":
            num_layers = 48 + 1
        else:
            raise Exception(f"Unknown xlsr_name: {xlsr_name}")

        for layer in layers:
            if layer < 0 or layer >= num_layers:
                raise Exception(f"Layer must be in [0,{num_layers-1}]")

    # For printing...
    split_name = str(split).lower().split(".")[1]
    if input == "mfcc":
        print(f"Extracting mfcc for {split_name} set.")
    else:
        layers_str = ",".join(str(x) for x in layers)
        print(f"Extracting {xlsr_name} layers {layers_str} for {split_name} set.")

    # Create dataset.
    csv_dataset = SimpleCsvDataset(input, split)
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
    if input == "mfcc":
        print(f"Creating MFCC calculator...")
        calculate_mfcc = MFCC(sample_rate=16000)
    else:
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
    for model_input, out_path in tqdm(csv_dataloader):
        if input == "mfcc":
            mfcc: torch.Tensor = calculate_mfcc(model_input)
            out_path = out_path % input # fill in %s/ with feature name

            # Transpose MFCC from (n_mfcc, T) to (T, n_mfcc).
            # This will match the wav2vec2 size of (T, 1024).
            mfcc = mfcc.permute((1, 0))
            result = [mfcc] # wrap in [] to match xlsr_states being list of Tensors
        else:
            out_path = out_path % xlsr_name # fill in %s/ with feature name
            with torch.no_grad():
                output = model(model_input.to(device), output_hidden_states=True)
            xlsr = [output.hidden_states[i].cpu() for i in layers]
            result = xlsr

        # Save results to .pt files.
        torch.save(result, full_path(out_path))


    print("")
    print(f"Finished.")


def extract_features(
    input: str,
    xlsr_name: str,
    layers: List[int],
    split: Split,
    ignore_run_once: bool = False,
):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    if input == "mfcc":
        flag_name = f"extracted_features_lfds4_mfcc_{split_name}"
        xlsr_name = None
        layers = None
    else:
        layers_str = ",".join(str(x) for x in layers)
        flag_name = f"extracted_features_lfds4_{xlsr_name}_layers_{layers_str}_{split_name}"

    # Normally we should extract TRAIN features, train on these features, and then
    # immediately after re-use these features for the TRAIN_SUBSET training.
    # ==> Raise exception if we ever try to extract _SUBSET features.
    if split in SUBSET_SPLITS:
        raise Exception(f"Normally feature extraction can be reused for {split_name} split. Please double-check setup.")

    if ignore_run_once:
        print("extract_features ignoring run once...")
        _extract_features(input, xlsr_name, layers, split)
        return

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _extract_features(input, xlsr_name, layers, split)
        else:
            print(f"Features already extracted for {split_name} split.")


if __name__ == "__main__":
    input = "xlsr" # mfcc or xlsr
    xlsr_name = "wav2vec2-xls-r-2b"
    layers = [15, 36]
    for split in [Split.TRAIN, Split.VAL]:
        extract_features(input, xlsr_name, layers, split)
