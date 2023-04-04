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

    def __init__(self, split: Split, example: bool, use_35: bool):
        super().__init__()

        # Returns a constants.DatasetDir containing information about the dataset.
        dataset = constants.get_dataset(split, example, use_35)

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
    start_layer: int,
    end_layer: int,
    split: Split, 
    example: bool,
    use_35: bool,
    zip_start_layer: int = None,
    zip_end_layer: int = None,
):
    if example:
        use_35 = False

    # Verify good layer indices.
    if xlsr_name == "wav2vec2-xls-r-300m":
        num_layers = 24 + 1
    elif xlsr_name == "wav2vec2-xls-r-1b":
        num_layers = 48 + 1
    elif xlsr_name == "wav2vec2-xls-r-2b":
        num_layers = 48 + 1
    else:
        raise Exception(f"Unknown xlsr_name: {xlsr_name}")

    msg = f"start_layer={start_layer}, end_layer={end_layer}"
    if start_layer >= end_layer:
        msg = "start_layer must be less than end_layer."
        msg += f" start_layer={start_layer}, end_layer={end_layer}"
        raise Exception(msg)
    if start_layer < 0:
        msg = "start_layer must be greater than zero."
        msg += f" start_layer={start_layer}"
        raise Exception(msg)
    if end_layer > num_layers:
        msg = f"end_layer cannot exceed num_layers ({num_layers})."
        msg += f" start_layer={start_layer}, end_layer={end_layer}"
        raise Exception(msg)

    # Check [zip_start_layer, zip_end_layer) in [start_layer, end_layer)
    if zip_start_layer is not None or zip_end_layer is not None:
        if zip_start_layer is None or zip_end_layer is None:
            msg = "zip_start_layer must both be specified (or unspecified)."
            raise Exception(msg)
    should_zip = zip_start_layer is not None or zip_end_layer is not None
    if should_zip and zip_start_layer >= zip_end_layer:
        msg = "zip_start_layer must be less than zip_end_layer."
        msg += f" zip_start_layer={zip_start_layer}, zip_end_layer={zip_end_layer}"
        raise Exception(msg)
    if zip_start_layer is not None and zip_start_layer < start_layer:
        msg = "zip_start_layer must be greater than or equal to start_layer."
        msg += f" zip_start_layer={zip_start_layer}, start_layer={start_layer}."
        raise Exception(msg)
    if zip_end_layer is not None and zip_end_layer > end_layer:
        msg = "zip_end_layer must be greater than or equal to end_layer."
        msg += f" zip_end_layer={zip_end_layer}, end_layer={end_layer}."
        raise Exception(msg)


    # For printing...
    split_name = str(split).lower().split(".")[1]
    example_str = "(example) " if example else ""
    print(f"{example_str}Extracting {xlsr_name} layers {start_layer}:{end_layer} for {split_name} set.")

    # Create dataset.
    csv_dataset = SimpleCsvDataset(split, example, use_35)
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
    )

    # MODEL REQUIRES 16 kHz SAMPLING RATE.
    SAMPLING_RATE = 16_000

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"{example_str}Using: %s" % device)

    # Create model.
    print(f"{example_str}Loading model...")
    if constants.XLSR_DIRS[xlsr_name].exists():
        model = AutoModel.from_pretrained(str(constants.XLSR_DIRS[xlsr_name]))
    else:
        model = Wav2Vec2Model.from_pretrained(f"facebook/{xlsr_name}")
    model = model.to(device)


    # ======================================================================= #
    #                           CALCULATE FEATURES                            #
    # ======================================================================= #

    print(f"{example_str}Calculating features for {len(csv_dataset)} audio files...")
    if should_zip:
        if split == Split.TRAIN or split == Split.TRAIN_SUBSET:
            zip_path = str(constants.XLSR_NEXT_TRAIN_ZIP_PATH)
        elif split == Split.VAL or split == Split.VAL_SUBSET:
            zip_path = str(constants.XLSR_NEXT_VAL_ZIP_PATH)
        else:
            raise Exception(f"Unable to zip features for the split: {split}")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zipf:
            for xlsr_input, xlsr_path in tqdm(csv_dataloader):
                with torch.no_grad():
                    output = model(xlsr_input.to(device), output_hidden_states=True)
                xlsr: List[torch.Tensor] = [
                    x.cpu()
                    for x in output.hidden_states[start_layer:end_layer]
                ]

                # Save results to .pt files.
                torch.save(xlsr, full_path(xlsr_path))

                # Save to ZIP.
                rel_start = zip_start_layer - start_layer
                rel_end = rel_start + (zip_end_layer - zip_start_layer)
                xlsr_to_zip = xlsr[rel_start:rel_end]
                feature_io = BytesIO()
                torch.save(xlsr_to_zip, feature_io)

                feature_io.seek(0)
                zipped_path = xlsr_path
                with zipf.open(zipped_path, "w") as file:
                    file.write(feature_io.read())
    else:
        for xlsr_input, xlsr_path in tqdm(csv_dataloader):
            with torch.no_grad():
                output = model(xlsr_input.to(device), output_hidden_states=True)
            xlsr: List[torch.Tensor] = [
                x.cpu()
                for x in output.hidden_states[start_layer:end_layer]
            ]

            # Save results to .pt files.
            torch.save(xlsr, full_path(xlsr_path))


    print("")
    print(f"{example_str}Finished.")


def extract_features(
    xlsr_name: str,
    start_layer: int,
    end_layer: int,
    split: Split,
    example: bool,
    use_35: bool,
    zip_start_layer: int = None,
    zip_end_layer: int = None,
    ignore_run_once: bool = False,
):
    if example:
        use_35 = False

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    example_name = "_example" if example else ""
    use_35_name = "_use_35" if use_35 else ""
    layers_name = "_layers_%02i_%02i" % (start_layer, end_layer)
    example_str = "(example) " if example else ""
    flag_name = f"extracted_features_{xlsr_name}{layers_name}_{split_name}{example_name}{use_35_name}"

    # Normally we should extract TRAIN features, train on these features, and then
    # immediately after re-use these features for the TRAIN_SUBSET training.
    # ==> Raise exception if we ever try to extract _SUBSET features.
    if split == Split.TRAIN_SUBSET or split == Split.VAL_SUBSET:
        raise Exception(f"{example_str}Normally feature extraction can be reused for {split_name} split. Please double-check setup.")

    if ignore_run_once:
        print("extract_features ignoring run once...")
        _extract_features(xlsr_name, start_layer, end_layer, split, example, use_35, zip_start_layer, zip_end_layer)
        return

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _extract_features(xlsr_name, start_layer, end_layer, split, example, use_35, zip_start_layer, zip_end_layer)
        else:
            print(f"{example_str}Features already extracted for {split_name} split.")


# HERE'S THE PLAN
# 1 layer output over the entire dataset is roughly 324 GB
# for the largest model
#  - x2 for larger hidden (1980 instead of 1024)
# ==> 324 GB * 2 * 48 layers = 31 TB data stored!!!
#
# Not feasible for anyone, and caching saves 49x resources since 50 epochs
#
# New plan:
# - Only use 50% of dataset ==> large model over 50% dataset = 324 GB per layer
# - Only train 7 layers at time ==> 7*324GB = 2268 GB which is feasible
# - Train ALL models that use this XLS-R as the base (6 BLSTM + 3 transformers) in
#   parallel using CUDA streams
# - Also, train full set and then subset to re-use saved features from same layer
#   I.E.:
#    > 300M  0:7  full:
#       1. extract features
#       2. train 7x9 regression heads in parallel
#    > 300M  0:7  subset:
#       /  (features already extracted)
#       1. train 7x9 regression heads in parallel
#    > 300M  7:14 full:
#    > 300M  7:14 subset:
#    ...
#    > 300M 21:25 full:
#    > 300M 21:25 subset:
#    >  1B   0:7  full:
#    >  1B   0:7  subset:
#    ...

if __name__ == "__main__":
    xlsr_name = "wav2vec2-xls-r-300m"
    start_layer = 0
    end_layer = 7 # exclusive
    example: bool = True
    use_35: bool = False
    zip_start_layer = 0
    zip_end_layer = 1
    for split in [Split.TRAIN, Split.VAL]:
        extract_features(xlsr_name, start_layer, end_layer, split, example, use_35, zip_start_layer, zip_end_layer)
