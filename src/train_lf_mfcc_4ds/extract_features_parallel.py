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

    def __init__(self, split: Split):
        super().__init__()

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
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        xlsr_input = inputs["input_values"]
        return audio_np, xlsr_input, out_path


def _extract_features(
    split: Split, 
):

    layers_300m = [7,20]
    layers_1b = [15,36]
    layers_2b = [15,36]

    # For printing...
    split_name = str(split).lower().split(".")[1]
    print(f"Extracting features for {split_name} set.")
    print("> MFCC")
    print("> XLS-R 300M: layers " + ",".join(str(x) for x in layers_300m))
    print("> XLS-R 1B: layers " + ",".join(str(x) for x in layers_1b))
    print("> XLS-R 2B: layers " + ",".join(str(x) for x in layers_2b))

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
    print("Creating MFCC calculator...")
    calculate_mfcc = MFCC(sample_rate=16000)
    print("Loading model...")
    _name_300m = "wav2vec2-xls-r-300m"
    _name_1b = "wav2vec2-xls-r-1b"
    _name_2b = "wav2vec2-xls-r-2b"
    print("> XlS-R 300M")
    if constants.XLSR_DIRS[_name_300m].exists():
        model_300m = AutoModel.from_pretrained(str(constants.XLSR_DIRS[_name_300m]))
    else:
        model_300m = Wav2Vec2Model.from_pretrained(f"facebook/{_name_300m}")
    model_300m = model_300m.to(device)
    print("> XlS-R 1B")
    if constants.XLSR_DIRS[_name_1b].exists():
        model_1b = AutoModel.from_pretrained(str(constants.XLSR_DIRS[_name_1b]))
    else:
        model_1b = Wav2Vec2Model.from_pretrained(f"facebook/{_name_1b}")
    model_1b = model_1b.to(device)
    print("> XlS-R 2B")
    if constants.XLSR_DIRS[_name_2b].exists():
        model_2b = AutoModel.from_pretrained(str(constants.XLSR_DIRS[_name_2b]))
    else:
        model_2b = Wav2Vec2Model.from_pretrained(f"facebook/{_name_2b}")
    model_2b = model_2b.to(device)
    print("Models loaded.")


    # ======================================================================= #
    #                           CALCULATE FEATURES                            #
    # ======================================================================= #

    print(f"Calculating features for {len(csv_dataset)} audio files...")
    for audio_np, xlsr_input, out_path in tqdm(csv_dataloader):
        # This will match the wav2vec2 size of (T, 1024).
        # Transpose MFCC from (n_mfcc, T) to (T, n_mfcc).
        mfcc: torch.Tensor = calculate_mfcc(audio_np)
        mfcc = mfcc.permute((1, 0))
        torch.save([mfcc], full_path(out_path % "mfcc"))

        # XLS-R
        xlsr_input_dev = xlsr_input.to(device)
        with torch.no_grad():
            output_300m = model_300m(xlsr_input_dev, output_hidden_states=True)
            output_1b = model_1b(xlsr_input_dev, output_hidden_states=True)
            output_2b = model_2b(xlsr_input_dev, output_hidden_states=True)
            xlsr_300m = [output_300m.hidden_states[i].cpu() for i in layers_300m]
            xlsr_1b = [output_1b.hidden_states[i].cpu() for i in layers_1b]
            xlsr_2b = [output_2b.hidden_states[i].cpu() for i in layers_2b]

        torch.save(xlsr_300m, full_path(out_path % _name_300m))
        torch.save(xlsr_1b, full_path(out_path % _name_1b))
        torch.save(xlsr_2b, full_path(out_path % _name_2b))


    print("")
    print(f"Finished.")


def extract_features(
    split: Split,
    ignore_run_once: bool = False,
):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    flag_name = f"extracted_features_lfds4_parallel_{split_name}"

    # Normally we should extract TRAIN features, train on these features, and then
    # immediately after re-use these features for the TRAIN_SUBSET training.
    # ==> Raise exception if we ever try to extract _SUBSET features.
    if split in SUBSET_SPLITS:
        raise Exception(f"Normally feature extraction can be reused for {split_name} split. Please double-check setup.")

    if ignore_run_once:
        print("extract_features ignoring run once...")
        _extract_features(split)
        return

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _extract_features(split)
        else:
            print(f"Features already extracted for {split_name} split.")


if __name__ == "__main__":
    for split in [Split.TRAIN, Split.VAL]:
        extract_features(split)
