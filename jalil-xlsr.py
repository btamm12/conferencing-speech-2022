import csv
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoModel
from tqdm.auto import tqdm


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

    def __init__(self, csv_path):
        super().__init__()

        # Path to CSV containing: in_audio_path, out_xlsr_path
        self.csv_path = csv_path

        # Load CSV.
        self.csv_data = []
        with open(csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):
                if idx == 0: # Skip header row.
                    continue
                audio_path, xlsr_path = in_row
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
        audio_np = load_audio(audio_path, sampling_rate=self.sampling_rate)
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        xlsr_input = inputs["input_values"]
        return xlsr_input, xlsr_path


def extract_features(csv_path: str, xlsr_name: str, layer: int):
    
    # Create dataset.
    csv_dataset = SimpleCsvDataset(csv_path)
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=2,
        persistent_workers=False,
    )

    # Device for model computations.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: %s" % device)

    # Create model.
    ### wav2vec2-xls-r-300m = 24 + 1 layers, 1024-D
    ### wav2vec2-xls-r-1b = 48 + 1 layers, 1280-D
    ### wav2vec2-xls-r-2b = 48 + 1 layers, 1920-D
    print(f"Loading model...")
    # model = AutoModel.from_pretrained(PATH_TO_LOCAL_XLSR) # local model
    model = Wav2Vec2Model.from_pretrained(f"facebook/{xlsr_name}") # download model
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
        torch.save(xlsr, xlsr_path)


    print("")
    print(f"Finished.")


if __name__ == "__main__":
    csv_path = ""
    xlsr_name = "wav2vec2-xls-r-2b"
    layer = 42
    extract_features(csv_path, xlsr_name, layer)
