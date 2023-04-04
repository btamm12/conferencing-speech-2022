import csv
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model, AutoModel
from tqdm.auto import tqdm
from typing import List

import numpy as np
from scipy.io.wavfile import write

from src import constants
from src.model_layer_fusion.config import XLSR_2B_CONFIGS, FEAT_SEQ_LEN
from src.model_layer_fusion.model_pt import FusionModel
from src.model_layer_fusion.model_xlsr_layers import Model
from src.utils.run_once import run_once
from src.predict_layer_fusion_corrupted.csv_dataset import CsvDataset, MyCrop
from src.utils.split import Split

def _load_best_models(model_dir: Path, ds_idx: int):
    # ds_idx = 0 --> full
    # ds_idx = 1 --> subset
    SELECTED_CFG = 5

    # Load losses csv.
    # epoch,xlsr,loop,dataset,config_idx,fusion_idx,loss
    losses_rows = []
    with open(model_dir.joinpath("losses.csv"), mode="r") as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue # remove header
            loop = row[2]
            cfg = int(row[4])
            if loop == "train":
                continue # remove train rows
            if cfg != SELECTED_CFG:
                continue # only keep SELECTED_CFG
            losses_rows.append(row)

    # Find number of fusion configs.
    max_fusion = -1
    for row in losses_rows:
        _fusion_idx = int(row[5])
        if _fusion_idx > max_fusion:
            max_fusion = _fusion_idx
    num_fusions = max_fusion + 1

    # Find lowest loss per fusion config.
    best_models = []
    for idx in range(num_fusions):
        min_loss = None
        best_epoch = None
        for row in losses_rows:
            epoch = int(row[0])
            loss = float(row[-1])
            if min_loss is None or loss < min_loss:
                min_loss = loss
                best_epoch = epoch

        _cfg = XLSR_2B_CONFIGS[SELECTED_CFG]
        best_model_i = FusionModel(_cfg, num_layers=2, fusion_id=idx)
        _full_model_path = model_dir.joinpath("all-epoch=%03d.ckpt" % best_epoch)
        _full_model = Model.load_from_checkpoint(_full_model_path)
        fusion_idx = idx
        config_idx = SELECTED_CFG
        B = _full_model.num_fusions
        C = _full_model.num_configs
        model_idx = (B*C) * ds_idx + C * fusion_idx + config_idx
        best_model_i.load_state_dict(_full_model.models[model_idx].state_dict())
        best_models.append(best_model_i)

    return best_models


def _predict_model(split: Split, cpus: int):
    split_name = str(split).lower().split(".")[1]

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using: %s" % device)

    # Load best model.
    # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/924#issuecomment-591108496
    model_name = "trained_model_wav2vec2-xls-r-2b_layerfusion_15,36"
    model_dir = constants.MODELS_DIR.joinpath(model_name)
    ds_idx = 1 if "subset" in split_name else 0
    models = _load_best_models(model_dir, ds_idx)
    N = len(models)
    for i in range(N):
        models[i] = models[i].to(device)

    # Create dataloader.
    csv_dataset = CsvDataset(split)

    my_crop = MyCrop(FEAT_SEQ_LEN)

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using: %s" % device)

    # Create model.
    print(f"Loading model...")
    xlsr_name = "wav2vec2-xls-r-2b"
    if constants.XLSR_DIRS[xlsr_name].exists():
        model = AutoModel.from_pretrained(str(constants.XLSR_DIRS[xlsr_name]))
    else:
        model = Wav2Vec2Model.from_pretrained(f"facebook/{xlsr_name}")
    model = model.to(device)

    # Get corruption names.
    corruption_names = csv_dataset.corruption_names
    num_corrupts = len(corruption_names)

    # Output path.
    dataset = constants.get_dataset(split, example=False, use_35=False)
    out_file_names = [[f"prediction_{split_name}_{x}_fusion_{i}.csv" for i in range(N)] for x in corruption_names]
    out_paths = [[dataset.predictions_dir.joinpath(x) for x in y] for y in out_file_names]
    exists = any(any(x.exists() for x in y) for y in out_paths)
    all_exist = all(all(x.exists() for x in y) for y in out_paths)
    if exists and not all_exist:
        raise Exception("only some CSV's exist")
    if exists:
        r_files = [[open(x, "r", encoding="utf-8", buffering=1) for x in y] for y in out_paths]
        N_done = [[len(f.readlines())-1 for f in y] for y in r_files]
        for i in range(len(r_files)):
            for j in range(len(r_files[i])):
                r_files[i][j].close()
    else:
        N_done = [[0 for _ in y] for y in out_paths]
    out_files = [[open(x, "a", encoding="utf-8", buffering=1) for x in y] for y in out_paths]
    min_N_done = [min(y) for y in N_done]
    min_min = min(min_N_done)
    print(f"Skipping first: {min_min}")
    csv_dataset.skip_first = min_min

    dl = DataLoader(csv_dataset, batch_size=None, shuffle=False, num_workers=cpus-1)



    # Iterate through data.
    print(f"Running inference for {len(csv_dataset)} audio files...")
    layers = [15,36]
    for i in range(N):
        models[i].eval()
        if not exists:
            for j in range(num_corrupts):
                out_files[j][i].write("prediction" + "\n")
    for idx, (corrupted_xlsr_inputs, mos_norm) in enumerate(tqdm(dl)):
        # for i in range(num_corrupts):
        #     data = corrupted_xlsr_inputs[i].numpy().transpose()
        #     scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        #     file_name = corruption_names[i] + ".wav"
        #     write(file_name, 16000, scaled)
        for j in range(num_corrupts):
            if idx < min_N_done[j]:
                continue
            xlsr_input = corrupted_xlsr_inputs[j]
            with torch.no_grad():
                output = model(xlsr_input.to(device), output_hidden_states=True)
            xlsr_dev = [my_crop(output.hidden_states[i].squeeze(0)).unsqueeze(0) for i in layers]
            for i in range(N):
                if idx < N_done[j][i]:
                    continue
                out: torch.Tensor = models[i](xlsr_dev).cpu()
                out_denorm = out * 4.0 + 1.0  # Range 0-1 --> 1-5
                out_files[j][i].write("%0.7f" % out_denorm.item() + "\n")

    for i in range(N):
        for j in range(num_corrupts):
            out_files[j][i].close()


def predict_model(split: Split, cpus: int):

    # Flag name. Make sure this operation is only performed once.
    split_name = str(split).lower().split(".")[1]
    flag_name = f"predicted_models_hybrid_fusion_{split_name}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _predict_model(split, cpus)
        else:
            print(
                f"Prediction already made for hybrid fusion models on split {split_name}.")


if __name__ == "__main__":
    cpus: int = 1
    for split in [Split.VAL, Split.VAL_SUBSET]:
        predict_model(split, cpus)
