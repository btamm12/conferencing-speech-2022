import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from src import constants
from src.model_lf_mfcc_4ds.config import MFCC_CONFIGS, CONFIGS_PER_XLSR_SIZE
from src.model_lf_mfcc_4ds.model_pt import FusionModel
from src.model_lf_mfcc_4ds.model_ptl import Model
from src.predict_all_layer_fusion_41.csv_dataset import CsvDataset
from src.utils.split import Split

def _load_best_models(input: str, model_dir: Path, ds_idx: int):
    # input in {"mfcc", "wav2vec2-xls-r-300m", "wav2vec2-xls-r-1b", "wav2vec2-xls-r-2b"}
    # ds_idx = 0 --> full
    # ds_idx = 1 --> subset
    SELECTED_CFG = 1 # transformer_32_deep 

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

        if input == "mfcc":
            _cfg = MFCC_CONFIGS[SELECTED_CFG]
        else:
            _cfg = CONFIGS_PER_XLSR_SIZE[input][SELECTED_CFG]
        _num_layers = 1 if input == "mfcc" else 2
        best_model_i = FusionModel(_cfg, num_layers=_num_layers, fusion_id=idx)
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


def _predict_models(split: Split, cpus: int):
    split_name = str(split).lower().split(".")[1]

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using: %s" % device)

    # Load best model.
    # Source: https://github.com/PyTorchLightning/pytorch-lightning/issues/924#issuecomment-591108496
    # model_name = "trained_model_wav2vec2-xls-r-2b_layerfusion_15,36"
    model_name_mfcc = "trained_model_lfds4_mfcc"
    model_name_300m = "trained_model_lfds4_wav2vec2-xls-r-300m_5,21"
    model_name_1b = "trained_model_lfds4_wav2vec2-xls-r-1b_10,41"
    model_name_2b = "trained_model_lfds4_wav2vec2-xls-r-2b_10,41"

    # ** CURRENT MODEL **
    model_dir_mfcc = constants.MODELS_DIR.joinpath(model_name_mfcc)
    model_dir_300m = constants.MODELS_DIR.joinpath(model_name_300m)
    model_dir_1b = constants.MODELS_DIR.joinpath(model_name_1b)
    model_dir_2b = constants.MODELS_DIR.joinpath(model_name_2b)

    # **BLIND SUBMISSION**
    # model_dir_mfcc = constants.MODELS_DIR.joinpath("__BLIND_SUBMISSION", model_name_mfcc + ".BLIND_SUBMISSION")
    # model_dir_300m = constants.MODELS_DIR.joinpath("__BLIND_SUBMISSION", model_name_300m + ".BLIND_SUBMISSION")
    # model_dir_1b = constants.MODELS_DIR.joinpath("__BLIND_SUBMISSION", model_name_1b + ".BLIND_SUBMISSION")
    # model_dir_2b = constants.MODELS_DIR.joinpath("__BLIND_SUBMISSION", model_name_2b + ".BLIND_SUBMISSION")
    models_mfcc = [
        _load_best_models("mfcc", model_dir_mfcc, ds_idx=0)[0],
        _load_best_models("mfcc", model_dir_mfcc, ds_idx=1)[0],
    ]
    models_300m = [
        _load_best_models("wav2vec2-xls-r-300m", model_dir_300m, ds_idx=0),
        _load_best_models("wav2vec2-xls-r-300m", model_dir_300m, ds_idx=1),
    ] 
    models_1b = [
        _load_best_models("wav2vec2-xls-r-1b", model_dir_1b, ds_idx=0),
        _load_best_models("wav2vec2-xls-r-1b", model_dir_1b, ds_idx=1),
    ]
    models_2b = [
        _load_best_models("wav2vec2-xls-r-2b", model_dir_2b, ds_idx=0),
        _load_best_models("wav2vec2-xls-r-2b", model_dir_2b, ds_idx=1),
    ]
    N_300m = len(models_300m[0])
    N_1b = len(models_1b[0])
    N_2b = len(models_2b[0])

    _root = "/home/luna.kuleuven.be/u0131128/GitHub/btamm12/conferencing-speech-2022/src/predict_all_layer_fusion_41/_best_models_for_hosting.v2/"
    torch.save(models_mfcc[0].state_dict(), _root + "model_mfcc_full.pt")
    torch.save(models_mfcc[1].state_dict(), _root + "model_mfcc_subset.pt")
    # torch.save(models_300m[0][0].state_dict(), _root + "model_300m_lay5_full.pt")
    # torch.save(models_300m[0][1].state_dict(), _root + "model_300m_lay21_full.pt")
    # torch.save(models_300m[0][2].state_dict(), _root + "model_300m_fusion_full.pt")
    # torch.save(models_300m[1][0].state_dict(), _root + "model_300m_lay5_subset.pt")
    # torch.save(models_300m[1][1].state_dict(), _root + "model_300m_lay21_subset.pt")
    # torch.save(models_300m[1][2].state_dict(), _root + "model_300m_fusion_subset.pt")
    # torch.save(models_1b[0][0].state_dict(), _root + "model_1b_lay10_full.pt")
    # torch.save(models_1b[0][1].state_dict(), _root + "model_1b_lay41_full.pt")
    # torch.save(models_1b[0][2].state_dict(), _root + "model_1b_fusion_full.pt")
    # torch.save(models_1b[1][0].state_dict(), _root + "model_1b_lay10_subset.pt")
    # torch.save(models_1b[1][1].state_dict(), _root + "model_1b_lay41_subset.pt")
    # torch.save(models_1b[1][2].state_dict(), _root + "model_1b_fusion_subset.pt")
    # torch.save(models_2b[0][0].state_dict(), _root + "model_2b_lay10_full.pt")
    # torch.save(models_2b[0][1].state_dict(), _root + "model_2b_lay41_full.pt")
    # torch.save(models_2b[0][2].state_dict(), _root + "model_2b_fusion_full.pt")
    # torch.save(models_2b[1][0].state_dict(), _root + "model_2b_lay10_subset.pt")
    # torch.save(models_2b[1][1].state_dict(), _root + "model_2b_lay41_subset.pt")
    # torch.save(models_2b[1][2].state_dict(), _root + "model_2b_fusion_subset.pt")
    
    for d in range(2):
        models_mfcc[d] = models_mfcc[d].to(device)
        for i in range(N_300m):
            models_300m[d][i] = models_300m[d][i].to(device)
        for i in range(N_1b):
            models_1b[d][i] = models_1b[d][i].to(device)
        for i in range(N_2b):
            models_2b[d][i] = models_2b[d][i].to(device)


    # Create dataloader.
    csv_dataset = CsvDataset(split)

    # Device for model computations.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using: %s" % device)

    # Output path.
    dataset = constants.get_dataset(split, example=False, use_35=False)
    out_file_names_mfcc = [f"prediction_{split_name}_mfcc_dstrain_{d}.csv" for d in range(2)]
    out_file_names_300m = [[f"prediction_{split_name}_xlsr_300m_fusion_{i}_dstrain_{d}.csv" for i in range(N_300m)] for d in range(2)]
    out_file_names_1b = [[f"prediction_{split_name}_xlsr_1b_fusion_{i}_dstrain_{d}.csv" for i in range(N_1b)] for d in range(2)]
    out_file_names_2b = [[f"prediction_{split_name}_xlsr_2b_fusion_{i}_dstrain_{d}.csv" for i in range(N_2b)] for d in range(2)]

    out_paths_mfcc = [dataset.predictions_dir.joinpath(x) for x in out_file_names_mfcc]
    out_paths_300m = [[dataset.predictions_dir.joinpath(x) for x in y] for y in out_file_names_300m]
    out_paths_1b = [[dataset.predictions_dir.joinpath(x) for x in y] for y in out_file_names_1b]
    out_paths_2b = [[dataset.predictions_dir.joinpath(x) for x in y] for y in out_file_names_2b]

    exists_mfcc = any(x.exists() for x in out_paths_mfcc)
    exists_300m = any(any(x.exists() for x in y) for y in out_paths_300m)
    exists_1b = any(any(x.exists() for x in y) for y in out_paths_1b)
    exists_2b = any(any(x.exists() for x in y) for y in out_paths_2b)
    exists = any([exists_mfcc, exists_300m, exists_1b, exists_2b])
    
    all_exist_mfcc = all(x.exists() for x in out_paths_mfcc)
    all_exist_300m = all(all(x.exists() for x in y) for y in out_paths_300m)
    all_exist_1b = all(all(x.exists() for x in y) for y in out_paths_1b)
    all_exist_2b = all(all(x.exists() for x in y) for y in out_paths_2b)
    all_exist = all([all_exist_mfcc, all_exist_300m, all_exist_1b, all_exist_2b])

    if exists and not all_exist:
        raise Exception("only some CSV's exist")
    if exists:
        r_files_mfcc = [open(x, "r", encoding="utf-8", buffering=1) for x in out_paths_mfcc]
        r_files_300m = [[open(x, "r", encoding="utf-8", buffering=1) for x in y] for y in out_paths_300m]
        r_files_1b = [[open(x, "r", encoding="utf-8", buffering=1) for x in y] for y in out_paths_1b]
        r_files_2b = [[open(x, "r", encoding="utf-8", buffering=1) for x in y] for y in out_paths_2b]
        N_done_mfcc = [len(f.readlines())-1 for f in r_files_mfcc]
        N_done_300m = [[len(f.readlines())-1 for f in y] for y in r_files_300m]
        N_done_1b = [[len(f.readlines())-1 for f in y] for y in r_files_1b]
        N_done_2b = [[len(f.readlines())-1 for f in y] for y in r_files_2b]
        for i in range(len(r_files_mfcc)):
                r_files_mfcc[i].close()
        for i in range(len(r_files_300m)):
            for j in range(len(r_files_300m[i])):
                r_files_300m[i][j].close()
        for i in range(len(r_files_1b)):
            for j in range(len(r_files_1b[i])):
                r_files_1b[i][j].close()
        for i in range(len(r_files_2b)):
            for j in range(len(r_files_2b[i])):
                r_files_2b[i][j].close()
    else:
        N_done_mfcc = [0 for _ in out_paths_mfcc]
        N_done_300m = [[0 for _ in y] for y in out_paths_300m]
        N_done_1b = [[0 for _ in y] for y in out_paths_1b]
        N_done_2b = [[0 for _ in y] for y in out_paths_2b]

    out_files_mfcc = [open(x, "a", encoding="utf-8", buffering=1) for x in out_paths_mfcc]
    out_files_300m = [[open(x, "a", encoding="utf-8", buffering=1) for x in y] for y in out_paths_300m]
    out_files_1b = [[open(x, "a", encoding="utf-8", buffering=1) for x in y] for y in out_paths_1b]
    out_files_2b = [[open(x, "a", encoding="utf-8", buffering=1) for x in y] for y in out_paths_2b]
    
    min_N_done_300m = [min(y) for y in N_done_300m]
    min_N_done_1b = [min(y) for y in N_done_1b]
    min_N_done_2b = [min(y) for y in N_done_2b]
    min_min = min([min(N_done_mfcc), min(min_N_done_300m), min(min_N_done_1b), min(min_N_done_2b)])
    print(f"Skipping first: {min_min}")
    csv_dataset.skip_first = min_min

    dl = DataLoader(csv_dataset, batch_size=None, shuffle=False, num_workers=cpus-1)

    # Iterate through data.
    print(f"Running inference for {len(csv_dataset)} audio files...")
    for d in range(2):
        models_mfcc[d].eval()
        for i in range(N_300m):
            models_300m[d][i].eval()
        for i in range(N_1b):
            models_1b[d][i].eval()
        for i in range(N_2b):
            models_2b[d][i].eval()
    if not exists:
        for d in range(2):
            out_files_mfcc[d].write("prediction" + "\n")
            for i in range(N_300m):
                out_files_300m[d][i].write("prediction" + "\n")
            for i in range(N_1b):
                out_files_1b[d][i].write("prediction" + "\n")
            for i in range(N_2b):
                out_files_2b[d][i].write("prediction" + "\n")
    for idx, (input_mfcc, input_300m, input_1b, input_2b) in enumerate(tqdm(dl)):

        # MFCC
        for d in range(2):
            if idx < N_done_mfcc[d]:
                continue
            feat_dev = tuple(x.to(device) for x in input_mfcc)
            out: torch.Tensor = models_mfcc[d](feat_dev).cpu()
            out_denorm = out * 4.0 + 1.0  # Range 0-1 --> 1-5
            out_files_mfcc[d].write("%0.7f" % out_denorm.item() + "\n")

        # 300M
        for d in range(2):
            if idx < min_N_done_300m[d]:
                continue
            feat_dev = tuple(x.to(device) for x in input_300m)
            for i in range(N_300m):
                if idx < N_done_300m[d][i]:
                    continue
                out: torch.Tensor = models_300m[d][i](feat_dev).cpu()
                out_denorm = out * 4.0 + 1.0  # Range 0-1 --> 1-5
                out_files_300m[d][i].write("%0.7f" % out_denorm.item() + "\n")

        # 1B
        for d in range(2):
            if idx < min_N_done_1b[d]:
                continue
            feat_dev = tuple(x.to(device) for x in input_1b)
            for i in range(N_1b):
                if idx < N_done_1b[d][i]:
                    continue
                out: torch.Tensor = models_1b[d][i](feat_dev).cpu()
                out_denorm = out * 4.0 + 1.0  # Range 0-1 --> 1-5
                out_files_1b[d][i].write("%0.7f" % out_denorm.item() + "\n")

        # 2B
        for d in range(2):
            if idx < min_N_done_2b[d]:
                continue
            feat_dev = tuple(x.to(device) for x in input_2b)
            for i in range(N_2b):
                if idx < N_done_2b[d][i]:
                    continue
                out: torch.Tensor = models_2b[d][i](feat_dev).cpu()
                out_denorm = out * 4.0 + 1.0  # Range 0-1 --> 1-5
                out_files_2b[d][i].write("%0.7f" % out_denorm.item() + "\n")

    for d in range(2):
        out_files_mfcc[d].close()
        for i in range(N_300m):
            out_files_300m[d][i].close()
        for i in range(N_1b):
            out_files_1b[d][i].close()
        for i in range(N_2b):
            out_files_2b[d][i].close()


def predict_models(split: Split, cpus: int):

    _predict_models(split, cpus)


    # # Flag name. Make sure this operation is only performed once.
    # split_name = str(split).lower().split(".")[1]
    # flag_name = f"predicted_models_hybrid_fusion_41_{split_name}"

    # # Run exactly once.
    # with run_once(flag_name) as should_run:
    #     if should_run:
    #         _predict_model(split, cpus, part, num_parts)
    #     else:
    #         print(
    #             f"Prediction already made for hybrid fusion models on split {split_name}.")


if __name__ == "__main__":
    cpus: int = 1
    for split in [Split.VAL, Split.VAL_SUBSET]:
        predict_models(split, cpus)
