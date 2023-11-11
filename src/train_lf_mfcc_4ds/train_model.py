import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
import torch
from torch.utils.data import DataLoader
from typing import List

from src import constants
from src.model_lf_mfcc_4ds.config import CONFIGS_PER_XLSR_SIZE, TRAIN_ARGS_PER_XLSR_SIZE
from src.model_lf_mfcc_4ds.config import MFCC_CONFIGS, TRAIN_ARGS_MFCC
from src.model_lf_mfcc_4ds.model_ptl import Model
from src.train_lf_mfcc_4ds.csv_dataset import CsvDataset
from src.train_lf_mfcc_4ds.extract_features import extract_features
from src.utils_4ds.run_once import run_once
from src.utils_4ds.split import Split


def make_dataloader(
    feat_name: str,
    batch_size: int,
    split: Split,
    cpus: int,
    use_localsym: bool,
):

    # Fix temporal alignment with random cropping (error present with blind submission)
    fix_rnd_init = True

    # Create DataLoader.
    csv_dataset = CsvDataset(feat_name, split, batch_size, fix_rnd_init, use_localsym)
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
        prefetch_factor=4 if (cpus > 1) else 2, # (default 2)
    )
    return csv_dataloader


def _train_model(
    input: str,
    xlsr_name: str,
    layers: List[int],
    cpus: int,
    force_rezip: bool = False,
    use_localsym: bool = False,
):

    # If something went wrong and we need to restart the job pipeline...
    if force_rezip:
        print("Force rezip! Recalculating XLS-R features and creating fresh ZIP for this training iteration", flush=True)
        extract_features(input, xlsr_name, layers, Split.TRAIN, ignore_run_once=True)
        extract_features(input, xlsr_name, layers, Split.VAL, ignore_run_once=True)

    # Trainer parameters.
    if input == "mfcc":
        out_name = f"trained_model_lfds4_mfcc"
    else:
        layers_str = ",".join(str(x) for x in layers)
        out_name = f"trained_model_lfds4_{xlsr_name}_{layers_str}"
    model_dir = constants.MODELS_DIR.joinpath(out_name)

    # Create model.
    if input == "mfcc":
        _configs = MFCC_CONFIGS
        num_layers = None
    else:
        _configs = CONFIGS_PER_XLSR_SIZE[xlsr_name]
        num_layers = len(layers)
    _out_csv_path = model_dir / "losses.csv"
    model = Model(_configs, num_layers, _out_csv_path)

    # Train args.
    if input == "mfcc":
        _train_args = TRAIN_ARGS_MFCC
    else:
        _train_args = TRAIN_ARGS_PER_XLSR_SIZE[xlsr_name]


    # Create dataloader(s).
    _feat_name = "mfcc" if input == "mfcc" else xlsr_name
    train_dl = make_dataloader(_feat_name, _train_args.batch_size, Split.TRAIN, cpus, use_localsym)
    val_dl = make_dataloader(_feat_name, _train_args.batch_size, Split.VAL, cpus, use_localsym)

    all_ckpt_callback = ModelCheckpoint(
        dirpath=str(model_dir),
        filename="all-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    summary_callback = ModelSummary(max_depth=-1)
    tb_logger = TensorBoardLogger(save_dir=str(constants.DIR_LOGS / out_name))

    # Device for model computations.
    if torch.cuda.is_available():
        gpus = 1
        device = "cuda"
    else:
        gpus = 0
        device = "cpu"
    print(f"Using: %s" % device)

    trainer_params = {
        "gpus": gpus,
        "max_epochs": _train_args.max_epochs,
        "callbacks": [all_ckpt_callback, progress_bar_callback, summary_callback],
        "enable_progress_bar": True,
        "num_sanity_val_steps": 2,
        "log_every_n_steps": 10,
        "logger": tb_logger,
    }
    trainer = pl.Trainer(**trainer_params)

    ckpt_paths = list(model_dir.glob("all*.ckpt"))
    if len(ckpt_paths) > 0:
        # Find max epoch path .
        stems = [os.path.splitext(os.path.basename(str(x)))[0] for x in ckpt_paths]
        parts_per_stem = [x.split("-") for x in stems]
        dict_per_stem = [{p.split("=")[0]: p.split("=")[1]
                        for p in parts if "=" in p} for parts in parts_per_stem]
        epoch_per_stem = [x["epoch"] for x in dict_per_stem]
        max_idx = max(range(len(epoch_per_stem)),
                        key=epoch_per_stem.__getitem__)
        ckpt_path = ckpt_paths[max_idx]
    else:
        ckpt_path = None
    

    # Start main operation:
    if ckpt_path is not None:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, train_dl, val_dl)



def train_model(
    input: str,
    xlsr_name: str,
    layers: List[int],
    cpus: int,
    force_rezip: bool = False,
    use_localsym: bool = False,
):

    # Flag name. Make sure this operation is only performed once.
    if input == "mfcc":
        flag_name = f"trained_model_lfds4_mfcc"
    else:
        layers_str = ",".join(str(x) for x in layers)
        flag_name = f"trained_model_lfds4_{xlsr_name}_{layers_str}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _train_model(
                input,
                xlsr_name,
                layers,
                cpus,
                force_rezip=force_rezip,
                use_localsym=use_localsym,
            )
        else:
            print(f"Model already trained: {flag_name}.")


if __name__ == "__main__":
    input = "mfcc"
    xlsr_name = None
    layers = None
    # input = "xlsr"
    # xlsr_name = "wav2vec2-xls-r-300m"
    # layers = [15,36]
    cpus: int = 1
    train_model(
        input,
        xlsr_name,
        layers,
        cpus,
        force_rezip=False,
        use_localsym=False,
    )
