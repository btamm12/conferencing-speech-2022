import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
import torch
from torch.utils.data import DataLoader
from typing import List

from src import constants
from src.model_lf_all_layers_4ds.config import (
    CONFIGS_PER_XLSR_SIZE,
    TRAIN_ARGS_PER_XLSR_SIZE,
)
from src.model_lf_all_layers_4ds.model_ptl import Model
from src.train_lf_all_layers_4ds.csv_dataset import CsvDataset
from src.utils_4ds.run_once import run_once
from src.utils_4ds.split import Split


def make_dataloader(
    feat_name: str,
    batch_size: int,
    split: Split,
    cpus: int,
    xlsr_model=None,
    device=None,
):

    # Create DataLoader.
    csv_dataset = CsvDataset(feat_name, split, batch_size, xlsr_model, device)
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpus - 1,
        persistent_workers=(cpus > 1),
        prefetch_factor=4 if (cpus > 1) else 2,  # (default 2)
    )
    return csv_dataloader, csv_dataset.xlsr_model, csv_dataset.device


def _train_model(
    xlsr_name: str,
    cpus: int,
):

    # Trainer parameters.
    out_name = f"trained_model_lfds4_{xlsr_name}_all_layers"
    model_dir = constants.MODELS_DIR.joinpath(out_name)

    # Create model.
    _configs = CONFIGS_PER_XLSR_SIZE[xlsr_name]
    num_layers = 24 if "300m" in xlsr_name else 48
    _out_csv_path = model_dir / "losses.csv"
    model = Model(_configs, num_layers, _out_csv_path)

    # Train args.
    _train_args = TRAIN_ARGS_PER_XLSR_SIZE[xlsr_name]

    # Create dataloader(s).
    _feat_name = xlsr_name
    train_dl, xlsr_model, device = make_dataloader(
        _feat_name, _train_args.batch_size, Split.TRAIN, cpus
    )
    val_dl, _, _ = make_dataloader(
        _feat_name, _train_args.batch_size, Split.VAL, cpus, xlsr_model, device
    )

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
        dict_per_stem = [
            {p.split("=")[0]: p.split("=")[1] for p in parts if "=" in p}
            for parts in parts_per_stem
        ]
        epoch_per_stem = [x["epoch"] for x in dict_per_stem]
        max_idx = max(range(len(epoch_per_stem)), key=epoch_per_stem.__getitem__)
        ckpt_path = ckpt_paths[max_idx]
    else:
        ckpt_path = None

    # Start main operation:
    if ckpt_path is not None:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, train_dl, val_dl)


def train_model(
    xlsr_name: str,
    cpus: int,
):

    # Flag name. Make sure this operation is only performed once.
    flag_name = f"trained_model_lfds4_{xlsr_name}_all_layers"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _train_model(
                xlsr_name,
                cpus,
            )
        else:
            print(f"Model already trained: {flag_name}.")


if __name__ == "__main__":
    xlsr_name = "wav2vec2-xls-r-300m"
    cpus: int = 1
    train_model(
        xlsr_name,
        cpus,
    )
