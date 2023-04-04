import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
import torch
from torch.utils.data import DataLoader

from src import constants
from src.model_layer_fusion.config import CONFIGS_PER_XLSR_SIZE, TRAIN_ARGS_PER_XLSR_SIZE
from src.model_layer_fusion.model_xlsr_layers import Model
from src.train_layer_fusion.csv_dataset import CsvDataset
from src.train_layer_fusion.extract_features import extract_features
from src.utils.run_once import run_once
from src.utils.split import Split


def make_dataloader(
    xlsr_name: str,
    split: Split,
    cpus: int,
    include_subset: bool,
):

    _train_args = TRAIN_ARGS_PER_XLSR_SIZE[xlsr_name]
    
    # Create DataLoader.
    csv_dataset = CsvDataset(split,include_subset)
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=_train_args.batch_size,
        shuffle=True,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
    )
    return csv_dataloader


def _train_model(
    xlsr_name: str,
    layers: int,
    cpus: int,
    force_rezip: bool = False,
):

    # If something went wrong and we need to restart the job pipeline...
    if force_rezip:
        print("Force rezip! Recalculating XLS-R features and creating fresh ZIP for this training iteration", flush=True)
        extract_features(xlsr_name, layers, Split.TRAIN, ignore_run_once=True)
        extract_features(xlsr_name, layers, Split.VAL, ignore_run_once=True)

    # Trainer parameters.
    layers_str = ",".join(str(x) for x in layers)
    out_name = f"trained_model_{xlsr_name}_layerfusion_{layers_str}"
    model_dir = constants.MODELS_DIR.joinpath(out_name)

    # Create model.
    _configs = CONFIGS_PER_XLSR_SIZE[xlsr_name]
    _out_csv_path = model_dir / "losses.csv"
    model = Model(_configs, len(layers), _out_csv_path)

    # Create dataloader(s).
    train_dl = make_dataloader(
        xlsr_name,
        Split.TRAIN,
        cpus,
        include_subset=True,
    )
    val_dl = make_dataloader(
        xlsr_name,
        Split.VAL,
        cpus,
        include_subset=True,
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

    _train_args = TRAIN_ARGS_PER_XLSR_SIZE[xlsr_name]
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
    xlsr_name: str,
    layers: int,
    cpus: int,
    force_rezip: bool = False,
):

    # Flag name. Make sure this operation is only performed once.
    layers_str = ",".join(str(x) for x in layers)
    flag_name = f"trained_model_{xlsr_name}_layerfusion_{layers_str}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _train_model(
                xlsr_name,
                layers,
                cpus,
                force_rezip=force_rezip,
            )
        else:
            print(f"Model already trained: {flag_name}.")


if __name__ == "__main__":
    xlsr_name = "wav2vec2-xls-r-2b"
    layers = [15,36]
    cpus: int = 5
    train_model(
        xlsr_name,
        layers,
        cpus,
        force_rezip=False,
    )
