from multiprocessing import Process
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
import shutil
import torch
from torch.utils.data import DataLoader

from src import constants
from src.model.config import CONFIGS_PER_XLSR_SIZE, TRAIN_ARGS_PER_XLSR_SIZE
from src.model.model_xlsr_layers import Model
from src.train.csv_dataset import CsvDataset
from src.train.extract_features import extract_features
from src.train.zip_next_features.zip_next_features import zip_next_features
from src.utils.run_once import run_once
from src.utils.split import Split


def make_dataloader(
    xlsr_name: str,
    split: Split,
    example: bool,
    use_35: bool,
    use_caching: bool,
    use_ram: bool,
    cpus: int,
    include_subset: bool,
    cache_start_layer: int,
    cache_end_layer: int,
    xlsr_start_layer: int,
    xlsr_end_layer: int,
    use_zip: bool,
):

    _train_args = TRAIN_ARGS_PER_XLSR_SIZE[xlsr_name]
    
    # Create DataLoader.
    csv_dataset = CsvDataset(
        xlsr_name,
        split,
        example,
        use_35,
        use_caching,
        use_ram,
        include_subset,
        cache_start_layer,
        cache_end_layer,
        xlsr_start_layer,
        xlsr_end_layer,
        use_zip,
    )
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
    xlsr_start_layer: int,
    xlsr_end_layer: int,
    example: bool,
    use_35: bool,
    use_caching: bool,
    use_ram: bool,
    full_and_subset_training: bool,
    use_subset: bool,
    cpus: int,
    cache_start_layer: int = None,
    cache_end_layer: int = None,
    use_zip: bool = False,
    force_rezip: bool = False,
):
    if example:
        use_35 = False
    if full_and_subset_training:
        use_subset = False
    if not use_caching:
        cpus = 1
    if use_caching and use_ram:
        cpus = 2 # avoid having multiple cache instances over multiple processes

    # If something went wrong and we need to restart the job pipeline...
    if force_rezip:
        print("Force rezip! Recalculating XLS-R features and creating fresh ZIP for this training iteration", flush=True)
        ignore_run_once = True
        _stage_args = (xlsr_name, cache_start_layer, cache_end_layer, Split.TRAIN, example, use_35, xlsr_start_layer, xlsr_end_layer, ignore_run_once)
        extract_features(*_stage_args)
        _stage_args = (xlsr_name, cache_start_layer, cache_end_layer, Split.VAL, example, use_35, xlsr_start_layer, xlsr_end_layer, ignore_run_once)
        extract_features(*_stage_args)

    # Trainer parameters.
    layers_name = "_layers_%02i_%02i" % (xlsr_start_layer, xlsr_end_layer)
    example_name = "_example" if example else ""
    use_35_name = "_use_35" if use_35 else ""
    example_str = "(example) " if example else ""
    subset_str = "_subset" if use_subset else ""
    out_name = f"trained_model_{xlsr_name}{layers_name}{example_name}{use_35_name}{subset_str}"
    model_dir = constants.MODELS_DIR.joinpath(out_name)

    # Create model.
    _configs = CONFIGS_PER_XLSR_SIZE[xlsr_name]
    _out_csv_path = model_dir / "losses.csv"
    model = Model(_configs, xlsr_start_layer, xlsr_end_layer, _out_csv_path)

    # Handle ZIP files.
    if use_zip:
        if not constants.XLSR_ZIPS_READY_FLAG.exists():
            _src = constants.XLSR_NEXT_TRAIN_ZIP_PATH
            _dst = constants.XLSR_CUR_TRAIN_ZIP_PATH
            print(f"Moving cache zip from {_src} to {_dst}")
            shutil.move(_src, _dst)
            _src = constants.XLSR_NEXT_VAL_ZIP_PATH
            _dst = constants.XLSR_CUR_VAL_ZIP_PATH
            print(f"Moving cache zip from {_src} to {_dst}")
            shutil.move(_src, _dst)
            with open(constants.XLSR_ZIPS_READY_FLAG, mode="w") as f:
                f.write("")

    # Create dataloader(s).
    if use_subset:
        train_split = Split.TRAIN_SUBSET
        val_split = Split.VAL_SUBSET
    else:
        train_split = Split.TRAIN
        val_split = Split.VAL
    train_dl = make_dataloader(
        xlsr_name,
        train_split,
        example,
        use_35,
        use_caching,
        use_ram,
        cpus,
        include_subset=full_and_subset_training,
        cache_start_layer=cache_start_layer,
        cache_end_layer=cache_end_layer,
        xlsr_start_layer=xlsr_start_layer,
        xlsr_end_layer=xlsr_end_layer,
        use_zip=use_zip,
    )
    if val_split is None:
        val_dl = None
    else:
        val_dl = make_dataloader(
            xlsr_name,
            val_split,
            example,
            use_35,
            use_caching,
            use_ram,
            cpus,
            include_subset=full_and_subset_training,
            cache_start_layer=cache_start_layer,
            cache_end_layer=cache_end_layer,
            xlsr_start_layer=xlsr_start_layer,
            xlsr_end_layer=xlsr_end_layer,
            use_zip=use_zip,
        )



    assert val_split is not None, "val_split == None, this training type is not supported"

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
    print(f"{example_str}Using: %s" % device)

    # TODO: sloppy code
    # - for full training: max epochs == 20 (convergence around 10)
    # - for 35% model training: max_epochs == 40
    # + 1 layer: XLSR-300M = 176.5 GB / XLSR-1B = 220.5 GB / XLSR-2B = 331 GB
    # ==> measured 375 GB for 10% XLSR-1B data (dummy me said use_25, then frac=0.10)
    # ==> 100% XLS-R data = 10*375 = 3750 GB
    # ==> this is for 17 layers, so 1 layer of XLSR-1B data is 3750/17 = 220.5 GB
    # ==> multiply by (1024/1280) for XLSR-300M / multiply by (1920/1280) for XLSR-2B
    # if use_35:
    #     print(f"35% training...")
    #     _max_epochs = 40
    # else:
    #     print(f"Full training...")
    #     _max_epochs = 20
    # _max_epochs = TRAIN_ARGS_PER_XLSR_SIZE
    # print(f"==> Max epochs: {_max_epochs}")
    _train_args = TRAIN_ARGS_PER_XLSR_SIZE[xlsr_name]
    trainer_params = {
        "gpus": gpus,
        "max_epochs": _train_args.max_epochs,
        # "max_epochs": _max_epochs,
        # "strategy": "ddp",  # distributed computing
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

    # 1. in the meantime, prepare zip for next iteration...
    do_zip = True
    num_xlsr = xlsr_end_layer - xlsr_start_layer
    next_xlsr_start_layer = xlsr_end_layer
    next_xlsr_end_layer = next_xlsr_start_layer + num_xlsr
    if next_xlsr_end_layer > cache_end_layer:
        next_xlsr_end_layer = cache_end_layer
    if next_xlsr_start_layer >= cache_end_layer:
        do_zip = False # end of disk cache...

    if do_zip:
        _args = (
            next_xlsr_start_layer,
            next_xlsr_end_layer,
            cache_start_layer,
            cache_end_layer,
            example,
            use_35,
        )
        p = Process(target=zip_next_features, args=_args)
        p.start()

    if ckpt_path is not None:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, train_dl, val_dl)

    if do_zip:
        p.join() # wait until zip finished

    # Remove flag for next iteration.
    os.remove(str(constants.XLSR_ZIPS_READY_FLAG))


def train_model(
    xlsr_name: str,
    xlsr_start_layer: int,
    xlsr_end_layer: int,
    example: bool,
    use_35: bool,
    use_caching: bool,
    use_ram: bool,
    full_and_subset_training: bool,
    use_subset: bool,
    cpus: int,
    use_zip: bool = False,
    cache_start_layer: int = None,
    cache_end_layer: int = None,
    force_rezip: bool = False,
):
    if example:
        use_35 = False
    if full_and_subset_training:
        use_subset = False

    # Flag name. Make sure this operation is only performed once.
    layers_name = "_layers_%02i_%02i" % (xlsr_start_layer, xlsr_end_layer)
    example_name = "_example" if example else ""
    use_35_name = "_use_35" if use_35 else ""
    example_str = "(example) " if example else ""
    subset_str = "_subset" if use_subset else ""
    flag_name = f"trained_model_{xlsr_name}{layers_name}{example_name}{use_35_name}{subset_str}"

    # Run exactly once.
    with run_once(flag_name) as should_run:
        if should_run:
            _train_model(
                xlsr_name,
                xlsr_start_layer,
                xlsr_end_layer,
                example,
                use_35,
                use_caching,
                use_ram,
                full_and_subset_training,
                use_subset,
                cpus,
                cache_start_layer=cache_start_layer,
                cache_end_layer=cache_end_layer,
                use_zip=use_zip,
                force_rezip=force_rezip,
            )
        else:
            print(f"{example_str}Model already trained: {flag_name}.")


if __name__ == "__main__":
    xlsr_name = "wav2vec2-xls-r-300m"
    xlsr_start_layer = 0
    xlsr_end_layer = 1
    example: bool = True
    use_35: bool = False
    use_caching: bool = True
    use_ram: bool = True
    full_and_subset_training = True
    use_subset = False
    cpus: int = 1
    cache_start_layer = 0
    cache_end_layer = 17
    use_zip: bool = True
    train_model(
        xlsr_name,
        xlsr_start_layer,
        xlsr_end_layer,
        example,
        use_35,
        use_caching,
        use_ram,
        full_and_subset_training,
        use_subset,
        cpus,
        use_zip,
        cache_start_layer,
        cache_end_layer,
        force_rezip=True,
    )
