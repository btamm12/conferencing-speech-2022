from pathlib import Path
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Union

from src.model_lf_mfcc_4ds.config import Config, Input
from src.model_lf_mfcc_4ds.config import TRAIN_ARGS_MFCC, TRAIN_ARGS_PER_XLSR_SIZE
from src.model_lf_mfcc_4ds.model_pt import FusionModel
from src.train.csv_dataset import CsvDataset


def ds_to_str(ds_idx: int):
    assert ds_idx >= 0 and ds_idx < 6
    datasets = [
        "full",
        "subset",
        "pstn",
        "tencent",
        "nisqa",
        "iub",
    ]
    return datasets[ds_idx]


class Model(pl.LightningModule):
    def __init__(
        self,
        configs: Union[Config, List[Config]],
        num_layers: int,  # None for MFCC and normally 2 for XLS-R
        out_csv_path: Path,
    ):
        super().__init__()
        if not isinstance(configs, list):
            configs = [configs]
        self.configs = configs
        self.num_configs = len(configs)
        # fix my bad code when loading from checkpoint
        if "mfcc" in configs[0].name.lower() and num_layers == 1:
            num_layers = None
        if num_layers is None:
            num_layers = 1
            self.num_layers = 1
            self.num_fusions = 1  # only main output, no extra fusion
        else:
            self.num_layers = num_layers
            # self.num_fusions = num_layers + 2 # main outputs + 2 fusion paradigms
            self.num_fusions = num_layers + 1  # only sum of inputs paradigm (better)

        _input = configs[0].input
        _xlsr_name = configs[0].xlsr_name
        print("Constructing configs:")
        for idx in range(len(configs)):
            print(f"> {configs[idx].name}")
            # Check that all XLS-R names are the same.
            if configs[idx].input != _input:
                raise Exception(
                    "All configs must have the same Input type (MFCC / XLSR)."
                )
            if configs[idx].xlsr_name != _xlsr_name:
                raise Exception("All configs must be based on same XLS-R model size.")
        self.input = _input
        self.xlsr_name = _xlsr_name
        if self.input == Input.MFCC:
            self.train_args = TRAIN_ARGS_MFCC
        else:
            self.train_args = TRAIN_ARGS_PER_XLSR_SIZE[self.xlsr_name]
        self.grad_accum = self.train_args.grad_accum
        self.num_ds = (
            2  # all + challenge subset... adding more is too much I/O bottleneck
        )
        # self.num_ds = 6 # 4ds + challenge_subset + all

        # Needed to configure learning rate.
        # See: training_step()
        self.automatic_optimization = False

        # Intermediate transformer.
        _models = [
            FusionModel(self.configs[i], num_layers, f_id)
            for _ in range(self.num_ds)
            for f_id in range(self.num_fusions)
            for i in range(len(self.configs))
        ]
        _models_str = [
            f"ds{a}_fusion{f_id}_cfg{b}"
            for a in range(self.num_ds)
            for f_id in range(self.num_fusions)
            for b in range(len(self.configs))
        ]
        print(f"Constructing {len(_models)} models...")
        for idx, _str in enumerate(_models_str):
            _idx = "%02i" % idx
            print(f"> Model {_idx}: {_str}")

        # Intialize same models across DS to have the same weights
        for idx in range(self.num_configs * self.num_fusions):
            _state_dict = _models[idx].state_dict()
            for ds in range(1, self.num_ds):
                dsN_idx = idx + ds * (self.num_configs * self.num_fusions)
                _models[dsN_idx].load_state_dict(_state_dict)
        self.models = nn.ModuleList(_models)
        self.num_models = len(self.models)

        self.out_csv_path = out_csv_path
        self.subset_batch_idx = None
        self.subset_val_batch_idx = None
        self.full_global_step = None
        self.subset_global_step = None

        # # Create CSV if it doesn't exist. We will append it every epoch.
        # HEADER = ["epoch", "input", "loop", "dataset", "config_idx", "fusion_idx", "loss"]
        # if not out_csv_path.exists():
        #     out_csv_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        #     with open(out_csv_path, mode="w") as f:
        #         f.write(",".join(HEADER) + "\n")

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizers = [
            torch.optim.AdamW(
                params=self.models[i].parameters(),
                lr=self.train_args.default_lr,
                weight_decay=self.train_args.default_weight_decay,
            )
            for i in range(self.num_models)
        ]
        return optimizers

    def on_train_epoch_start(self) -> None:
        self.sum_train_losses = [torch.tensor(0.0) for _ in range(self.num_models)]
        self.sum_val_losses = [torch.tensor(0.0) for _ in range(self.num_models)]
        self.num_train_losses = [torch.tensor(0.0) for _ in range(self.num_models)]
        self.num_val_losses = [torch.tensor(0.0) for _ in range(self.num_models)]
        self.sum_eff_batch_train_losses = [
            torch.tensor(0.0) for _ in range(self.num_models)
        ]
        self.sum_eff_batch_val_losses = [
            torch.tensor(0.0) for _ in range(self.num_models)
        ]
        # self.train_losses = [[] for _ in range(self.num_models)]
        # self.val_losses = [[] for _ in range(self.num_models)]
        self.subset_batch_idx = None
        self.subset_val_batch_idx = None

        dl_train = self.trainer.train_dataloader
        if dl_train is None:
            ds_train = None
        else:
            ds_train: CsvDataset = dl_train.dataset.datasets
        ds_val: CsvDataset = self.trainer.val_dataloaders[0].dataset

        ds_val.on_epoch_start()  # Required by CsvDataset
        if ds_train is not None:
            ds_train.on_epoch_start()  # Required by CsvDataset

    def _forward_loss(
        self,
        model_idx,
        features,
        labels,
        backward: bool = False,
        opt_step: bool = False,
        no_grad=False,
    ):
        if no_grad:
            with torch.no_grad():
                out = self.models[model_idx].forward(features)
        else:
            out = self.models[model_idx].forward(features)
        loss = F.mse_loss(out, labels)
        self.models[model_idx].last_loss = loss.detach()
        if backward:
            loss.backward()
        if opt_step:
            self.optimizers()[model_idx].step()
            self.optimizers()[model_idx].zero_grad()
        return loss

    def training_step(self, train_batch, batch_idx):
        features, labels, subset_emitted = train_batch
        any_emitted = torch.any(subset_emitted).item()
        all_emitted = torch.all(subset_emitted).item()
        subset_emitted = any_emitted
        if subset_emitted:
            assert all_emitted
            if self.subset_batch_idx is None:
                self.subset_batch_idx = 0
            else:
                self.subset_batch_idx += 1
        if self.subset_global_step is None:
            self.subset_global_step = 0
        if self.full_global_step is None:
            self.full_global_step = 0

        xlsr_states_per_ds = features
        labels_per_ds = labels
        # print("train_step start... ", end="")

        # Source: https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html

        # Is this the end of an effective batch? (gradient accumulation, logging)
        _opt_step = (batch_idx + 1) % self.grad_accum == 0
        if subset_emitted:
            _opt_step_subset = (self.subset_batch_idx + 1) % self.grad_accum == 0
        else:
            _opt_step_subset = False
        _opt_step_per_ds_idx = [_opt_step, _opt_step_subset]

        _global_step_per_ds_idx = [self.full_global_step, self.subset_global_step]

        torch.cuda.synchronize(self.device)
        for idx in range(self.num_models):
            _stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(_stream):
                A = self.num_ds
                B = self.num_fusions
                C = self.num_configs
                ds_idx = (idx % (A * B * C)) // (B * C)
                fusion_idx = (idx % (B * C)) // C
                config_idx = idx % C

                if ds_idx == 1 and not subset_emitted:
                    continue

                # Warmup implementation, scale lr down with factor derived from
                # linear warmup. Based on:
                # https://github.com/Lightning-AI/lightning/issues/328#issuecomment-550114178
                _lr = self.train_args.default_lr
                warmup_steps = self.train_args.default_warmup_steps
                cur_steps = _global_step_per_ds_idx[ds_idx]
                # cur_steps = _global_step // self.num_models
                if cur_steps < warmup_steps and _opt_step:
                    lr_scale = float(cur_steps + 1) / warmup_steps
                    _lr *= lr_scale
                    for pg in self.optimizers()[idx].param_groups:
                        pg["lr"] = _lr

                # Forward, backward, optimize (when effective batch is complete = grad_accum)
                _features = xlsr_states_per_ds[ds_idx]
                _labels = labels_per_ds[ds_idx]
                _step = _opt_step_per_ds_idx[ds_idx]
                self._forward_loss(
                    idx, _features, _labels, backward=True, opt_step=_step
                )
                _loss_detach = self.models[idx].last_loss.detach().cpu()
                self.sum_train_losses[idx] += _loss_detach
                self.num_train_losses[idx] += 1.0
                self.sum_eff_batch_train_losses[idx] += _loss_detach

                # Logging (on effective batch).
                if _step:
                    indices_str = f"ds{ds_idx}_fusion{fusion_idx}_cfg{config_idx}"
                    _eff_loss = self.sum_eff_batch_train_losses[idx] / self.grad_accum
                    self.sum_eff_batch_train_losses[idx].fill_(0.0)
                    self.log(f"train_loss_{indices_str}", _eff_loss)
                    if idx == 0:
                        self.log("lr", torch.tensor(_lr))
                        self.log("eff_step", torch.tensor(cur_steps))

        torch.cuda.synchronize(self.device)

        if _opt_step:
            self.full_global_step += 1
        if _opt_step_subset:
            self.subset_global_step += 1

        # losses = torch.zeros((self.num_models,), requires_grad=False)
        # for i in range(self.num_models):
        #     losses[i] = self.models[i].last_loss.cpu()
        # self.train_losses.append(losses)

        # # Logging (on effective batch).
        # if _opt_step:
        #     for idx in range(self.num_models):
        #         A = self.num_ds
        #         B = self.num_fusions
        #         C = self.num_configs
        #         ds_idx = (idx % (A*B*C)) // (B*C)
        #         fusion_idx = (idx % (B*C)) // C
        #         config_idx = idx % C

        #         indices_str = f"ds{ds_idx}_fusion{fusion_idx}_cfg{config_idx}"
        #         _eff_batch_losses = self.train_losses[idx][-self.grad_accum:]
        #         _eff_loss = torch.stack(_eff_batch_losses).mean()
        #         # _eff_batch_losses = self.train_losses[-self.grad_accum:]
        #         # _eff_loss = torch.stack(tuple(x[idx] for x in _eff_batch_losses)).mean()
        #         self.log(f"train_loss_{indices_str}", _eff_loss)
        #         if idx == 0:
        #             self.log("lr", torch.tensor(_lr))
        #             self.log("eff_step", torch.tensor(cur_steps))

        # print("done")

    def on_train_epoch_end(self) -> None:
        _mean_loss_per_model = [
            self.sum_train_losses[i] / self.num_train_losses[i]
            for i in range(self.num_models)
        ]
        _mean_loss_per_model = torch.stack(_mean_loss_per_model, dim=0)
        # _losses = torch.stack(self.train_losses, dim=0)
        # _mean_loss_per_model = _losses.mean(dim=0)

        with open(self.out_csv_path, mode="a") as f:
            for idx in range(_mean_loss_per_model.numel()):
                A = self.num_ds
                B = self.num_fusions
                C = self.num_configs
                ds_idx = (idx % (A * B * C)) // (B * C)
                fusion_idx = (idx % (B * C)) // C
                config_idx = idx % C

                _epoch = "%i" % self.current_epoch
                _input = self.xlsr_name
                if _input is None:
                    _input = "mfcc"
                _loop = "train"
                _ds = ds_to_str(ds_idx)
                _config_idx = "%i" % config_idx
                _fusion_idx = "%i" % fusion_idx
                _loss = "%0.8f" % _mean_loss_per_model[idx].item()
                row = [_epoch, _input, _loop, _ds, _config_idx, _fusion_idx, _loss]
                f.write(",".join(row) + "\n")

    def validation_step(self, val_batch, batch_idx):
        features, labels, subset_emitted = val_batch
        any_emitted = torch.any(subset_emitted).item()
        all_emitted = torch.all(subset_emitted).item()
        subset_emitted = any_emitted
        if subset_emitted:
            assert all_emitted
            if self.subset_val_batch_idx is None:
                self.subset_val_batch_idx = 0
            else:
                self.subset_val_batch_idx += 1

        xlsr_states_per_ds = features
        labels_per_ds = labels

        # Is this the end of an effective batch? (gradient accumulation, logging)
        _opt_step = (batch_idx + 1) % self.grad_accum == 0
        if subset_emitted:
            _opt_step_subset = (self.subset_val_batch_idx + 1) % self.grad_accum == 0
        else:
            _opt_step_subset = False
        _opt_step_per_ds_idx = [_opt_step, _opt_step_subset]

        # losses = torch.zeros((self.num_models,))
        torch.cuda.synchronize(self.device)
        for idx in range(self.num_models):
            A = self.num_ds
            B = self.num_fusions
            C = self.num_configs
            ds_idx = (idx % (A * B * C)) // (B * C)
            fusion_idx = (idx % (B * C)) // C
            config_idx = idx % C

            if ds_idx == 1 and not subset_emitted:
                continue

            _stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(_stream), torch.no_grad():
                model = self.models[idx]
                out = model.forward(xlsr_states_per_ds[ds_idx])
                loss = F.mse_loss(out, labels_per_ds[ds_idx])
                # losses[idx] = loss.cpu()
                if self.trainer.sanity_checking:
                    continue

                _loss_detach = loss.detach().cpu()
                self.sum_val_losses[idx] += loss.detach().cpu()
                self.num_val_losses[idx] += 1.0
                self.sum_eff_batch_val_losses[idx] += _loss_detach

                _step = _opt_step_per_ds_idx[ds_idx]

                # Loss over effective batch (reduce noise in loss plots, consistent no matter how )
                if _step and not self.trainer.sanity_checking:
                    indices_str = f"ds{ds_idx}_fusion{fusion_idx}_cfg{config_idx}"
                    _eff_loss = self.sum_eff_batch_val_losses[idx] / self.grad_accum
                    self.sum_eff_batch_val_losses[idx].fill_(0.0)
                    # _eff_batch_losses = self.val_losses[-self.grad_accum:]
                    # _eff_loss = torch.stack(tuple(x[idx] for x in _eff_batch_losses)).mean()
                    self.log(f"val_loss_{indices_str}", _eff_loss)

                # indices_str = f"ds{ds_idx}_cfg{config_idx}_lay{layer_str}"
                # self.log(f"val_loss_{indices_str}", loss, on_step=False, on_epoch=True)
        torch.cuda.synchronize(self.device)

        # sanity check will enter here
        if self.trainer.sanity_checking:
            return

        # self.val_losses.append(losses)

    def on_validation_epoch_end(self) -> None:
        # sanity check will enter here
        if self.trainer.sanity_checking:
            print("Skipping sanity check write to CSV")
            return

        _mean_loss_per_model = [
            self.sum_val_losses[i] / self.num_val_losses[i]
            for i in range(self.num_models)
        ]
        _mean_loss_per_model = torch.stack(_mean_loss_per_model, dim=0)

        # _losses = torch.stack(self.val_losses, dim=0)
        # _mean_loss_per_model = _losses.mean(dim=0)

        with open(self.out_csv_path, mode="a") as f:
            for idx in range(_mean_loss_per_model.numel()):
                A = self.num_ds
                B = self.num_fusions
                C = self.num_configs
                ds_idx = (idx % (A * B * C)) // (B * C)
                fusion_idx = (idx % (B * C)) // C
                config_idx = idx % C

                _epoch = "%i" % self.current_epoch
                _input = self.xlsr_name
                if _input is None:
                    _input = "mfcc"
                _loop = "val"
                _ds = ds_to_str(ds_idx)
                _config_idx = "%i" % config_idx
                _fusion_idx = "%i" % fusion_idx
                _loss = "%0.8f" % _mean_loss_per_model[idx].item()
                row = [_epoch, _input, _loop, _ds, _config_idx, _fusion_idx, _loss]
                f.write(",".join(row) + "\n")
