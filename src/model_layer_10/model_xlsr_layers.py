from pathlib import Path
import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Union

from src.model_layer_10.config import Config, Transformer, Head
from src.model_layer_10.config import TRAIN_ARGS_PER_XLSR_SIZE
from src.model_layer_10.blstm_wrapper import BlstmWrapper
from src.model_layer_10.head_wrapper import HeadWrapper
from src.model_layer_10.transformer_wrapper import TransformerWrapper
from src.train.csv_dataset import CsvDataset


class Model(pl.LightningModule):

    def __init__(
        self,
        configs: Union[Config, List[Config]],
        layer: int,
        out_csv_path: Path,
        include_subset: bool = True,
    ):
        super().__init__()
        if not isinstance(configs, list):
            configs = [configs]
        self.configs = configs
        self.num_configs = len(configs)
        self.layer = layer


        _xlsr_name = configs[0].xlsr_name
        print("Constructing configs:")
        for idx in range(len(configs)):
            print(f"> {configs[idx].name}")
            # Check that all XLS-R names are the same.
            if configs[idx].xlsr_name != _xlsr_name:
                raise Exception("All configs must be based on same XLS-R model size.")
        self.xlsr_name = _xlsr_name
        self.train_args = TRAIN_ARGS_PER_XLSR_SIZE[self.xlsr_name]
        self.grad_accum = self.train_args.grad_accum
        self.include_subset = include_subset

        if self.include_subset:
            self.num_ds = 2
        else:
            self.num_ds = 1


        # Needed to configure learning rate.
        # See: training_step()
        self.automatic_optimization = False

        # Intermediate transformer.
        _models = [
            nn.Sequential(
                self._construct_transformer(self.configs[i]),
                self._construct_head(self.configs[i])
            )
            for _ in range(self.num_ds)
            for i in range(len(self.configs))
        ]
        _models_str = [
            f"ds{a}_cfg{b}_layer{self.layer}"
            for a in range(self.num_ds)
            for b in range(len(self.configs))
        ]
        print(f"Constructing {len(_models)} models...")
        for idx, _str in enumerate(_models_str):
            _idx = "%02i" % idx
            print(f"> Model {_idx}: {_str}")

        # Intialize subset model to have the same weights as the full-set model
        if self.num_ds == 2:
            for idx in range(self.num_configs):
                ds1_idx = idx + self.num_configs
                _state_dict = _models[idx].state_dict()
                _models[ds1_idx].load_state_dict(_state_dict)
        self.models = nn.ModuleList(_models)
        self.num_models = len(self.models)

        self.out_csv_path = out_csv_path

        # Create CSV if it doesn't exist. We will append it every epoch.
        HEADER = ["epoch", "xlsr", "loop", "dataset", "config_idx", "layer", "loss"]
        if not out_csv_path.exists():
            out_csv_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            with open(out_csv_path, mode="w") as f:
                f.write(",".join(HEADER) + "\n")

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                params=self.models[i].parameters(),
                lr=self.train_args.base_lr,
            )
            for i in range(self.num_models)
        ]
        lr_schedulers = [
            torch.optim.lr_scheduler.CyclicLR(
                optimizer=optimizers[i],
                base_lr=self.train_args.base_lr,
                max_lr=self.train_args.max_lr,
                cycle_momentum=False,
            )
            for i in range(self.num_models)
        ]
        return optimizers, lr_schedulers


    def on_train_epoch_start(self) -> None:
        self.train_losses = []
        self.val_losses = []

        dl_train = self.trainer.train_dataloader
        if dl_train is None:
            ds_train = None
        else:
            ds_train: CsvDataset = dl_train.dataset.datasets
        ds_val: CsvDataset = self.trainer.val_dataloaders[0].dataset

        ds_val.on_epoch_start() # Required by CsvDataset
        if ds_train is not None:
            ds_train.on_epoch_start() # Required by CsvDataset


    def training_step(self, train_batch, batch_idx):
        features, labels = train_batch
        if self.include_subset:
            xlsr_states_per_ds = features
            labels_per_ds = labels
        else:
            xlsr_states_per_ds = [features]
            labels_per_ds = [labels]
        # print("train_step start... ", end="")

        # Source: https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html

        losses = torch.zeros((self.num_models,), requires_grad=False)
        torch.cuda.synchronize(self.device)
        for idx in range(self.num_models):
            A = self.num_ds
            B = self.num_configs
            ds_idx = (idx % (A*B)) // B
            config_idx = idx % B
            layer_str = "%02i" % self.layer

            _stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(_stream):
                model = self.models[idx]
                out = model.forward(xlsr_states_per_ds[ds_idx])
                out = out.squeeze(1)
                loss = F.mse_loss(out, labels_per_ds[ds_idx])
                losses[idx] = loss.detach().cpu()
                self.manual_backward(loss)

                # Gradient accumulation:
                # Perform optimization after effective batch is complete.
                if (batch_idx + 1) % self.grad_accum == 0:
                    opt = self.optimizers()[idx]
                    opt.step()
                    opt.zero_grad() # Reset gradients for next effective batch
                    indices_str = f"ds{ds_idx}_cfg{config_idx}_lay{layer_str}"
                    _eff_batch_losses = self.train_losses[-self.grad_accum:]
                    _eff_loss = torch.stack(tuple(x[idx] for x in _eff_batch_losses)).mean()
                    self.log(f"train_loss_{indices_str}", _eff_loss)

                    # update lr
                    sch = self.lr_schedulers()[idx]
                    sch.step()

        torch.cuda.synchronize(self.device)

        self.train_losses.append(losses)
        # print("done")


    def on_train_epoch_end(self) -> None:
        _losses = torch.stack(self.train_losses, dim=0)
        _mean_loss_per_model = _losses.mean(dim=0)

        with open(self.out_csv_path, mode="a") as f:
            for idx in range(_mean_loss_per_model.numel()):
                A = self.num_ds
                B = self.num_configs
                ds_idx = (idx % (A*B)) // B
                config_idx = idx % B
                layer_str = "%02i" % self.layer

                _epoch = "%i" % self.current_epoch
                _xlsr = self.xlsr_name
                _loop = "train"
                _ds = "full" if ds_idx == 0 else "subset"
                _config_idx = "%i" % config_idx
                _layer = layer_str
                _loss = "%0.8f" % _mean_loss_per_model[idx].item()
                row = [_epoch, _xlsr, _loop, _ds, _config_idx, _layer, _loss]
                f.write(",".join(row) + "\n")


    def validation_step(self, val_batch, batch_idx):
        features, labels = val_batch
        if self.include_subset:
            xlsr_states_per_ds = features
            labels_per_ds = labels
        else:
            xlsr_states_per_ds = [features]
            labels_per_ds = [labels]

        losses = torch.zeros((self.num_models,))
        torch.cuda.synchronize(self.device)
        for idx in range(self.num_models):
            A = self.num_ds
            B = self.num_configs
            ds_idx = (idx % (A*B)) // B
            config_idx = idx % B
            layer_str = "%02i" % self.layer

            _stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.stream(_stream), torch.no_grad():
                model = self.models[idx]
                out = model.forward(xlsr_states_per_ds[ds_idx])
                if out.dim() > 1:
                    out = out.squeeze(1)
                loss = F.mse_loss(out, labels_per_ds[ds_idx])
                losses[idx] = loss.cpu()


                # Loss over effective batch (reduce noise in loss plots, consistent no matter how )
                if (batch_idx + 1) % self.grad_accum == 0 and not self.trainer.sanity_checking:
                    indices_str = f"ds{ds_idx}_cfg{config_idx}_lay{layer_str}"
                    _eff_batch_losses = self.val_losses[-self.grad_accum:]
                    _eff_loss = torch.stack(tuple(x[idx] for x in _eff_batch_losses)).mean()
                    self.log(f"val_loss_{indices_str}", _eff_loss)


                # indices_str = f"ds{ds_idx}_cfg{config_idx}_lay{layer_str}"
                # self.log(f"val_loss_{indices_str}", loss, on_step=False, on_epoch=True)
        torch.cuda.synchronize(self.device)

        # sanity check will enter here
        if self.trainer.sanity_checking:
            return

        self.val_losses.append(losses)

    def on_validation_epoch_end(self) -> None:
        # sanity check will enter here
        if self.trainer.sanity_checking:
            print("Skipping sanity check write to CSV")
            return
        _losses = torch.stack(self.val_losses, dim=0)
        _mean_loss_per_model = _losses.mean(dim=0)

        with open(self.out_csv_path, mode="a") as f:
            for idx in range(_mean_loss_per_model.numel()):
                A = self.num_ds
                B = self.num_configs
                ds_idx = (idx % (A*B)) // B
                config_idx = idx % B
                layer_str = "%02i" % self.layer

                _epoch = "%i" % self.current_epoch
                _xlsr = self.xlsr_name
                _loop = "val"
                _ds = "full" if ds_idx == 0 else "subset"
                _config_idx = "%i" % config_idx
                _layer = layer_str
                _loss = "%0.8f" % _mean_loss_per_model[idx].item()
                row = [_epoch, _xlsr, _loop, _ds, _config_idx, _layer, _loss]
                f.write(",".join(row) + "\n")


    def _construct_transformer(self, config: Config) -> nn.Module:
        if config.transformer == Transformer.NONE:
            result = nn.Identity()
        if config.transformer == Transformer.BLSTM:
            result = BlstmWrapper(config)
        if config.transformer == Transformer.TRANSFORMER:
            result = TransformerWrapper(config)
        return result

    def _construct_head(self, config: Config) -> nn.Module:
        if config.head == Head.POOLATTFF:
            result = HeadWrapper(config)
        return result
