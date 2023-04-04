import pytorch_lightning as pl
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List

from src.model_layer_fusion.config import Config, Extractor, Transformer, Head
from src.model_layer_fusion.config import TRAIN_ARGS
from src.model_layer_fusion.blstm_wrapper import BlstmWrapper
from src.model_layer_fusion.head_wrapper import HeadWrapper, PoolAttFF
from src.model_layer_fusion.transformer_wrapper import TransformerWrapper
from src.model_layer_fusion.wav2vec2_wrapper import Wav2Vec2Wrapper


class Model(pl.LightningModule):

    def __init__(self, config: Config, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = num_layers

        # Needed to configure learning rate.
        # See: training_step()
        self.automatic_optimization = False

        # Intermediate transformer.
        _transformers = [
            self._construct_transformer(config.transformer)
            for _ in range(num_layers)
        ]
        self.transformers = nn.ModuleList(_transformers)

        # Pooling.
        _poolings = [
            PoolAttFF(self.config)
            for _ in range(num_layers)
        ]
        self.poolings = nn.ModuleList(_poolings)

        # Fusion head.
        self.fusion_norm = nn.BatchNorm1d(num_layers)
        self.fusion_linear = nn.Linear(num_layers, 1)
        self.fusion_sigmoid = nn.Sigmoid()


        self.save_hyperparameters()

    def forward(self, features: List[Tensor]):

        # Transformer.
        # Note: this is either None or Bi-LSTM in the paper.
        for idx, feat in enumerate(features):
            if self.transformer is not None:
                features[idx] = self.transformers[idx](feat)

        # Pooling.
        # Note: this is always PoolAttFF in the paper.
        for idx, feat in enumerate(features):
            features[idx] = self.poolings[idx](feat)

        # Fusion head.
        x = torch.cat(features, dim=-1)
        x = self.fusion_norm(x)
        x = self.fusion_linear(x)
        x = self.fusion_sigmoid(x)

        # Squeeze.
        x = x.squeeze()
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=TRAIN_ARGS.base_lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer,
            base_lr=TRAIN_ARGS.base_lr,
            max_lr=TRAIN_ARGS.max_lr,
            cycle_momentum=False,
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, }

    def training_step(self, train_batch, batch_idx):
        features, labels = train_batch

        # Source: https://pytorch-lightning.readthedocs.io/en/latest/common/optimization.html

        # Zero training gradients
        opt = self.optimizers()
        opt.zero_grad()

        # forward + backward + optimize
        out = self.forward(features)
        loss = F.mse_loss(out, labels)
        self.manual_backward(loss)
        opt.step()

        # update lr
        sch = self.lr_schedulers()
        sch.step()

        # self.log('train_loss', loss, on_step=False, on_epoch=True) # maybe this slows down code
        return loss

    def validation_step(self, val_batch, batch_idx):
        features, labels = val_batch
        out = self.forward(features)
        loss = F.mse_loss(out, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def _construct_extractor(self, extractor: Extractor) -> nn.Module:
        msg = "Constructing extractor...\n"
        if extractor == Extractor.NONE:
            msg += "> No extractor constructed: not using extractor."
            result = None
        if extractor == Extractor.XLSR:
            msg += "> Finetuning wav2vec2 model."
            result = Wav2Vec2Wrapper()
        print(msg)
        return result

    def _construct_transformer(self, transformer: Transformer) -> nn.Module:
        msg = "Constructing transformer...\n"
        if transformer == Transformer.NONE:
            msg += "> No transformer constructed: not using transformer."
            result = None
        if transformer == Transformer.BLSTM:
            msg += "> Using BLSTM."
            result = BlstmWrapper(self.config)
        if transformer == Transformer.TRANSFORMER:
            msg += "> Using transformer."
            result = TransformerWrapper(self.config)
        print(msg)
        return result

    def _construct_head(self, head: str) -> nn.Module:
        msg = "Constructing head...\n"
        if head == Head.POOLATTFF:
            msg += "> Using PoolAttFF head."
            result = HeadWrapper(self.config)
        print(msg)
        return result
