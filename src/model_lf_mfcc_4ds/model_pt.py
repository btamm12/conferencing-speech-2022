import torch
from torch import nn, Tensor
from typing import List

from src.model_lf_mfcc_4ds.blstm_wrapper import BlstmWrapper
from src.model_lf_mfcc_4ds.config import Config, Transformer
from src.model_lf_mfcc_4ds.head_wrapper import PoolAttFF
from src.model_lf_mfcc_4ds.transformer_wrapper import TransformerWrapper

class FusionModel(nn.Module):

    def __init__(self, config: Config, num_layers, fusion_id):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.fusion_id = fusion_id

        # Batch norm for each XLS-R layer.
        _norm_inputs = [
            nn.BatchNorm1d(config.dim_input)
            for _ in range(num_layers)
        ]
        self.norm_inputs = nn.ModuleList(_norm_inputs)

        # No fusion, only input 0, 1, ..., N-1.
        if fusion_id < num_layers:
            self.transformer = self._construct_transformer(config.transformer)
            self.norm_trans = nn.BatchNorm1d(config.dim_transformer)
            self.pool = PoolAttFF(config)

        # Early fusion (weighted sum of XLS-R states)
        elif fusion_id == num_layers:
            self.lin_infusion = nn.Linear(2, 1)
            self.transformer = self._construct_transformer(config.transformer)
            self.norm_trans = nn.BatchNorm1d(config.dim_transformer)
            self.pool = PoolAttFF(config)

        # Late fusion (weighted sum of outputs).
        elif fusion_id == num_layers + 1:
            _transformers = [
                self._construct_transformer(config.transformer)
                for _ in range(num_layers)
            ]
            self.transformers = nn.ModuleList(_transformers)
            _norm_trans = [
                nn.BatchNorm1d(config.dim_transformer)
                for _ in range(num_layers)
            ]
            self.norm_trans = nn.ModuleList(_norm_trans)
            _pools = [
                PoolAttFF(config)
                for _ in range(num_layers)
            ]
            self.pools = nn.ModuleList(_pools)
            self.lin_outfusion = nn.Linear(num_layers, 1)

        self.sigmoid = nn.Sigmoid()

        self.register_buffer("last_loss", torch.tensor(0.))

    def forward(self, features: List[Tensor]):

        # Batch norm for each XLS-R layer.
        features = tuple(
            # Transform from (N, L, C) to (N, C, L) and back.
            self.norm_inputs[i].forward(features[i].permute((0,2,1))).permute((0,2,1))
            for i in range(self.num_layers)
        )

        # No fusion, only input 0, 1, ..., N-1.
        if self.fusion_id < self.num_layers:
            x = features[self.fusion_id]
            x = self.transformer(x)
            x = self.norm_trans(x.permute((0,2,1))).permute((0,2,1))
            x = self.pool(x)

        # Early fusion (weighted sum of XLS-R states)
        elif self.fusion_id == self.num_layers:
            x = torch.stack(features, dim=-1)
            x = self.lin_infusion(x).squeeze(-1)
            x = self.transformer(x)
            x = self.norm_trans(x.permute((0,2,1))).permute((0,2,1))
            x = self.pool(x)

        # Late fusion (weighted sum of outputs).
        elif self.fusion_id == self.num_layers + 1:
            x = tuple(
                self.transformers[i].forward(features[i])
                for i in range(self.num_layers)
            )
            x = tuple(
                self.norm_trans[i].forward(x[i].permute((0,2,1))).permute((0,2,1))
                for i in range(self.num_layers)
            )
            x = tuple(
                self.pools[i].forward(x[i])
                for i in range(self.num_layers)
            )
            x = torch.stack(x, dim=-1)
            x = self.lin_outfusion(x).squeeze(-1)

        if x.dim() > 1:
            x = x.squeeze(1)
        x = self.sigmoid(x)
        return x


    def _construct_transformer(self, transformer: Transformer) -> nn.Module:
        if transformer == Transformer.NONE:
            return None
        elif transformer == Transformer.BLSTM:
            return BlstmWrapper(self.config)
        elif transformer == Transformer.TRANSFORMER:
            return TransformerWrapper(self.config)
        else:
            raise Exception("Failed to construct transformer")

