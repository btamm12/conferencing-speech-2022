import torch
from torch import nn, Tensor
from typing import List

from src.model_lf_all_layers_4ds.blstm_wrapper import BlstmWrapper
from src.model_lf_all_layers_4ds.config import Config, Transformer
from src.model_lf_all_layers_4ds.head_wrapper import PoolAttFF
from src.model_lf_all_layers_4ds.transformer_wrapper import TransformerWrapper

class FusionModel(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        num_layers = config.xlsr_layers
        self.num_layers = num_layers

        # Batch norm for each XLS-R layer.
        _norm_inputs = [
            nn.BatchNorm1d(config.dim_input)
            for _ in range(num_layers)
        ]
        self.norm_inputs = nn.ModuleList(_norm_inputs)

        # Early fusion (weighted sum of XLS-R states)
        self.lin_infusion = nn.Linear(num_layers, 1)
        self.transformer = self._construct_transformer(config.transformer)
        self.norm_trans = nn.BatchNorm1d(config.dim_transformer)
        self.pool = PoolAttFF(config)

        self.sigmoid = nn.Sigmoid()

        self.register_buffer("last_loss", torch.tensor(0.))

    def forward(self, features: List[Tensor]):

        # Batch norm for each XLS-R layer.
        features = tuple(
            # Transform from (N, L, C) to (N, C, L) and back.
            self.norm_inputs[i].forward(features[i].permute((0,2,1))).permute((0,2,1))
            for i in range(self.num_layers)
        )

        # Early fusion (weighted sum of XLS-R states)
        x = torch.stack(features, dim=-1)
        x = self.lin_infusion(x).squeeze(-1)
        x = self.transformer(x)
        x = self.norm_trans(x.permute((0,2,1))).permute((0,2,1))
        x = self.pool(x)

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

