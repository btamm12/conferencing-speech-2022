from enum import Enum
from typing import Dict, List

from src import constants

class Transformer(Enum):
    NONE = 0
    BLSTM = 1
    TRANSFORMER = 2


class Head(Enum):
    POOLATTFF = 0


class TrainConfig():

    max_epochs: int = None
    batch_size: int = None
    grad_accum: int = None
    base_lr: float = None
    max_lr: float = None

    def __init__(
        self,
        max_epochs: int,
        batch_size: int,
        base_lr: float,
        max_lr: float,
        grad_accum: int = 1,
    ) -> None:
        assert max_epochs > 0
        assert batch_size > 0
        assert base_lr > 0
        assert max_lr > 0
        assert grad_accum > 0

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.grad_accum = grad_accum


class Config():

    name: str = None
    transformer: Transformer = None
    down_proj: bool = None
    head: Head = None
    xlsr_layers: int = None

    # Dimensions.
    feat_seq_len: int = None
    dim_input: int = None
    dim_extractor: int = None
    dim_transformer: int = None
    dim_head_in: int = None
    dim_head_out: int = None

    def __init__(
        self,
        name: str,
        transformer: Transformer,
        head: Head,
        feat_seq_len: int,
        dim_transformer: int = None,
        xlsr_name: str = "wav2vec2-xls-r-300m",
        down_proj: bool = True,
        nhead_transformer: int = 4,
        nlayers_transformer: int = 2,
    ):

        # Check valid parameters.
        if transformer == Transformer.BLSTM:
            msg = "Must specify dim_transformer."
            assert dim_transformer is not None, msg
            msg = "dim_transformer must be positive."
            assert dim_transformer > 0, msg
        msg = "feat_seq_len must be positive."
        assert feat_seq_len > 0, msg

        # Check XLS-R name
        msg = f"xlsr_name must be in {constants.XLSR_NAMES}"
        assert xlsr_name in constants.XLSR_NAMES, msg

        # Save parameters.
        self.name = name
        self.transformer = transformer
        self.head = head
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.down_proj = down_proj
        self.xlsr_name = xlsr_name
        self.nhead_transformer = nhead_transformer
        self.nlayers_transformer = nlayers_transformer

        # From XLS-R paper Table 2: Model architectures. 
        # +1 since we output B hidden layers and the final embedding
        if xlsr_name == "wav2vec2-xls-r-300m":
            _b = 24
        elif xlsr_name == "wav2vec2-xls-r-1b":
            _b = 48
        elif xlsr_name == "wav2vec2-xls-r-2b":
            _b = 48
        self.xlsr_layers = _b + 1

        # Set model parameters.
        # From XLS-R paper Table 2: Model architectures.
        if self.xlsr_name == "wav2vec2-xls-r-2b":
            _h = 1920
        elif self.xlsr_name == "wav2vec2-xls-r-1b": # 1280 apparently...
            _h = 1280
        else:
            _h = 1024
        self.dim_input = _h

        self.dim_extractor = self.dim_input

        if transformer == Transformer.NONE:
            self.dim_transformer = self.dim_extractor
        elif transformer == Transformer.BLSTM:
            self.dim_transformer = dim_transformer
        elif transformer == Transformer.TRANSFORMER:
            self.dim_transformer = dim_transformer
        else:
            raise Exception("Unknown transformer.")

        if head == Head.POOLATTFF:
            self.dim_head_in = self.dim_transformer # * self.feat_seq_len
            self.dim_head_out = 1

        self.dropout = 0.0


TRAIN_ARGS = TrainConfig(
    max_epochs=30,
    batch_size=15, # effective batch size 60, was 64 in paper
    base_lr=3e-4, # WAS 1e-3 IN ORIGINAL PAPER
    max_lr=3e-3, # WAS 1e-2 IN ORIGINAL PAPER
    grad_accum=4,
)

TRAIN_ARGS_FULL = TrainConfig(
    max_epochs=20,
    batch_size=TRAIN_ARGS.batch_size,
    base_lr=TRAIN_ARGS.base_lr,
    max_lr=TRAIN_ARGS.max_lr,
    grad_accum=TRAIN_ARGS.grad_accum,
)

# We want to saturate the GPU, so increase the batch size for smaller models.
# Effective batch size stays the same by compensating grad_accum term.

TRAIN_ARGS_PER_XLSR_SIZE: Dict[str, TrainConfig] = {
    "wav2vec2-xls-r-300m": TrainConfig(
        max_epochs=TRAIN_ARGS.max_epochs,
        batch_size=30, # effective batch size 60 -> 24 GB VRAM
        base_lr=TRAIN_ARGS.base_lr,
        max_lr=TRAIN_ARGS.max_lr,
        grad_accum=2,
    ),
    "wav2vec2-xls-r-1b": TrainConfig(
        max_epochs=TRAIN_ARGS.max_epochs,
        batch_size=20, # effective batch size 60 -> 20 GB VRAM
        base_lr=TRAIN_ARGS.base_lr,
        max_lr=TRAIN_ARGS.max_lr,
        grad_accum=3,
    ),
    "wav2vec2-xls-r-2b": TrainConfig(
        max_epochs=TRAIN_ARGS.max_epochs,
        batch_size=20, # effective batch size 60 -> 22.5 GB VRAM
        base_lr=TRAIN_ARGS.base_lr,
        max_lr=TRAIN_ARGS.max_lr,
        grad_accum=3,
    ),
}


XLSR_300M_BLSTM_CONFIG = Config(
    "XLSR_300M_BLSTM_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=False,
    nlayers_transformer=2,
)

XLSR_1B_BLSTM_CONFIG = Config(
    "XLSR_1B_BLSTM_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=False,
    nlayers_transformer=2,
)

XLSR_2B_BLSTM_CONFIG = Config(
    "XLSR_2B_BLSTM_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=False,
    nlayers_transformer=2,
)

XLSR_300M_BLSTM_DP_CONFIG = Config(
    "XLSR_300M_BLSTM_DP_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nlayers_transformer=2,
)

XLSR_1B_BLSTM_DP_CONFIG = Config(
    "XLSR_1B_BLSTM_DP_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nlayers_transformer=2,
)

XLSR_2B_BLSTM_DP_CONFIG = Config(
    "XLSR_2B_BLSTM_DP_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nlayers_transformer=2,
)

XLSR_300M_BLSTM_DP_DEEP_CONFIG = Config(
    "XLSR_300M_BLSTM_DP_DEEP_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nlayers_transformer=4,
)

XLSR_1B_BLSTM_DP_DEEP_CONFIG = Config(
    "XLSR_1B_BLSTM_DP_DEEP_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nlayers_transformer=4,
)

XLSR_2B_BLSTM_DP_DEEP_CONFIG = Config(
    "XLSR_2B_BLSTM_DP_DEEP_CONFIG",
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nlayers_transformer=4,
)

XLSR_300M_TRANSFORMER_32_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_1B_TRANSFORMER_32_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_32_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_2B_TRANSFORMER_32_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_32_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_300M_TRANSFORMER_64_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_64_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_1B_TRANSFORMER_64_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_64_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_2B_TRANSFORMER_64_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_64_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_300M_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32DEEP_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_1B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_32DEEP_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_2B_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_32DEEP_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_300M_TRANSFORMER_64DEEP_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_64DEEP_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_1B_TRANSFORMER_64DEEP_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_64DEEP_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_2B_TRANSFORMER_64DEEP_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_64DEEP_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_300M_TRANSFORMER_32DEEPER_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32DEEPER_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=6,
)

XLSR_1B_TRANSFORMER_32DEEPER_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_32DEEPER_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=6,
)

XLSR_2B_TRANSFORMER_32DEEPER_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_32DEEPER_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=6,
)

XLSR_300M_TRANSFORMER_64DEEPER_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_64DEEPER_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=6,
)

XLSR_1B_TRANSFORMER_64DEEPER_CONFIG = Config(
    "XLSR_1B_TRANSFORMER_64DEEPER_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-1b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=6,
)

XLSR_2B_TRANSFORMER_64DEEPER_CONFIG = Config(
    "XLSR_2B_TRANSFORMER_64DEEPER_CONFIG",
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=6,
)



FEAT_SEQ_LEN=384
AUDIO_SEQ_LEN=122960 # 7.685 seconds => gives exactly 384 features (* see below)

# *: 122960-123279 give 384 features, but these give basically same results
# checked with:
#   import torch
#   from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
#   helper = Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.0,do_normalize=True,return_attention_mask=True)
#   model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m")
#   data = torch.randn((123279,))
#   input_rnd1 = helper(data[:122960], sampling_rate=16000, return_tensors="pt")["input_values"]
#   input_rnd2 = helper(data[:123279], sampling_rate=16000, return_tensors="pt")["input_values"]
#   out_rnd1 = model(input_rnd1).last_hidden_state.squeeze()
#   out_rnd2 = model(input_rnd2).last_hidden_state.squeeze()
#   torch.allclose(out_rnd1, out_rnd2, atol=0.005, rtol=0)


# Configs used by paper.
ALL_CONFIGS: List[Config] = [
    XLSR_300M_BLSTM_CONFIG, # without DP to match first paper
    XLSR_1B_BLSTM_CONFIG,
    XLSR_2B_BLSTM_CONFIG,
    XLSR_300M_BLSTM_DP_CONFIG, # with down-projection before BLSTM for most fair comparison
    XLSR_1B_BLSTM_DP_CONFIG,
    XLSR_2B_BLSTM_DP_CONFIG,
    XLSR_300M_BLSTM_DP_DEEP_CONFIG, # 4 layers
    XLSR_1B_BLSTM_DP_DEEP_CONFIG,
    XLSR_2B_BLSTM_DP_DEEP_CONFIG,
    XLSR_300M_TRANSFORMER_32_CONFIG, # h=32, 2 layers
    XLSR_1B_TRANSFORMER_32_CONFIG,
    XLSR_2B_TRANSFORMER_32_CONFIG,
    XLSR_300M_TRANSFORMER_64_CONFIG, # h=64, 2 layers
    XLSR_1B_TRANSFORMER_64_CONFIG,
    XLSR_2B_TRANSFORMER_64_CONFIG,
    XLSR_300M_TRANSFORMER_32DEEP_CONFIG, # h=32, 4 layers
    XLSR_1B_TRANSFORMER_32DEEP_CONFIG,
    XLSR_2B_TRANSFORMER_32DEEP_CONFIG,
    # XLSR_300M_TRANSFORMER_64DEEP_CONFIG, # h=64, 4 layers
    # XLSR_1B_TRANSFORMER_64DEEP_CONFIG,
    # XLSR_2B_TRANSFORMER_64DEEP_CONFIG,
    # XLSR_300M_TRANSFORMER_32DEEPER_CONFIG, # h=32, 6 layers
    # XLSR_1B_TRANSFORMER_32DEEPER_CONFIG, 
    # XLSR_2B_TRANSFORMER_32DEEPER_CONFIG,
    # XLSR_300M_TRANSFORMER_64DEEPER_CONFIG, # h=64, 6 layers
    # XLSR_1B_TRANSFORMER_64DEEPER_CONFIG,
    # XLSR_2B_TRANSFORMER_64DEEPER_CONFIG,
]

# XLS-R 300M configs
XLSR_300M_CONFIGS: List[Config] = [
    XLSR_300M_BLSTM_CONFIG,
    XLSR_300M_BLSTM_DP_CONFIG,
    XLSR_300M_BLSTM_DP_DEEP_CONFIG,
    XLSR_300M_TRANSFORMER_32_CONFIG,
    XLSR_300M_TRANSFORMER_64_CONFIG,
    XLSR_300M_TRANSFORMER_32DEEP_CONFIG,
    # XLSR_300M_TRANSFORMER_64DEEP_CONFIG, # not converging well with 35% data
    # XLSR_300M_TRANSFORMER_32DEEPER_CONFIG,
    # XLSR_300M_TRANSFORMER_64DEEPER_CONFIG,
]

# XLS-R 1B configs
XLSR_1B_CONFIGS: List[Config] = [
    XLSR_1B_BLSTM_CONFIG,
    XLSR_1B_BLSTM_DP_CONFIG,
    XLSR_1B_BLSTM_DP_DEEP_CONFIG,
    XLSR_1B_TRANSFORMER_32_CONFIG,
    XLSR_1B_TRANSFORMER_64_CONFIG,
    XLSR_1B_TRANSFORMER_32DEEP_CONFIG,
    # XLSR_1B_TRANSFORMER_64DEEP_CONFIG, # not converging well with 35% data
    # XLSR_1B_TRANSFORMER_32DEEPER_CONFIG,
    # XLSR_1B_TRANSFORMER_64DEEPER_CONFIG,
]

# XLS-R 2B configs
XLSR_2B_CONFIGS: List[Config] = [
    XLSR_2B_BLSTM_CONFIG,
    XLSR_2B_BLSTM_DP_CONFIG,
    XLSR_2B_BLSTM_DP_DEEP_CONFIG,
    XLSR_2B_TRANSFORMER_32_CONFIG,
    XLSR_2B_TRANSFORMER_64_CONFIG,
    XLSR_2B_TRANSFORMER_32DEEP_CONFIG, 
    # XLSR_2B_TRANSFORMER_64DEEP_CONFIG, # not converging well with 35% data
    # XLSR_2B_TRANSFORMER_32DEEPER_CONFIG,
    # XLSR_2B_TRANSFORMER_64DEEPER_CONFIG,
]

CONFIGS_PER_XLSR_SIZE: Dict[str, List[Config]] = {
    "wav2vec2-xls-r-300m": XLSR_300M_CONFIGS,
    "wav2vec2-xls-r-1b": XLSR_1B_CONFIGS,
    "wav2vec2-xls-r-2b": XLSR_2B_CONFIGS,
}
