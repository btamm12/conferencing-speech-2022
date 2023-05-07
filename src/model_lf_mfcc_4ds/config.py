from enum import Enum
from typing import Dict, List

from src import constants

class Input(Enum):
    MFCC = 0
    XLSR = 1

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
    default_lr: float = None
    default_warmup_steps: int = None
    default_weight_decay: float = None

    def __init__(
        self,
        max_epochs: int,
        batch_size: int,
        default_lr: float,
        default_warmup_steps: int,
        default_weight_decay: float = 1e-2,
        grad_accum: int = 1
    ) -> None:
        assert max_epochs > 0
        assert batch_size > 0
        assert default_lr > 0
        assert default_warmup_steps >= 0
        assert default_weight_decay >= 0
        assert grad_accum > 0

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.default_lr = default_lr
        self.default_warmup_steps = default_warmup_steps
        self.default_weight_decay = default_weight_decay
        self.grad_accum = grad_accum

class Config():

    name: str = None
    input: Input = None
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
        input: Input,
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
        if input == Input.MFCC:
            xlsr_name = None
        if transformer == Transformer.BLSTM:
            msg = "Must specify dim_transformer."
            assert dim_transformer is not None, msg
            msg = "dim_transformer must be positive."
            assert dim_transformer > 0, msg
        msg = "feat_seq_len must be positive."
        assert feat_seq_len > 0, msg

        # Check XLS-R name
        if xlsr_name is not None:
            msg = f"xlsr_name must be in {constants.XLSR_NAMES}"
            assert xlsr_name in constants.XLSR_NAMES, msg

        # Save parameters.
        self.name = name
        self.input = input
        self.transformer = transformer
        self.head = head
        self.feat_seq_len = feat_seq_len
        self.dim_transformer = dim_transformer
        self.down_proj = down_proj
        self.xlsr_name = xlsr_name
        self.nhead_transformer = nhead_transformer
        self.nlayers_transformer = nlayers_transformer

        if xlsr_name is not None:
            # From XLS-R paper Table 2: Model architectures. 
            # +1 since we output B hidden layers and the final embedding
            if xlsr_name == "wav2vec2-xls-r-300m":
                _b = 24
            elif xlsr_name == "wav2vec2-xls-r-1b":
                _b = 48
            elif xlsr_name == "wav2vec2-xls-r-2b":
                _b = 48
            self.xlsr_layers = _b + 1
        else:
            self.xlsr_layers = None

        # Set model parameters.
        if xlsr_name is not None:
            # From XLS-R paper Table 2: Model architectures.
            if self.xlsr_name == "wav2vec2-xls-r-2b":
                _h = 1920
            elif self.xlsr_name == "wav2vec2-xls-r-1b": # 1280 apparently...
                _h = 1280
            else:
                _h = 1024
            self.dim_input = _h
        else:
            self.dim_input = 40 # MFCC

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

        self.dropout = 0.1 # NOTE: FIXED THIS AFTER BLIND SUBMISSION! (previously 0.0)

TRAIN_ARGS = TrainConfig(
    max_epochs=30,
    batch_size=15,
    grad_accum=4,
    default_lr=3e-3,
    default_warmup_steps=100,
    default_weight_decay=1e-2,
)


# We want to saturate the GPU, so increase the batch size for smaller models.
# Effective batch size stays the same by compensating grad_accum term.

TRAIN_ARGS_PER_XLSR_SIZE: Dict[str, TrainConfig] = {
    "wav2vec2-xls-r-300m": TrainConfig(
        max_epochs=TRAIN_ARGS.max_epochs,
        batch_size=10, # effective batch size 60 -> 24 GB VRAM
        grad_accum=6,
        default_lr=TRAIN_ARGS.default_lr,
        default_warmup_steps=TRAIN_ARGS.default_warmup_steps,
        default_weight_decay=TRAIN_ARGS.default_weight_decay,
    ),
    "wav2vec2-xls-r-1b": TrainConfig(
        max_epochs=TRAIN_ARGS.max_epochs,
        batch_size=6, # effective batch size 60 -> 24 GB VRAM
        grad_accum=10,
        default_lr=TRAIN_ARGS.default_lr,
        default_warmup_steps=TRAIN_ARGS.default_warmup_steps,
        default_weight_decay=TRAIN_ARGS.default_weight_decay,
    ),
    "wav2vec2-xls-r-2b": TrainConfig(
        max_epochs=TRAIN_ARGS.max_epochs,
        batch_size=6, # effective batch size 60 -> 24 GB VRAM
        grad_accum=10,
        default_lr=TRAIN_ARGS.default_lr,
        default_warmup_steps=TRAIN_ARGS.default_warmup_steps,
        default_weight_decay=TRAIN_ARGS.default_weight_decay,
    ),
}

TRAIN_ARGS_MFCC = TrainConfig(
        max_epochs=TRAIN_ARGS.max_epochs,
        batch_size=12,
        grad_accum=5,
        default_lr=TRAIN_ARGS.default_lr,
        default_warmup_steps=TRAIN_ARGS.default_warmup_steps,
        default_weight_decay=TRAIN_ARGS.default_weight_decay,
    )


############################# BLSTM_CONFIG ######################
MFCC_BLSTM_CONFIG = Config(
    "MFCC_BLSTM_CONFIG",
    Input.MFCC,
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=False,
    nlayers_transformer=2,
)

XLSR_300M_BLSTM_CONFIG = Config(
    "XLSR_300M_BLSTM_CONFIG",
    Input.XLSR,
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
    Input.XLSR,
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
    Input.XLSR,
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=False,
    nlayers_transformer=2,
)

########################## BLSTM_DP_CONFIG ################################
MFCC_BLSTM_DP_CONFIG = Config(
    "MFCC_BLSTM_DP_CONFIG",
    Input.MFCC,
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nlayers_transformer=2,
)

XLSR_300M_BLSTM_DP_CONFIG = Config(
    "XLSR_300M_BLSTM_DP_CONFIG",
    Input.XLSR,
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
    Input.XLSR,
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
    Input.XLSR,
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nlayers_transformer=2,
)

########################## BLSTM_DP_DEEP_CONFIG #########################
MFCC_BLSTM_DP_DEEP_CONFIG = Config(
    "MFCC_BLSTM_DP_DEEP_CONFIG",
    Input.MFCC,
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nlayers_transformer=4,
)

XLSR_300M_BLSTM_DP_DEEP_CONFIG = Config(
    "XLSR_300M_BLSTM_DP_DEEP_CONFIG",
    Input.XLSR,
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
    Input.XLSR,
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
    Input.XLSR,
    Transformer.BLSTM, # Use Bi-LSTM module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # 32 in each direction
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nlayers_transformer=4,
)

############################### TRANSFORMER_32_CONFIG ###################
MFCC_TRANSFORMER_32_CONFIG = Config(
    "MFCC_TRANSFORMER_32_CONFIG",
    Input.MFCC,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_300M_TRANSFORMER_32_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32_CONFIG",
    Input.XLSR,
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
    Input.XLSR,
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
    Input.XLSR,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

############################ TRANSFORMER_64_CONFIG ##############""
MFCC_TRANSFORMER_64_CONFIG = Config(
    "MFCC_TRANSFORMER_64_CONFIG",
    Input.MFCC,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

XLSR_300M_TRANSFORMER_64_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_64_CONFIG",
    Input.XLSR,
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
    Input.XLSR,
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
    Input.XLSR,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=2,
)

####################### TRANSFORMER_32DEEP_CONFIG ####################
MFCC_TRANSFORMER_32DEEP_CONFIG = Config(
    "MFCC_TRANSFORMER_32DEEP_CONFIG",
    Input.MFCC,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_300M_TRANSFORMER_32DEEP_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_32DEEP_CONFIG",
    Input.XLSR,
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
    Input.XLSR,
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
    Input.XLSR,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=32, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

########################## TRANSFORMER_64DEEP_CONFIG ##################
MFCC_TRANSFORMER_64DEEP_CONFIG = Config(
    "MFCC_TRANSFORMER_64DEEP_CONFIG",
    Input.MFCC,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-300m",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)

XLSR_300M_TRANSFORMER_64DEEP_CONFIG = Config(
    "XLSR_300M_TRANSFORMER_64DEEP_CONFIG",
    Input.XLSR,
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
    Input.XLSR,
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
    Input.XLSR,
    Transformer.TRANSFORMER, # Use Transformer module.
    Head.POOLATTFF, # PoolAttFF regressor head.
    feat_seq_len=384,
    dim_transformer=64, # hidden size
    xlsr_name="wav2vec2-xls-r-2b",
    down_proj=True,
    nhead_transformer=4,
    nlayers_transformer=4,
)


FEAT_SEQ_LEN=384

### DON'T USE! The model requires sharp cropping of XLSR features, not of the input
### audio. Otherwise performance sucks.
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


ALL_CONFIGS: List[Config] = [
    MFCC_BLSTM_CONFIG,
    XLSR_300M_BLSTM_CONFIG, # without DP to match first paper
    XLSR_1B_BLSTM_CONFIG,
    XLSR_2B_BLSTM_CONFIG,
    MFCC_BLSTM_DP_CONFIG,
    XLSR_300M_BLSTM_DP_CONFIG, # with down-projection before BLSTM for most fair comparison
    XLSR_1B_BLSTM_DP_CONFIG,
    XLSR_2B_BLSTM_DP_CONFIG,
    MFCC_BLSTM_DP_DEEP_CONFIG,
    XLSR_300M_BLSTM_DP_DEEP_CONFIG, # 4 layers
    XLSR_1B_BLSTM_DP_DEEP_CONFIG,
    XLSR_2B_BLSTM_DP_DEEP_CONFIG,
    MFCC_TRANSFORMER_32_CONFIG,
    XLSR_300M_TRANSFORMER_32_CONFIG, # h=32, 2 layers
    XLSR_1B_TRANSFORMER_32_CONFIG,
    XLSR_2B_TRANSFORMER_32_CONFIG,
    MFCC_TRANSFORMER_64_CONFIG,
    XLSR_300M_TRANSFORMER_64_CONFIG, # h=64, 2 layers
    XLSR_1B_TRANSFORMER_64_CONFIG,
    XLSR_2B_TRANSFORMER_64_CONFIG,
    MFCC_TRANSFORMER_32DEEP_CONFIG,
    XLSR_300M_TRANSFORMER_32DEEP_CONFIG, # h=32, 4 layers
    XLSR_1B_TRANSFORMER_32DEEP_CONFIG,
    XLSR_2B_TRANSFORMER_32DEEP_CONFIG,
    MFCC_TRANSFORMER_64DEEP_CONFIG,
    XLSR_300M_TRANSFORMER_64DEEP_CONFIG, # h=64, 4 layers
    XLSR_1B_TRANSFORMER_64DEEP_CONFIG,
    XLSR_2B_TRANSFORMER_64DEEP_CONFIG,
]

# MFCC configs
MFCC_CONFIGS: List[Config] = [
    # MFCC_BLSTM_CONFIG,
    MFCC_BLSTM_DP_CONFIG,
    # MFCC_BLSTM_DP_DEEP_CONFIG,
    # MFCC_TRANSFORMER_32_CONFIG,
    # MFCC_TRANSFORMER_64_CONFIG,
    MFCC_TRANSFORMER_32DEEP_CONFIG,
    # MFCC_TRANSFORMER_64DEEP_CONFIG, # not converging well with 35% data
]

# XLS-R 300M configs
XLSR_300M_CONFIGS: List[Config] = [
    # XLSR_300M_BLSTM_CONFIG,
    XLSR_300M_BLSTM_DP_CONFIG,
    # XLSR_300M_BLSTM_DP_DEEP_CONFIG,
    # XLSR_300M_TRANSFORMER_32_CONFIG,
    # XLSR_300M_TRANSFORMER_64_CONFIG,
    XLSR_300M_TRANSFORMER_32DEEP_CONFIG,
    # XLSR_300M_TRANSFORMER_64DEEP_CONFIG, # not converging well with 35% data
]

# XLS-R 1B configs
XLSR_1B_CONFIGS: List[Config] = [
    # XLSR_1B_BLSTM_CONFIG,
    XLSR_1B_BLSTM_DP_CONFIG,
    # XLSR_1B_BLSTM_DP_DEEP_CONFIG,
    # XLSR_1B_TRANSFORMER_32_CONFIG,
    # XLSR_1B_TRANSFORMER_64_CONFIG,
    XLSR_1B_TRANSFORMER_32DEEP_CONFIG,
    # XLSR_1B_TRANSFORMER_64DEEP_CONFIG, # not converging well with 35% data
]

# XLS-R 2B configs
XLSR_2B_CONFIGS: List[Config] = [
    # XLSR_2B_BLSTM_CONFIG,
    XLSR_2B_BLSTM_DP_CONFIG,
    # XLSR_2B_BLSTM_DP_DEEP_CONFIG,
    # XLSR_2B_TRANSFORMER_32_CONFIG,
    # XLSR_2B_TRANSFORMER_64_CONFIG,
    XLSR_2B_TRANSFORMER_32DEEP_CONFIG,
    # XLSR_2B_TRANSFORMER_64DEEP_CONFIG, # not converging well with 35% data
]

CONFIGS_PER_XLSR_SIZE: Dict[str, List[Config]] = {
    "wav2vec2-xls-r-300m": XLSR_300M_CONFIGS,
    "wav2vec2-xls-r-1b": XLSR_1B_CONFIGS,
    "wav2vec2-xls-r-2b": XLSR_2B_CONFIGS,
}
