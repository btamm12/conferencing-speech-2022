from torch import Tensor, nn
from transformers import AutoModel, Wav2Vec2Model

from src import constants


class Wav2Vec2Wrapper(nn.Module):

    def __init__(self, xlsr_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.xlsr_name = xlsr_name

        # Create model.
        if constants.XLSR_DIRS[xlsr_name].exists():
            _model = AutoModel.from_pretrained(str(constants.XLSR_DIRS[xlsr_name]))
        else:
            _model = Wav2Vec2Model.from_pretrained(f"facebook/{xlsr_name}")
        self.model = _model

    def forward(self, input: Tensor):
        # Model.
        if input.dim() > 2:
            input = input.squeeze(0)
        return self.model.forward(input, output_hidden_states=True)
