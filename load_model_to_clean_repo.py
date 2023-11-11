import torch

from xls_r_sqa.config import (
    Config,
    MFCC_TRANSFORMER_32DEEP_CONFIG,
    XLSR_300M_TRANSFORMER_32DEEP_CONFIG,
    XLSR_1B_TRANSFORMER_32DEEP_CONFIG,
    XLSR_2B_TRANSFORMER_32DEEP_CONFIG,
)
from xls_r_sqa.sqa_model import SingleLayerModel, FusionModel


def load_model(in_path: str, config: Config, layer_idx: int):
    _state_dict_src = torch.load(in_path)
    _state_dict_src["norm_input.weight"] = _state_dict_src[f"norm_inputs.{layer_idx}.weight"]
    _state_dict_src["norm_input.bias"] = _state_dict_src[f"norm_inputs.{layer_idx}.bias"]
    _state_dict_src["norm_input.running_mean"] = _state_dict_src[f"norm_inputs.{layer_idx}.running_mean"]
    _state_dict_src["norm_input.running_var"] = _state_dict_src[f"norm_inputs.{layer_idx}.running_var"]
    _state_dict_src["norm_input.num_batches_tracked"] = _state_dict_src[f"norm_inputs.{layer_idx}.num_batches_tracked"]
    del _state_dict_src["last_loss"]
    if "mfcc" in config.name.lower():
        _num_inputs = 1
    else:
        _num_inputs = 2
    for i in range(_num_inputs):
        del _state_dict_src[f"norm_inputs.{i}.weight"]
        del _state_dict_src[f"norm_inputs.{i}.bias"]
        del _state_dict_src[f"norm_inputs.{i}.running_mean"]
        del _state_dict_src[f"norm_inputs.{i}.running_var"]
        del _state_dict_src[f"norm_inputs.{i}.num_batches_tracked"]
    _model = SingleLayerModel(config)
    _model.load_state_dict(_state_dict_src)
    return _model

def load_fusion_model(in_path, config):
    _state_dict_src = torch.load(in_path)
    del _state_dict_src["last_loss"]
    _model = FusionModel(config)
    _model.load_state_dict(_state_dict_src)
    return _model


if __name__ == "__main__":
    model_dir = "/home/luna.kuleuven.be/u0131128/GitHub/lcn-kul/xls-r-analysis-sqa/models/"

    # MFCC
    print("Converting MFCC models.")
    _config = MFCC_TRANSFORMER_32DEEP_CONFIG
    # src_dir = model_dir + "original/mfcc/"
    # dst_dir = model_dir + "sqa/mfcc/"
    src_dir = model_dir + "original-v2/mfcc/"
    dst_dir = model_dir + "sqa-v2/mfcc/"
    for _name in ["model_mfcc_full.pt", "model_mfcc_subset.pt"]:
        _model = load_model(src_dir + _name, _config, 0)
        torch.save(_model.state_dict(), dst_dir + _name)

    # XLS-R 300M
    print("Converting XLS-R 300M models.")
    _config = XLSR_300M_TRANSFORMER_32DEEP_CONFIG
    # src_dir = model_dir + "original/xls-r-300m/"
    # dst_dir = model_dir + "sqa/xls-r-300m/"
    src_dir = model_dir + "original-v2/xls-r-300m/"
    dst_dir = model_dir + "sqa-v2/xls-r-300m/"
    for _name in ["model_300m_lay5_full.pt", "model_300m_lay5_subset.pt"]:
        _model = load_model(src_dir + _name, _config, 0)
        torch.save(_model.state_dict(), dst_dir + _name)
    for _name in ["model_300m_lay21_full.pt", "model_300m_lay21_subset.pt"]:
        _model = load_model(src_dir + _name, _config, 1)
        torch.save(_model.state_dict(), dst_dir + _name)
    for _name in ["model_300m_fusion_full.pt", "model_300m_fusion_subset.pt"]:
        _model = load_fusion_model(src_dir + _name, _config)
        torch.save(_model.state_dict(), dst_dir + _name)

    # XLS-R 1B
    print("Converting XLS-R 1B models.")
    _config = XLSR_1B_TRANSFORMER_32DEEP_CONFIG
    # src_dir = model_dir + "original/xls-r-1b/"
    # dst_dir = model_dir + "sqa/xls-r-1b/"
    src_dir = model_dir + "original-v2/xls-r-1b/"
    dst_dir = model_dir + "sqa-v2/xls-r-1b/"
    for _name in ["model_1b_lay10_full.pt", "model_1b_lay10_subset.pt"]:
        _model = load_model(src_dir + _name, _config, 0)
        torch.save(_model.state_dict(), dst_dir + _name)
    for _name in ["model_1b_lay41_full.pt", "model_1b_lay41_subset.pt"]:
        _model = load_model(src_dir + _name, _config, 1)
        torch.save(_model.state_dict(), dst_dir + _name)
    for _name in ["model_1b_fusion_full.pt", "model_1b_fusion_subset.pt"]:
        _model = load_fusion_model(src_dir + _name, _config)
        torch.save(_model.state_dict(), dst_dir + _name)

    # XLS-R 2B
    print("Converting XLS-R 2B models.")
    _config = XLSR_2B_TRANSFORMER_32DEEP_CONFIG
    # src_dir = model_dir + "original/xls-r-2b/"
    # dst_dir = model_dir + "sqa/xls-r-2b/"
    src_dir = model_dir + "original-v2/xls-r-2b/"
    dst_dir = model_dir + "sqa-v2/xls-r-2b/"
    for _name in ["model_2b_lay10_full.pt", "model_2b_lay10_subset.pt"]:
        _model = load_model(src_dir + _name, _config, 0)
        torch.save(_model.state_dict(), dst_dir + _name)
    for _name in ["model_2b_lay41_full.pt", "model_2b_lay41_subset.pt"]:
        _model = load_model(src_dir + _name, _config, 1)
        torch.save(_model.state_dict(), dst_dir + _name)
    for _name in ["model_2b_fusion_full.pt", "model_2b_fusion_subset.pt"]:
        _model = load_fusion_model(src_dir + _name, _config)
        torch.save(_model.state_dict(), dst_dir + _name)


