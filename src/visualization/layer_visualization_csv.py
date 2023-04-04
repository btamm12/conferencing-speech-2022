import csv
from glob import glob

from src.model.config import XLSR_300M_CONFIGS

def layer_visualization_csv(csv_glob: str, ds: str = "both", cfg_idx: int = -1):

    if ds not in ["full", "subset", "both"]:
        raise Exception("ds must be in [full, subset, both]")
    N = len(XLSR_300M_CONFIGS)
    if cfg_idx != -1 and cfg_idx not in range(N):
        raise Exception(f"cfg_idx must be either -1 (minimum over all configs) or in range({N})")


    # cfg_idx == -1 --> select minimum over all configs
    # ds_idx == -1 --> select both datasets

    # 0: epoch, 1: xlsr, 2: split, 3: ds, 4: cfg_idx, 5: layer_idx, 6: loss
    #    21,wav2vec2-xls-r-1b,train,subset,5,00,0.01284766
    expected_rows = 7
    expected_xlsr = None
    kept_rows_full = []
    kept_rows_subset = []
    min_losses_full = []
    min_losses_subset = []
    for file_path in sorted(glob(csv_glob)):
        with open(file_path, mode="r") as f:
            csv_reader = csv.reader(f)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                if len(row) != expected_rows:
                    raise Exception("Expected")
                _epoch, _xlsr, _split, _ds, _cfg_idx, _layer_idx, _loss = row
                _cfg_idx = int(_cfg_idx)
                _layer_idx = int(_layer_idx)
                _loss_f = float(_loss)
                if expected_xlsr is None:
                    expected_xlsr = _xlsr
                if _xlsr != expected_xlsr:
                    raise Exception(f"expected {expected_xlsr} but found {_xlsr}")
                if _split == "train":
                    continue
                if cfg_idx != -1 and _cfg_idx != cfg_idx:
                    continue

                if _ds == "full":
                    if _layer_idx >= len(min_losses_full):
                        min_losses_full.append(_loss_f)
                        kept_rows_full.append(row)
                    elif _loss_f < min_losses_full[_layer_idx]:
                        min_losses_full[_layer_idx] = _loss_f
                        kept_rows_full[_layer_idx] = row
                elif _ds == "subset":
                    if _layer_idx >= len(min_losses_subset):
                        min_losses_subset.append(_loss_f)
                        kept_rows_subset.append(row)
                    elif _loss_f < min_losses_subset[_layer_idx]:
                        min_losses_subset[_layer_idx] = _loss_f
                        kept_rows_subset[_layer_idx] = row
                else:
                    raise Exception(f"unknown ds in row {idx}: {_ds}")

    if ds == "full" or ds == "both":
        print("Full:")
        for row in kept_rows_full:
            print(",".join(row))

    if ds == "subset" or ds == "both":
        print("Subset:")
        for row in kept_rows_subset:
            print(",".join(row))
