# -*- coding: utf-8 -*-
"""
yimingxiao
"""
import numpy as np
import os
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error


def _eval(csv):
    df = pd.read_csv(csv)
    mos = df["mos"]
    mos_pred = df["mos_pred"]
    pccs = np.corrcoef(mos, mos_pred)[0][1]
    rmse = np.sqrt(mean_squared_error(mos, mos_pred))
    SROCC = stats.spearmanr(mos_pred, mos)[0]
    _name = os.path.splitext(os.path.basename(csv))[0][15:]
    print(_name + "," + str(round(rmse, 4)))
    # print("PCC:  " + str(round(pccs,4)))
    # print("SRCC: " + str(round(SROCC,4)))
    # print("RMSE: " + str(round(rmse,4)))


def eval():
    from pathlib import Path

    script_dir = Path(__file__).parent
    in_dir = script_dir.joinpath("eval_input")

    def f(path: Path):
        stem = path.stem
        x = 0
        if "300m" in stem:
            x = 1
        if "1b" in stem:
            x = 2
        if "2b" in stem:
            x = 3

        # First sort by dstrain
        return str(x) + stem[-1] + stem[:-1]

    for subdir in in_dir.iterdir():
        print(f"Processing {subdir}")
        paths = list(subdir.iterdir())
        for csv_path in sorted(paths, key=f):
            # print(f"Processing {csv_path}")
            _eval(str(csv_path))
            # print("============================")
        print("============================")


if __name__ == "__main__":
    eval()
