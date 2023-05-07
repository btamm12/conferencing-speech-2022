import csv
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from src import constants
from src.constants import DatasetDir
from src.utils_4ds.split import Split

from src.eval.combine_csvs import combine_csvs

def visualize_6ds():

    val_ds: DatasetDir = constants.get_dataset(Split.VAL, example=False, use_35=False)
    script_dir = Path(__file__).parent
    eval_input_dir = script_dir.joinpath("eval_input")
    img_dir = script_dir.joinpath("img")

    # Run combine_csvs.
    # combine_csvs()

    # Load some data.
    paths = [
        eval_input_dir.joinpath("6ds_val2iub","prediction_val_dnsmos.csv"),
        eval_input_dir.joinpath("6ds_val2iub","prediction_val_mfcc_dstrain_1.csv"),
        eval_input_dir.joinpath("6ds_val2iub","prediction_val_xlsr_1b_fusion_1_dstrain_1.csv"),
        eval_input_dir.joinpath("6ds_val2nisqa","prediction_val_dnsmos.csv"),
        eval_input_dir.joinpath("6ds_val2nisqa","prediction_val_mfcc_dstrain_1.csv"),
        eval_input_dir.joinpath("6ds_val2nisqa","prediction_val_xlsr_1b_fusion_1_dstrain_1.csv"),
    ]
    x_np_per_path = []
    y_np_per_path = []
    for path in paths:
        x = []
        y = []
        with open(path, mode="r") as f:
            csv_reader = csv.reader(f)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue # skip header
                x.append(float(row[0])) # GT
                y.append(float(row[1])) # Predicted
        x_np_per_path.append(np.asarray(x, dtype=np.float32))
        y_np_per_path.append(np.asarray(y, dtype=np.float32))
            
    fig = plt.figure(figsize=(12,2.5))
    (f1, f2) = fig.subfigures(1,2)
    f1: Figure = f1
    f2: Figure = f2
    gs1 = f1.add_gridspec(1, 3, hspace=0, wspace=0.1)
    gs2 = f2.add_gridspec(1, 3, hspace=0, wspace=0.1)
    (ax1, ax2, ax3) = gs1.subplots(sharex="col", sharey="row")
    (ax4, ax5, ax6) = gs2.subplots(sharex="col", sharey="row")

    all_ax = [ax1, ax2, ax3, ax4, ax5, ax6]
    titles = ["DNSMOS", "MFCC", "XLS-R 1B Layer41"] * 2
    
    # heatmap, xedges, yedges = np.histogram2d(x_np, y_np, bins=60, range=[[1,5],[1,5]])

    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    f1.suptitle("NISQA", fontweight="bold")
    f2.suptitle("IUB", fontweight="bold")
    f1.supxlabel("Prediction")
    f2.supxlabel("Prediction")
    f1.supylabel("MOS")
    f2.supylabel("MOS")

    for idx in range(len(all_ax)):
        _ax: plt.Axes = all_ax[idx]
        sns.kdeplot(x=x_np_per_path[idx], y=y_np_per_path[idx], fill=True, legend=True, ax=_ax)
        _ax.plot([1,5], [1,5], 'k--', linewidth=2, zorder=1)
        _ax.set_aspect('equal')
        _ax.set_xlim(1,5)
        _ax.set_ylim(1,5)
        _ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        _ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        _ax.set_title(titles[idx])

    # sns.kdeplot(x=x_np, y=y_np, fill=True, legend=True, ax=ax2)
    # ax2.plot([1,5], [1,5], 'k--', linewidth=2, zorder=1)
    # ax2.set_aspect('equal')
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax2.set_title("MFCC")
    # ax2.xaxis.set_label("Prediction")

    # sns.kdeplot(x=x_np, y=y_np, fill=True, legend=True, ax=ax3)
    # ax3.plot([1,5], [1,5], 'k--', linewidth=2, zorder=1)
    # ax3.set_aspect('equal')
    # ax3.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax3.set_title("XLS-R 1B Layer41")


    # f2.suptitle("NISQA")

    # sns.kdeplot(x=x_np, y=y_np, fill=True, legend=True, ax=ax4)
    # ax4.plot([1,5], [1,5], 'k--', linewidth=2, zorder=1)
    # ax4.set_aspect('equal')
    # ax4.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax4.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax4.yaxis.set_label("MOS")
    # ax4.set_title("DNSMOS")

    # sns.kdeplot(x=x_np, y=y_np, fill=True, legend=True, ax=ax5)
    # ax5.plot([1,5], [1,5], 'k--', linewidth=2, zorder=1)
    # ax5.set_aspect('equal')
    # ax5.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax5.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax5.xaxis.set_label("Prediction")
    # ax5.set_title("MFCC")

    # sns.kdeplot(x=x_np, y=y_np, fill=True, legend=True, ax=ax6)
    # ax6.plot([1,5], [1,5], 'k--', linewidth=2, zorder=1)
    # ax6.set_aspect('equal')
    # ax6.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax6.yaxis.set_major_locator(plt.MaxNLocator(5))
    # ax6.set_title("XLS-R 1B Layer41")

    # plt.clf()
    # plt.imshow(heatmap.T, extent=extent, origin='lower')
    out_path = img_dir / "mos-prediction-visualization.pdf"
    plt.savefig(out_path, bbox_inches='tight')

if __name__ == "__main__":
    visualize_6ds()