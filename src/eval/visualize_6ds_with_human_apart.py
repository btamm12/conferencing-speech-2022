import ast
import csv
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from src import constants
from src.constants import DatasetDir
from src.utils_4ds.split import Split
from src.utils_4ds.csv_info import STANDARDIZED_CSV_INFO


def visualize_6ds_with_human():

    ######## NOTES (done for IUB)

    # MOS in [1,2): 341
    # MOS in [2,3): 2103
    # MOS in [3,4): 2162
    # MOS in [4,5]: 727

    # MOS in [1.0,1.5): 13     (!)
    # MOS in [1.5,2.0): 328    (~)
    # MOS in [2.0,2.5): 1026
    # MOS in [2.5,3.0): 1077
    # MOS in [3.0,3.5): 1091
    # MOS in [3.5,4.0): 1071
    # MOS in [4.0,4.5): 655
    # MOS in [4.5,5.0): 72     (!)

    # Since number of samples in the (!) regions are low, the chance is large that
    # the KDE will omit these samples unless they are very well clustered.
    #
    # I expect that this is the case for DNSMOS, that the points y<2 are spread
    # across the x axis too much to fit a couple of Gaussians
    
    # Test DNSMOS:
    # MOS in [1.0, 1.75) and pred in [1.00,2.00): 1
    # MOS in [1.0, 1.75) and pred in [2.00,2.25): 3
    # MOS in [1.0, 1.75) and pred in [2.25,2.50): 21
    # MOS in [1.0, 1.75) and pred in [2.50,2.75): 37
    # MOS in [1.0, 1.75) and pred in [2.75,3.00): 16
    # MOS in [1.0, 1.75) and pred in [3.00,3.25): 1
    # MOS in [1.0, 1.75) and pred in [3.25,5.00): 0

    # Test XLS-R 1B:
    # MOS in [1.0, 1.75) and pred in [1.00,2.00): 25
    # MOS in [1.0, 1.75) and pred in [2.00,2.25): 23
    # MOS in [1.0, 1.75) and pred in [2.25,2.50): 16
    # MOS in [1.0, 1.75) and pred in [2.50,2.75): 13
    # MOS in [1.0, 1.75) and pred in [2.75,3.00): 0
    # MOS in [1.0, 1.75) and pred in [3.00,3.25): 2
    # MOS in [1.0, 1.75) and pred in [3.25,5.00): 0

    ######## HUMAN VOTES

    val_iub_ds: DatasetDir = constants.get_dataset(Split.VAL_IUB, example=False, use_35=False)

    col_mos = STANDARDIZED_CSV_INFO.col_mos
    col_acrs = STANDARDIZED_CSV_INFO.col_acrs

    # Load CSV.
    csv_data = []  # MOS, ACRs
    x_human = [] # GT
    y_human = [] # Prediction
    with open(val_iub_ds.csv_path, encoding="utf-8", mode="r") as in_csv:
        csv_reader = csv.reader(in_csv)
        for idx, in_row in enumerate(csv_reader):

            # Skip header row.
            if idx == 0:
                continue

            # Save MOS, ACRs
            mos = float(in_row[col_mos])
            acrs = ast.literal_eval(in_row[col_acrs])
            csv_data.append([mos, acrs])

            # Append to lists.
            for acr in acrs:
                x_human.append(mos)
                y_human.append(acr)

    ############ OTHER VOTES


    val_ds: DatasetDir = constants.get_dataset(Split.VAL, example=False, use_35=False)
    script_dir = Path(__file__).parent
    eval_input_dir = script_dir.joinpath("eval_input")
    img_dir = script_dir.joinpath("img")

    # Run combine_csvs.
    # combine_csvs()

    # Load some data.
    paths = [
        eval_input_dir.joinpath("6ds_val2nisqa","prediction_val_dnsmos.csv"),
        eval_input_dir.joinpath("6ds_val2nisqa","prediction_val_mfcc_dstrain_1.csv"),
        eval_input_dir.joinpath("6ds_val2nisqa","prediction_val_xlsr_1b_fusion_1_dstrain_1.csv"),
        eval_input_dir.joinpath("6ds_val2iub","prediction_val_dnsmos.csv"),
        eval_input_dir.joinpath("6ds_val2iub","prediction_val_mfcc_dstrain_1.csv"),
        eval_input_dir.joinpath("6ds_val2iub","prediction_val_xlsr_1b_fusion_1_dstrain_1.csv"),
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

    x_np_per_path.append(np.asarray(x_human, dtype=np.float32))
    y_np_per_path.append(np.asarray(y_human, dtype=np.float32))

    f1 = plt.figure(figsize=(5,2.5))            
    f2 = plt.figure(figsize=(6.7,2.5))
    gs1 = f1.add_gridspec(1, 3, hspace=0, wspace=0.1)
    gs2 = f2.add_gridspec(1, 4, hspace=0, wspace=0.1)
    (ax1, ax2, ax3) = gs1.subplots(sharex="col", sharey="row")
    (ax4, ax5, ax6, ax7) = gs2.subplots(sharex="col", sharey="row")

    all_ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    titles = ["DNSMOS", "MFCC", "XLS-R 1B Layer41"]
    titles += ["DNSMOS", "MFCC", "XLS-R 1B Layer41", "Human"]
    
    # heatmap, xedges, yedges = np.histogram2d(x_np, y_np, bins=60, range=[[1,5],[1,5]])

    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    f1.suptitle("NISQA", fontweight="bold", y=0.94)
    f2.suptitle("IUB", fontweight="bold", y=0.94)
    f1.supxlabel("MOS", y=0.06)
    f2.supxlabel("MOS", y=0.06)
    f1.supylabel("Prediction", x=0.04)
    f2.supylabel("Prediction", x=0.06)

    bw_adj = 0.5

    for idx in range(len(all_ax)):
        _ax: plt.Axes = all_ax[idx]
        sns.kdeplot(x=x_np_per_path[idx], y=y_np_per_path[idx], fill=True, legend=True, ax=_ax, bw_adjust=bw_adj)
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
    _basename = "mos-prediction-visualization-with-human-nisqa"
    _bw = f"bw_{bw_adj}"
    out_path = img_dir / f"{_basename}-{_bw}.pdf"
    f1.savefig(out_path, bbox_inches='tight')
    out_path = img_dir / f"{_basename}-{_bw}.png"
    f1.savefig(out_path, bbox_inches='tight')
    out_path = img_dir / f"{_basename}-{_bw}.svg"
    f1.savefig(out_path, bbox_inches='tight')
    _basename = "mos-prediction-visualization-with-human-iub"
    _bw = f"bw_{bw_adj}"
    out_path = img_dir / f"{_basename}-{_bw}.pdf"
    f2.savefig(out_path, bbox_inches='tight')
    out_path = img_dir / f"{_basename}-{_bw}.png"
    f2.savefig(out_path, bbox_inches='tight')
    out_path = img_dir / f"{_basename}-{_bw}.svg"
    f2.savefig(out_path, bbox_inches='tight')

if __name__ == "__main__":
    visualize_6ds_with_human()