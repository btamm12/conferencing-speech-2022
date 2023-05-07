from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_layerwise_performance_fullds():

    xlabels = [str(x) for x in range(1,11)] + ["11+"]
    # values extracted from model loss.csv files (minimum per layer over all configs)
    # same values as Excel file 20221223_analyse_xlsr3.ods
    norm_mse_300m = [0.01363038, 0.013236435, 0.01284249, 0.012633995, 0.0124255, 0.01233477, 0.01224404, 0.012094505, 0.01194497, 0.01186892, 0.01179287, 0.011817835, 0.0118428, 0.01185229, 0.01186178, 0.011879015, 0.01189625, 0.011905105, 0.01191396, 0.01196263, 0.0120113, 0.012111445, 0.01221159, 0.012189335, 0.01216708, 0.01218769, 0.0122083, 0.01224594, 0.01228358, 0.012260755, 0.01223793, 0.01219625, 0.01215457, 0.012181415, 0.01220826, 0.012094105, 0.01197995, 0.01191462, 0.01184929, 0.011892655, 0.01193602, 0.011891235, 0.01184645, 0.01196279, 0.01207913, 0.012171745, 0.01226436, 0.01220878, 0.0121532]
    norm_mse_1b = [0.01436419, 0.01358723, 0.01299707, 0.01254081, 0.01223654, 0.01204852, 0.01178984, 0.01189447, 0.01167003, 0.01198425, 0.01178367, 0.011897, 0.01178787, 0.01181826, 0.01194799, 0.01175953, 0.01187683, 0.01207674, 0.01203354, 0.01207454, 0.01210842, 0.01213667, 0.01218312, 0.01202073, 0.01215985, 0.01219738, 0.01230866, 0.01223784, 0.0122415, 0.01222612, 0.01231638, 0.01225519, 0.01236598, 0.01221591, 0.01227658, 0.0121562, 0.01217671, 0.01215404, 0.01209086, 0.01216233, 0.01203557, 0.01189608, 0.01203343, 0.01212918, 0.01214409, 0.01231455, 0.01226624, 0.01219587, 0.01239537]
    norm_mse_2b = [0.01439209, 0.0137682, 0.01287261, 0.01223712, 0.01194977, 0.0119197, 0.01172592, 0.01178758, 0.01179174, 0.01167661, 0.01167119, 0.01172948, 0.01177421, 0.01174688, 0.01174533, 0.01184971, 0.01193885, 0.01196523, 0.01191573, 0.01195468, 0.01206718, 0.0121283, 0.01200855, 0.01209577, 0.01229775, 0.01222388, 0.0122015, 0.01218695, 0.01222879, 0.01223067, 0.01221188, 0.01228722, 0.01215586, 0.0121026, 0.01214232, 0.01205219, 0.01201338, 0.01193258, 0.01197667, 0.01201046, 0.01193849, 0.01205107, 0.01193352, 0.0121133, 0.01210821, 0.0121843, 0.01219403, 0.01218038, 0.0123498]

    xlabels = list(range(49))
    rmse_300m = [4.*sqrt(x) for x in norm_mse_300m]
    rmse_1b = [4.*sqrt(x) for x in norm_mse_1b]
    rmse_2b = [4.*sqrt(x) for x in norm_mse_2b]

    script_dir = Path(__file__).parent
    img_dir = script_dir.joinpath("img")

    y = {
        "XLS-R 300M\n(interpolated)": np.asarray(rmse_300m, dtype=np.float32),
        "XLS-R 1B": np.asarray(rmse_1b, dtype=np.float32),
        "XLS-R 2B": np.asarray(rmse_2b, dtype=np.float32),
        }

    fig = plt.figure(figsize=(6,3))
    ax = fig.gca()
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    ax.set_prop_cycle('color', colors)

    for label, data in y.items():
        ax.plot(xlabels, data, label=label, alpha=0.8, linewidth=3)

    plt.xlabel("Layer")
    plt.ylabel("RMSE")
    plt.xlim([0,48])
    plt.ylim([0.425,0.485])

    ax.legend()
    ax.set_title("XLS-R Layer-wise Performance")
            

    # plt.clf()
    # plt.imshow(heatmap.T, extent=extent, origin='lower')
    out_path = img_dir / "xlsr-layerwise-performance-fullds.pdf"
    plt.savefig(out_path, bbox_inches='tight')

if __name__ == "__main__":
    visualize_layerwise_performance_fullds()