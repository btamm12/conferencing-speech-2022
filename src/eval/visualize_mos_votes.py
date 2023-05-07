import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_mos_votes():

    xlabels = [str(x) for x in range(1,11)] + ["11+"]
    pstn_counts = [46, 394, 1446, 3225, 2817, 46, 138, 268, 338, 181, 0]
    iub_counts = [0, 0, 0, 0, 5333, 0, 0, 0, 0, 0, 0]
    nisqa_counts = [0, 0, 6, 121, 1735, 17, 29, 84, 46, 24, 106]

    script_dir = Path(__file__).parent
    img_dir = script_dir.joinpath("img")

    y = {
        "PSTN": np.asarray(pstn_counts, dtype=np.int64),
        "IUB": np.asarray(iub_counts, dtype=np.int64),
        "NISQA": np.asarray(nisqa_counts, dtype=np.int64),
        }

    fig = plt.figure(figsize=(6,2.5))
    ax = fig.gca()
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    ax.set_prop_cycle('color', colors)

    bottom = np.zeros(11)
    for label, counts in y.items():
        ax.bar(xlabels, counts, width=0.5, label=label, bottom=bottom)
        bottom += counts

    plt.xlabel("Votes Per File")
    plt.ylabel("File Count")

    ax.legend()
    ax.set_title("MOS Vote Count Statistics")
            

    # plt.clf()
    # plt.imshow(heatmap.T, extent=extent, origin='lower')
    out_path = img_dir / "mos-vote-statistics.pdf"
    plt.savefig(out_path, bbox_inches='tight')

if __name__ == "__main__":
    visualize_mos_votes()