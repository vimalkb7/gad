import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

from src.utils import ammd


DATASETS = {
    "sbm": "Synthetic",
    "metrla": "Traffic (METR-LA)",
    "molene": "Temperature (Molene)",
}
XTICKS = {
    "sbm": [100, 300, 500],
    "metrla": [0, 2500, 5000],
    "molene": [200, 600, 1000],
}
YLIMS = {
    "molene": (0.0, 0.5),
}
MODELS = {
    "ga": {"label": "GAD (Ours)", "color": "k", "linestyle": "-", "marker": "o"},
    "vp": {"label": "VPD", "color": "r", "linestyle": "--", "marker": "s"},
    "ve": {"label": "VED", "color": "b", "linestyle": ":", "marker": "^"},
}


def plot_ammd():
    os.makedirs("figures", exist_ok=True)

    with plt.style.context(['science', 'ieee', 'no-latex']):
        matplotlib.rcParams.update({'font.size': 12, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))

        for ax, (d, title) in zip(axs, DATASETS.items()):
            ax.set_title(title)
            ax.set_xlabel("Steps")
            ax.grid(linestyle=':')

            if not os.path.exists(f"results/{d}_steps.npy"):
                ax.text(0.5, 0.5, "Results not available", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            steps = np.load(f"results/{d}_steps.npy")
            for m, style in MODELS.items():
                mean, _ = ammd(np.load(f"results/{d}_{m}.npy"))
                ax.plot(steps, mean, markersize=7, **style)

            ax.set_xticks(XTICKS[d])
            if d in YLIMS:
                ax.set_ylim(*YLIMS[d])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        axs[0].set_ylabel("aMMD")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))
        plt.tight_layout()

        plt.savefig("figures/fig_3.pdf", bbox_inches="tight")
        plt.savefig("figures/fig_3.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
