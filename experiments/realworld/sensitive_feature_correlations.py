from src.utils.plot import plot_feature_sensitive_correlations
from src.utils.load_uci import load_diabetes, load_adult, load_bank, load_census, load_creditcard
import pandas as pd
from pathlib import Path
from experiments.realworld.sensitive_feature_correlations_plot import main_corr_legend_plot
dataset_map = {
    "Adult" : load_adult, 
    "Bank": load_bank, 
    "Census":load_census, 
    "Creditcard": load_creditcard,
    "Diabetes" : load_diabetes
    
}

def main_feat_corr(run=True):
    #run = True

    
    for dataset_name in ["Adult","Bank","Census", "Creditcard","Diabetes"]:
        print("running ", dataset_name)
        base_path=f"results/realworld/correlations/{dataset_name}/"
        Path(base_path).mkdir(parents=True,exist_ok=True)

        if run:
            X_w, X_wo, sens, y_true = dataset_map[dataset_name]()
            pearson, spearman, mi = plot_feature_sensitive_correlations(
                X_wo,
                sens,
                save_path=f"{base_path}", 
                title_inline= dataset_name
            )
            pearson.to_csv(f"{base_path}pearson.csv")
            spearman.to_csv(f"{base_path}spearman.csv")
            mi.to_csv(f"{base_path}mutual_information.csv")
        
        else: 
            replot_feature_sensitive_correlations(
                base_path,
                title_inline=dataset_name,
                show_ylabel=False,
                pearson_ylim=(-0.5, 0.7),
                spearman_ylim=(-1, 1),
                mi_ylim=None,
            )
    main_corr_legend_plot()

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils.plot import sens_attr_markers, sens_colors
def replot_feature_sensitive_correlations(
    base_path,
    title_inline=None,
    show_ylabel=False,
    pearson_ylim=(-1, 1),
    spearman_ylim=(-1, 1),
    mi_ylim=None,
    marker_offset=0.12,
):
    plt.rcParams.update({
            "font.size": 60,
            "axes.labelsize": 60,
            "axes.titlesize": 60,
            "legend.fontsize": 60,
            "xtick.labelsize": 40,
            "ytick.labelsize": 60
        })
    base_path = Path(base_path)

    pearson = pd.read_csv(
        base_path / "pearson.csv", index_col=0
    )
    spearman = pd.read_csv(
        base_path / "spearman.csv", index_col=0
    )
    mi = pd.read_csv(
        base_path / "mutual_information.csv", index_col=0
    )

    plots = [
        (pearson, "pearson.pdf", "Pearson correlation",
         pearson_ylim),
        (spearman, "spearman.pdf", "Spearman correlation",
         spearman_ylim),
        (mi, "mutual_information.pdf", "Mutual Information",
         mi_ylim if mi_ylim is not None
         else (0, mi.max().max() * 1.1)),
    ]

    for data, filename, ylabel, ylim in plots:
        fig, ax = plt.subplots(figsize=(14, 12))

        x = np.arange(len(data.index))
        n_sensitive = len(data.columns)
        offsets = (
            np.arange(n_sensitive) - (n_sensitive - 1) / 2
        ) * marker_offset
        for i,s in enumerate(data.columns):
            ax.plot(
                x + offsets[i],
                data[s].values,
                marker=sens_attr_markers[s],
                linestyle="None",  
                markersize=32,
                label=s,
                color=sens_colors[s],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            data.index, rotation=45, ha="right"
        )
        if show_ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
        ax.set_ylim(ylim)

        if ylabel != "Mutual Information":
            ax.axhline(0, linewidth=1)

        if title_inline:
            ax.text(
                0.02, 0.98, title_inline,
                transform=ax.transAxes,
                ha="left", va="top",
            )

        plt.tight_layout()
        fig.savefig(
            base_path / filename,
            bbox_inches=None,
        )
        plt.close(fig)

    return pearson, spearman, mi

if __name__ == "__main__": 
    main_feat_corr(run=False)