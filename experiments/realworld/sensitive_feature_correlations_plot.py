import pandas as pd 
"""
from utils.plot import plot_filtered_skyline
for name in ["Adult","Bank", "Census", "Creditcard", "Diabetes"]:
    df = pd.read_csv(f"results/{name}/opt_results.csv")
    score_df = df[df["criterion"] == "score"].copy()
    plot_filtered_skyline(score_df, x="n_clusters", path=f"results/{name}/optscore_n_clusters-")
    plot_filtered_skyline(score_df, x="disco", path=f"results/{name}/optscore_disco-")
    disco_df = df[df["criterion"] == "disco"].copy()
    plot_filtered_skyline(disco_df, x="n_clusters", path=f"results/{name}/optdisco_n_clusters-")
    plot_filtered_skyline(disco_df, x="disco", path=f"results/{name}/optdisco_disco-")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=sens_colors["sex"], marker="v", linewidth=3, label="Sex"),
        Line2D([], [], color=sens_colors["race"], marker="^", linewidth=3, label="Race"),
        Line2D([], [], color=sens_colors["marital"], marker="<", linewidth=3, label="Marital status"),
        Line2D([], [], color=sens_colors["age"], marker=">", linewidth=3, label="Age"),
        ]
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.utils.plot import plot_filtered_skyline

def main_plot_uci():
    for name in ["Adult","Bank", "Census", "Creditcard", "Diabetes"]:
        df = pd.read_csv(f"results/realworld/{name}/opt_results.csv")
        score_df = df[df["criterion"] == "score"].copy()
        plot_filtered_skyline(score_df, x="n_clusters", path=f"results/{name}/optscore_n_clusters-")
        plot_filtered_skyline(score_df, x="disco", path=f"results/{name}/optscore_disco-")
        disco_df = df[df["criterion"] == "disco"].copy()
        plot_filtered_skyline(disco_df, x="n_clusters", path=f"results/{name}/optdisco_n_clusters-")
        plot_filtered_skyline(disco_df, x="disco", path=f"results/{name}/optdisco_disco-")
def main_corr_legend_plot():

    sens_colors = {
        "sex": "#1f77b4",       # blue
        "race": "#d62728",      # red
        "marital": "#2ca02c",   # green
        "age": "#9467bd",       # purple
        }
    handles = [
        Line2D([], [], color=sens_colors["sex"], marker="v", linewidth=3, label="Sex"),
        Line2D([], [], color=sens_colors["race"], marker="^", linewidth=3, label="Race"),
        Line2D([], [], color=sens_colors["marital"], marker="<", linewidth=3, label="Marital status"),
        Line2D([], [], color=sens_colors["age"], marker=">", linewidth=3, label="Age"),
        ]

    fig = plt.figure(figsize=(4, 2))
    fig.legend(handles=handles,
            loc="center",
            ncol=4,
            frameon=False)

    fig.savefig(f"results/realworld/correlations/legend_all_sensitive.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)
if __name__=="__main__":
    main_corr_legend_plot()