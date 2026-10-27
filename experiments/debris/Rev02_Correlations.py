from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif

from src.generators.debris import DEBRIS


SEEDS = [11, 22, 33, 44, 55]
DIM = 5
RATIO_NOISE = 0.05
G = 2
CLUNUM = 5
CORE_NUM = [15] * CLUNUM
N = 1000

DISTRIBUTIONS = [
    [[0.5, 0.5], [0.5, 0.5]],
    [[0.6, 0.4], [0.4, 0.6]],
    [[0.7, 0.3], [0.3, 0.7]],
    [[0.8, 0.2], [0.2, 0.8]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.8, 0.2], [0.8, 0.2]],
    [[0.75, 0.25], [0.85, 0.15]],
    [[0.7, 0.3], [0.9, 0.1]],
    [[0.65, 0.35], [0.95, 0.05]],
    [[0.6, 0.4], [1.0, 0.0]],
]

BASE_PATH = Path("results/debris/ExperimentRev02")


def to_dataframe(data, prefix):
    """Convert an array or Series/DataFrame to a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True).copy()

    if isinstance(data, pd.Series):
        return data.to_frame().reset_index(drop=True)

    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    return pd.DataFrame(
        data,
        columns=[f"{prefix}{i}" for i in range(data.shape[1])],
    )


def calculate_correlations(X, sensitive, cluster_labels):
    """
    Calculate feature-sensitive associations separately within
    each ground-truth cluster.
    """
    X = to_dataframe(X, "feature_")
    sensitive = to_dataframe(sensitive, "sensitive_")
    cluster_labels = np.asarray(cluster_labels).ravel()

    if not (len(X) == len(sensitive) == len(cluster_labels)):
        raise ValueError(
            "X, sensitive, and cluster_labels must have equal lengths."
        )

    results = []

    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        X_cluster = X.loc[mask].reset_index(drop=True)
        sensitive_cluster = sensitive.loc[mask].reset_index(drop=True)

        n_cluster = len(X_cluster)

        # Correlations are not meaningful for extremely small clusters.
        if n_cluster < 3:
            continue

        for sensitive_name in sensitive_cluster.columns:
            y = sensitive_cluster[sensitive_name]

            for feature_name in X_cluster.columns:
                x = X_cluster[feature_name]

                try:
                    if x.nunique() < 2 or y.nunique() < 2:
                        pearson = np.nan
                    else:
                        pearson = pearsonr(x, y).statistic
                except (ValueError, TypeError):
                    pearson = np.nan

                try:
                    if x.nunique() < 2 or y.nunique() < 2:
                        spearman = np.nan
                    else:
                        spearman = spearmanr(x, y).statistic
                except (ValueError, TypeError):
                    spearman = np.nan

                try:
                    # MI requires a discrete target. Sensitive labels
                    # are treated as categorical values.
                    if y.nunique() < 2 or x.nunique() < 2:
                        mi = np.nan
                    else:
                        mi = mutual_info_classif(
                            x.to_numpy().reshape(-1, 1),
                            y.to_numpy(),
                            discrete_features=False,
                            random_state=0,
                        )[0]
                except (ValueError, TypeError):
                    mi = np.nan

                results.append({
                    "cluster_label": cluster,
                    "n_cluster": n_cluster,
                    "feature": feature_name,
                    "sensitive_attribute": sensitive_name,
                    "pearson": pearson,
                    "spearman": spearman,
                    "mutual_information": mi,
                })

    return pd.DataFrame(results)

def plot_correlations(correlations, save_path, title):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("pearson", "Pearson correlation", (-1, 1)),
        ("spearman", "Spearman correlation", (-1, 1)),
        ("mutual_information", "Mutual information", None),
    ]

    features = correlations["feature"].drop_duplicates().tolist()
    x_positions = np.arange(len(features))
    clusters = sorted(correlations["cluster_label"].unique())

    cmap = plt.get_cmap("tab20", max(len(clusters), 1))

    for metric, ylabel, ylim in metrics:
        fig, ax = plt.subplots(figsize=(14, 8))

        for i, cluster in enumerate(clusters):
            subdf = (
                correlations[correlations["cluster_label"] == cluster]
                .set_index("feature")
                .reindex(features)
            )

            ax.plot(
                x_positions,
                subdf[metric].to_numpy(),
                linestyle="None",
                marker="o",
                color=cmap(i),
                markersize=6,
                label=f"Cluster {cluster}",
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_xlabel("Feature")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: {ylabel}")
        ax.grid(axis="y", alpha=0.25)

        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.axhline(0, color="black", linewidth=0.8)

        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout()
        fig.savefig(save_path / f"{metric}.pdf", bbox_inches="tight")

        if handles:
            legend_fig = plt.figure(figsize=(10, 2))
            legend_fig.legend(
                handles,
                labels,
                loc="center",
                ncol=min(5, len(labels)),
                frameon=False,
            )
            legend_fig.savefig(
                save_path / "legend.pdf",
                bbox_inches="tight",
            )
            plt.close(legend_fig)

        plt.close(fig)

def plot_correlation_summary(summary, save_path):
    """
    Plot mean +/- standard deviation across seeds.

    X-axis: distribution index.
    Separate subplot: each ground-truth cluster.
    Separate series: each feature.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("pearson", "Pearson correlation", (-1, 1)),
        ("spearman", "Spearman correlation", (-1, 1)),
        ("mutual_information", "Mutual information", None),
    ]

    clusters = sorted(summary["cluster_label"].unique())
    features = sorted(summary["feature"].unique())
    cmap = plt.get_cmap("tab10", max(len(features), 1))

    for metric, ylabel, ylim in metrics:
        fig, axes = plt.subplots(
            nrows=len(clusters),
            ncols=1,
            figsize=(13, max(4, 3.2 * len(clusters))),
            sharex=True,
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, cluster in zip(axes, clusters):
            cluster_data = summary[
                summary["cluster_label"] == cluster
            ]

            for i, feature in enumerate(features):
                subdf = (
                    cluster_data[cluster_data["feature"] == feature]
                    .sort_values("distr_index")
                )

                if subdf.empty:
                    continue

                x = subdf["distr_index"].to_numpy()
                mean = subdf[(metric, "mean")].to_numpy()
                std = subdf[(metric, "std")].to_numpy()

                # Markers only: do not imply a continuous relationship.
                ax.errorbar(
                    x,
                    mean,
                    yerr=std,
                    fmt="o",
                    linestyle="None",
                    capsize=3,
                    markersize=5,
                    color=cmap(i),
                    label=feature,
                )

            ax.set_ylabel(ylabel)
            ax.set_title(f"Ground-truth cluster {cluster}")
            ax.grid(axis="y", alpha=0.25)

            if ylim is not None:
                ax.set_ylim(*ylim)
                ax.axhline(0, color="black", linewidth=0.8)

        axes[-1].set_xlabel("Distribution index")
        axes[-1].set_xticks(sorted(summary["distr_index"].unique()))

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=min(5, len(labels)),
                frameon=False,
                bbox_to_anchor=(0.5, 1.0),
            )

        fig.suptitle(
            f"{ylabel}: mean ± standard deviation across seeds",
            y=1.02,
        )
        fig.tight_layout()
        fig.savefig(
            save_path / f"summary_{metric}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    all_correlations = []

    for seed in SEEDS:
        seed_path = BASE_PATH / str(seed)
        seed_path.mkdir(parents=True, exist_ok=True)

        for distr_index, distr in enumerate(DISTRIBUTIONS):
            print(
                f"Generating seed={seed}, "
                f"distribution={distr_index}"
            )

            generator = DEBRIS(
                dim=DIM,
                clunum=CLUNUM,
                seed=seed,
                core_num=CORE_NUM,
                ratio_noise=RATIO_NOISE,
                g=G,
                distr=distr,
            )

            fairdata = generator.generate_data(N)

            fairdata.to_csv(
                seed_path / f"distr_index={distr_index}_data.csv",
                index=False,
            )

            X = to_dataframe(
                generator.get_features_wo_sensitive(),
                "feature_",
            )
            sensitive = to_dataframe(
                generator.get_sensitive(),
                "sensitive_",
            )

            # Use the actual ground-truth cluster assignment.
            cluster_labels = fairdata["groundtruth"].to_numpy()

            correlations = calculate_correlations(
                X,
                sensitive,
                cluster_labels,
            )

            correlations["seed"] = seed
            correlations["distr_index"] = distr_index
            correlations["distr"] = str(distr)

            output_dir = (
                seed_path
                / f"distr_index={distr_index}_cluster_correlations"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            correlations.to_csv(
                output_dir / "cluster_correlations.csv",
                index=False,
            )

            plot_correlations(
                correlations,
                output_dir,
                title=f"Distribution {distr_index}, seed {seed}",
            )

            all_correlations.append(correlations)

    if not all_correlations:
        print("No correlation results were generated.")
        return

    combined = pd.concat(all_correlations, ignore_index=True)

    combined.to_csv(
        BASE_PATH / "all_cluster_correlations.csv",
        index=False,
    )


    metrics = ["pearson", "spearman", "mutual_information"]

    # Aggregate over seeds for each distribution, cluster, and feature.
    summary = (
        combined.groupby(
            [
                "distr_index",
                "cluster_label",
                "feature",
                "sensitive_attribute",
            ]
        )[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(
        BASE_PATH / "cluster_correlation_summary.csv",
        index=False,
    )

    # Plot mean +/- standard deviation across seeds.
    plot_correlation_summary(
        summary,
        BASE_PATH / "summary_plots",
    )


    print(f"Finished. Results saved under: {BASE_PATH}")


if __name__ == "__main__":
    main()
