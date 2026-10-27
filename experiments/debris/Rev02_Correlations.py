
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif

from src.generators.debris import DEBRIS


SEEDS = [11, 22, 33, 44, 55]
DIM = 10
RATIO_NOISE = 0.05
G = 2
CLUNUM = 10
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

BASE_PATH = Path(
    "results/debris/ExperimentRev02"
)


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
        columns=[f"{prefix}{i}" for i in range(data.shape[1])]
    )


def calculate_correlations(X, sensitive):
    """Calculate Pearson, Spearman, and mutual information."""
    X = to_dataframe(X, "feature_")
    sensitive = to_dataframe(sensitive, "sensitive_")

    results = []

    for sensitive_name in sensitive.columns:
        y = sensitive[sensitive_name]

        for feature_name in X.columns:
            x = X[feature_name]

            # Pearson and Spearman correlations
            try:
                pearson = pearsonr(x, y).statistic
            except (ValueError, TypeError):
                pearson = np.nan

            try:
                spearman = spearmanr(x, y).statistic
            except (ValueError, TypeError):
                spearman = np.nan

            # Mutual information; assumes a discrete sensitive attribute.
            try:
                mi = mutual_info_classif(
                    X[[feature_name]],
                    y,
                    random_state=0,
                )[0]
            except (ValueError, TypeError):
                mi = np.nan

            results.append({
                "feature": feature_name,
                "sensitive_attribute": sensitive_name,
                "pearson": pearson,
                "spearman": spearman,
                "mutual_information": mi,
            })

    return pd.DataFrame(results)


def plot_correlations(correlations, save_path, title):
    """Plot the three association measures without an in-plot legend."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    markers = ["^", "v", "<", ">", "s", "D"]
    colors = [
        "#9467bd", "#1f77b4", "#2ca02c",
        "#d95f02", "#1b9e77", "#6a51a3",
    ]

    metrics = [
        ("pearson", "Pearson correlation", (-1, 1)),
        ("spearman", "Spearman correlation", (-1, 1)),
        ("mutual_information", "Mutual information", None),
    ]

    for metric, ylabel, ylim in metrics:
        fig, ax = plt.subplots(figsize=(14, 10))

        features = correlations["feature"].drop_duplicates().tolist()
        x_positions = np.arange(len(features))

        for i, sensitive_name in enumerate(
            correlations["sensitive_attribute"].unique()
        ):
            subdf = (
                correlations[
                    correlations["sensitive_attribute"] == sensitive_name
                ]
                .set_index("feature")
                .reindex(features)
            )

            ax.plot(
                x_positions,
                subdf[metric].to_numpy(),
                linestyle="None",
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                markersize=12,
                label=sensitive_name,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

        if ylim is not None:
            ax.set_ylim(*ylim)

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

            # Generate the dataset
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

            correlations = calculate_correlations(X, sensitive)
            correlations["seed"] = seed
            correlations["distr_index"] = distr_index
            correlations["distr"] = str(distr)

            output_dir = (
                seed_path / f"distr_index={distr_index}_correlations"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            correlations.to_csv(
                output_dir / "correlations.csv",
                index=False,
            )

            plot_correlations(
                correlations,
                output_dir,
                title=f"Distribution {distr_index}, seed {seed}",
            )

            all_correlations.append(correlations)

    # Combined data for analysis across all seeds and distributions.
    combined = pd.concat(all_correlations, ignore_index=True)
    combined.to_csv(
        BASE_PATH / "all_correlations.csv",
        index=False,
    )

    # Mean and standard deviation across seeds.
    metrics = ["pearson", "spearman", "mutual_information"]
    summary = (
        combined.groupby(
            ["distr_index", "feature", "sensitive_attribute"]
        )[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(
        BASE_PATH / "correlation_summary.csv",
        index=False,
    )

    print(f"Finished. Results saved under: {BASE_PATH}")


if __name__ == "__main__":
    main()
