from src.generators.debris import DEBRIS
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.search import (
    search_dbscan_all,
    search_hdbscan_all,
    search_fairden_all,
    search_kmeans_all,
    search_fairsc_all,
    search_backurs_all,
)
from src.utils.plot import plot_line
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth

def analyze_sensitive_group_proportions(
    base_path,
    seeds,
    strengths,
    g,
):
    base_path = Path(base_path)
    rows = []

    # Assumes sensitive labels are 0, ..., g-1.
    expected_values = list(range(g))

    for seed in seeds:
        seed_path = base_path / str(seed)

        for strength in strengths:
            # Matches the filename produced by run_Rev03().
            data_path = seed_path / f"_strength={strength}_data.csv"

            if not data_path.exists():
                print(f"Missing data file: {data_path}")
                continue

            data = pd.read_csv(data_path)

            if "sensitive_value" not in data.columns:
                raise ValueError(
                    f"'sensitive_value' not found in {data_path}"
                )

            sensitive = data["sensitive_value"]
            counts = sensitive.value_counts().reindex(
                expected_values, fill_value=0
            )

            n = len(data)

            for value in expected_values:
                count = int(counts.loc[value])

                rows.append({
                    "seed": seed,
                    "sens_correlation_strength": strength,
                    "sensitive_value": value,
                    "count": count,
                    "proportion": count / n if n > 0 else np.nan,
                    "n": n,
                })

    per_seed = pd.DataFrame(rows)

    if per_seed.empty:
        print("No data files found. Nothing to analyse.")
        return per_seed, pd.DataFrame()

    # Save the proportions for each individual dataset.
    per_seed.to_csv(
        base_path / "sensitive_group_proportions_by_seed.csv",
        index=False,
    )

    # Aggregate across seeds for each strength and group.
    summary = (
        per_seed.groupby(
            ["sens_correlation_strength", "sensitive_value"]
        )["proportion"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_proportion",
            "std": "std_proportion",
            "count": "n_seeds",
        })
    )

    summary.to_csv(
        base_path / "sensitive_group_proportions_summary.csv",
        index=False,
    )

    print("\nSensitive group proportions by seed:")
    print(per_seed.to_string(index=False))

    print("\nMean and standard deviation across seeds:")
    print(summary.to_string(index=False))

    return per_seed, summary


def run_Rev03(
    path,
    dim,
    clunum,
    seed,
    core_num,
    ratio_noise,
    g,
    n,
    sens_correlation_strength,
):
    fairdegen = DEBRIS(
        dim=dim,
        clunum=clunum,
        seed=seed,
        core_num=core_num,
        ratio_noise=ratio_noise,
        g=g,
        gap=0,
        sensitive_assignment="feature_correlated",
        sens_correlation_strength=sens_correlation_strength,
    )

    # Keep the data-generation seed fixed across strength values.
    fairdata = fairdegen.generate_data(n, seed=seed)

    fairdata.to_csv(
        f"{path}_strength={sens_correlation_strength}_data.csv",
        index=False,
    )

    features_wo = fairdegen.get_features_wo_sensitive()
    features_w = fairdegen.get_features_w_sensitive()

    sensitive = fairdegen.get_sensitive()
    y_true = fairdegen.get_groundtruth()

    y_unfair = fairdegen.get_unfair_groundtruth()
    k_unfair = len(np.unique(y_unfair[y_unfair != -1]))

    dbscan_df = search_dbscan_all(features_wo, y_true, sensitive)
    hdbscan_df = search_hdbscan_all(features_wo, y_true, sensitive)
    fairden_df = search_fairden_all(
        features_wo, y_true, sensitive, k_unfair=k_unfair
    )
    kmeans_df = search_kmeans_all(
        features_wo, y_true, sensitive, k_unfair=k_unfair
    )
    fairsc_df = search_fairsc_all(
        features_wo, y_true, sensitive, k_unfair=k_unfair
    )
    backurs_df = search_backurs_all(
        features_wo,
        y_true,
        sensitive,
        pq=(1, 2),
        k_unfair=k_unfair,
    )

    stacked = pd.concat(
        [
            dbscan_df,
            hdbscan_df,
            fairden_df,
            kmeans_df,
            fairsc_df,
            backurs_df,
        ],
        ignore_index=True,
    )

    stacked["score"] = stacked["disco"] + stacked["balance"]
    stacked["dim"] = dim
    stacked["n"] = n
    stacked["sens_correlation_strength"] = sens_correlation_strength

    rows = []

    for method in stacked["method"].unique():
        subdf = stacked[stacked["method"] == method]

        if subdf.empty:
            continue

        best_disco_row = subdf.loc[subdf["disco"].idxmax()].copy()
        best_disco_row["criterion"] = "disco"

        best_score_row = subdf.loc[subdf["score"].idxmax()].copy()
        best_score_row["criterion"] = "score"

        rows.append(best_disco_row.to_dict())
        rows.append(best_score_row.to_dict())

    groundtruth_eval = evaluate_groundtruth(fairdegen)
    groundtruth_eval["n"] = n
    groundtruth_eval["sens_correlation_strength"] = (
        sens_correlation_strength
    )
    rows.append(groundtruth_eval)

    return pd.DataFrame(rows), stacked


def main_Rev03(run=True, plot=False):

    SEEDS = [11, 22, 33, 44, 55]
    SENS_STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0]

    RATIO_NOISE = 0.05
    G = 2
    CLUNUM = 10
    CORE_NUM = [15] * CLUNUM
    DIM = 10

    N = 1000

    base_path = "results/debris/ExperimentRev03/"

    if run:
        for seed in SEEDS:
            path = f"{base_path}{seed}/"
            Path(path).mkdir(parents=True, exist_ok=True)

            opt_dfs = []

            for strength in SENS_STRENGTHS:
                opt_df, all_df = run_Rev03(
                    path=path,
                    dim=DIM,
                    clunum=CLUNUM,
                    seed=seed,
                    core_num=CORE_NUM,
                    ratio_noise=RATIO_NOISE,
                    g=G,
                    n=N,
                    sens_correlation_strength=strength,
                )

                opt_dfs.append(opt_df)

                all_df.to_csv(
                    f"{path}strength={strength}.csv",
                    index=False,
                )

            results = pd.concat(opt_dfs, ignore_index=True)
            results.to_csv(f"{path}results.csv", index=False)

            if plot:
                plot_line(
                    results=results,
                    x="sens_correlation_strength",
                    path=path,
                    categorical=True,
                )

    avg_results, gt_results = aggregate_over_seeds(
        base_path,
        seeds=SEEDS,
        group_cols=[
            "sens_correlation_strength",
            "method",
            "criterion",
        ],
    )

    if plot:
        plot_line(
            avg_results,
            x="sens_correlation_strength",
            tick_labels=SENS_STRENGTHS,
            path=base_path,
            categorical=False,
            groundtruth_results=gt_results,
            include_std=True,
        )

    
    proportions, proportions_summary = (
        analyze_sensitive_group_proportions(
            base_path=base_path,
            seeds=SEEDS,
            strengths=SENS_STRENGTHS,
            g=G,
        )
    )



if __name__ == "__main__":
    main_Rev03(run=False, plot=True)
