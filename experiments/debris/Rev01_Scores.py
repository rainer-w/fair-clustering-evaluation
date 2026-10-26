
from src.generators.debris import DEBRIS
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.search import search_dbscan_all, search_hdbscan_all, search_fairden_all, search_kmeans_all,search_fairlets_all,search_fairsc_all,search_backurs_all
from src.utils.plot import plot_line
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth

import pandas as pd

def select_best_candidates(
    candidates,
    weightings=((1.0, 0.0), (0.75, 0.25), (0.5, 0.5),
                (0.25, 0.75), (0.0, 1.0)),
    metrics=("disco", "dcsi", "balance", "minbal"),
    group_cols=("seed", "distr_index", "method"),
):
    model_candidates = candidates[
        candidates["method"] != "GroundTruth"
    ]
    groundtruth = candidates[
        candidates["method"] == "GroundTruth"
    ].copy()

    selected = []

    for _, group in model_candidates.groupby(
        list(group_cols), dropna=False
    ):
        # Individual metric objectives
        for metric in metrics:
            valid = group.dropna(subset=[metric])
            if valid.empty:
                continue

            idx = valid[metric].idxmax()
            row = valid.loc[idx].copy()
            row["criterion"] = metric
            row["selection_score"] = row[metric]
            selected.append(row)

        # Weighted balance/disco objectives
        valid = group.dropna(subset=["balance", "disco"])
        if not valid.empty:
            for a, b in weightings:
                scores = a * valid["balance"] + b * valid["disco"]
                idx = scores.idxmax()

                row = valid.loc[idx].copy()
                row["criterion"] = f"balance_{a:g}_disco_{b:g}"
                row["selection_score"] = scores.loc[idx]
                selected.append(row)

    selected_df = pd.DataFrame(selected)

    # Preserve ground-truth reference rows for the existing plots.
    # They are reference values, not candidates to optimize.
    if not groundtruth.empty:
        groundtruth["criterion"] = "GroundTruth"
        selected_df = pd.concat(
            [selected_df, groundtruth],
            ignore_index=True,
        )

    return selected_df.reset_index(drop=True)

def run_rev01(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n, distr_index):
    
    fairdegen = DEBRIS(
        dim = dim, 
        clunum=clunum,
        seed=seed,
        core_num= core_num, 
        ratio_noise = ratio_noise,
        g=g,
        distr = distr
	)
    fairdata = fairdegen.generate_data(n)
    fairdata.to_csv(f"{path}_{distr_index}_data.csv",index=False)
    features_wo = fairdegen.get_features_wo_sensitive()
    features_w = fairdegen.get_features_w_sensitive()

    sensitive = fairdegen.get_sensitive()
    y_true = fairdegen.get_groundtruth()

    y_unfair = fairdegen.get_unfair_groundtruth()
    k_unfair = len(np.unique(y_unfair[y_unfair != -1]))


    dbscan_df = search_dbscan_all(features_wo, y_true, sensitive)
    hdbscan_df = search_hdbscan_all(features_wo, y_true,sensitive)
    fairden_df = search_fairden_all(features_wo, y_true, sensitive, k_unfair=k_unfair)
    kmeans_df = search_kmeans_all(features_wo, y_true, sensitive, k_unfair=k_unfair)

    fairsc_df = search_fairsc_all(features_wo, y_true, sensitive, k_unfair=k_unfair)
    PQ = (1,2)
    if distr_index > 4: 
        PQ = (1,5)
    if distr_index >= 6: 
        PQ = (1,6)
    backurs_df = search_backurs_all(features_wo, y_true, sensitive,pq=PQ,k_unfair=k_unfair)
    stacked = pd.concat(
        [dbscan_df, hdbscan_df, fairden_df, kmeans_df, fairsc_df,backurs_df]
    )

   # stacked["score"] = stacked["disco"] + stacked["balance"]
    stacked["distr_index"] =  distr_index
    stacked["distr"] = str(distr)

    groundtruth_eval = pd.DataFrame([evaluate_groundtruth(fairdegen)])
    groundtruth_eval["distr_index"] = distr_index
    groundtruth_eval["distr"] = str(distr)
    stacked = pd.concat(
        [stacked, groundtruth_eval],
        ignore_index=True,
    )
    
    return stacked
def old_main_rev01(run=True,plot=True):

    
    SEEDS = [11,22,33,44,55]
    DIM = 10
    RATIO_NOISE = 0.05
    G = 2
    CLUNUM = 10
    CORE_NUM = [15] * CLUNUM
    base_path = "results/debris/ExperimentRev01/"
    if run:
        for seed in SEEDS: 
            path = f"{base_path}{seed}/"
            Path(path).mkdir(parents=True,exist_ok=True)
            all_dfs = []
        
            for i,distr in enumerate([
                [[0.5,0.5], [0.5,0.5]], 
                [[0.6,0.4], [0.4,0.6]], 
                [[0.7, 0.3], [0.3, 0.7]], 
                [[0.8,0.2], [0.2,0.8]], 
                [[1.0, 0.0], [0.0, 1.0]], 


                [[0.8,0.2], [0.8,0.2]], 
                [[0.75,0.25], [0.85,0.15]], 
                [[0.7, 0.3], [0.9, 0.1]], 
                [[0.65,0.35], [0.95,0.05]], 
                [[0.6, 0.4], [1.0, 0.0]],
            ]) :

                all_df = run_rev01(path,DIM,CLUNUM,seed,CORE_NUM,RATIO_NOISE,G,distr,n=1000, distr_index=i)
                all_dfs.append(all_df)
                all_df.to_csv(f"{path}distr_index={i}.csv", index=False)

            results = pd.concat(all_dfs, ignore_index=True)
            results.to_csv(f"{path}results.csv",index=False)

            #new_results = pd.concat(opt_dfs, ignore_index=True)
            #existing = pd.read_csv(f"{path}results.csv")
            #results = pd.concat([existing, new_results], ignore_index=True)
            #results.to_csv(f"{path}results.csv", index=False)

            # plot per seed
            if plot:
                plot_line(results=results, x="distr_index", path=path, categorical=True)
    avg_results, gt_results = aggregate_over_seeds(
        base_path, seeds = SEEDS, group_cols= ["distr_index", "method"]
        )

    x_split = 5
    avg_A = avg_results[avg_results["distr_index"] < x_split].copy()
    avg_B = avg_results[avg_results["distr_index"] >= x_split].copy()
    gt_results = gt_results.loc[:, ~gt_results.columns.duplicated()]
    gt_A = gt_results[gt_results["distr_index"] < x_split].copy() 
    gt_B = gt_results[gt_results["distr_index"] >= x_split].copy() 
    plot_line(avg_A, x="distr_index", tick_labels=["A0", "A1", "A2", "A3", "A4"], path=f"{base_path}A_", categorical=True, groundtruth_results=gt_A, include_std=True, xlabel="Subgroup Imbalance Level")
    plot_line(avg_B, x="distr_index",tick_labels=["B0", "B1", "B2", "B3", "B4"], path=f"{base_path}B_", categorical=True, groundtruth_results=gt_B, include_std=True, xlabel="Subgroup Imbalance Level")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def run_weighting_ablation(
    base_path,
    seeds,
    balance_weights=np.linspace(0, 1, 11),
    plot_metrics=("balance", "disco"),
):
    base_path = Path(base_path)
    selected_rows = []

    # Load the original candidate archive for every seed.
    for seed in seeds:
        candidate_path = base_path / str(seed) / "candidates"
        files = sorted(candidate_path.glob("distr_index=*.csv"))

        if not files:
            print(f"Warning: no candidates found for seed {seed}")
            continue

        candidates = pd.concat(
            [pd.read_csv(file) for file in files],
            ignore_index=True,
        )

        candidates["seed"] = seed
        candidates = candidates[
            candidates["method"] != "GroundTruth"
        ]

        required = {"method", "distr_index", "balance", "disco"}
        missing = required - set(candidates.columns)
        if missing:
            raise ValueError(
                f"Candidate files for seed {seed} lack: {sorted(missing)}"
            )

        for (distr_index, method), group in candidates.groupby(
            ["distr_index", "method"]
        ):
            valid = group.dropna(subset=["balance", "disco"]).copy()
            if valid.empty:
                continue

            # Normalize within this seed/distribution/method candidate set.
            for metric in ("balance", "disco"):
                min_val = valid[metric].min()
                max_val = valid[metric].max()

                if max_val > min_val:
                    valid[f"{metric}_norm"] = (
                        (valid[metric] - min_val) / (max_val - min_val)
                    )
                else:
                    # All candidates have the same value for this metric.
                    valid[f"{metric}_norm"] = 0.5

            for balance_weight in balance_weights:
                disco_weight = 1.0 - balance_weight

                score = (
                    balance_weight * valid["balance_norm"]
                    + disco_weight * valid["disco_norm"]
                )

                idx = score.idxmax()
                row = valid.loc[idx].copy()

                row["balance_weight"] = balance_weight
                row["disco_weight"] = disco_weight
                row["selection_score"] = score.loc[idx]
                selected_rows.append(row)

    if not selected_rows:
        raise ValueError("No candidate rows available for the ablation.")

    selected = pd.DataFrame(selected_rows)

    # Aggregate selected candidates across seeds.
    group_cols = ["distr_index", "method", "balance_weight"]

    available_metrics = [
        metric for metric in plot_metrics
        if metric in selected.columns
    ]

    if not available_metrics:
        raise ValueError(
            f"None of the requested metrics are available: {plot_metrics}"
        )

    grouped = selected.groupby(group_cols)[available_metrics]

    means = grouped.mean().reset_index()
    stds = grouped.std().reset_index()

    stds = stds.rename(
        columns={metric: f"{metric}_std" for metric in available_metrics}
    )

    summary = means.merge(stds, on=group_cols, how="left")

    # Save all selected ablation rows and the seed-level summary.
    output_path = base_path / "weighting_ablation"
    output_path.mkdir(parents=True, exist_ok=True)

    selected.to_csv(
        output_path / "selected_candidates.csv",
        index=False,
    )
    summary.to_csv(
        output_path / "summary.csv",
        index=False,
    )

    # Plot one figure per distribution and metric.
    for distr_index in sorted(summary["distr_index"].unique()):
        distr_df = summary[summary["distr_index"] == distr_index]

        for metric in available_metrics:
            fig, ax = plt.subplots(figsize=(12, 8))

            for method, subdf in distr_df.groupby("method"):
                subdf = subdf.sort_values("balance_weight")

                x = subdf["balance_weight"].to_numpy()
                y = subdf[metric].to_numpy()

                line, = ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    label=method,
                )

                std_col = f"{metric}_std"
                if std_col in subdf.columns:
                    std = subdf[std_col].fillna(0).to_numpy()
                    ax.fill_between(
                        x,
                        y - std,
                        y + std,
                        color=line.get_color(),
                        alpha=0.15,
                    )

            ax.set_xlabel("Balance weight")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"Weighting ablation — distribution {distr_index}")
            ax.set_xlim(0, 1)
            ax.set_xticks(balance_weights)
            ax.grid(True, alpha=0.25)
            ax.legend()

            fig.tight_layout()
            fig.savefig(
                output_path
                / f"distr_{distr_index}_{metric}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

    return selected, summary
def main_rev01(run=True, plot=True):
    SEEDS = [11, 22, 33, 44, 55]
    DIM = 10
    RATIO_NOISE = 0.05
    G = 2
    CLUNUM = 10
    CORE_NUM = [15] * CLUNUM

    base_path = Path("results/debris/ExperimentRev01/")

    distributions = [
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

    weightings = (
        (1.0, 0.0),
        (0.75, 0.25),
        (0.5, 0.5),
        (0.25, 0.75),
        (0.0, 1.0),
    )

    # Stage 1: Run searches and save every candidate.
    if run:
        for seed in SEEDS:
            seed_path = base_path / str(seed)
            candidate_path = seed_path / "candidates"
            candidate_path.mkdir(parents=True, exist_ok=True)

            for i, distr in enumerate(distributions):
                candidates = run_rev01(
                    str(seed_path) + "/",
                    DIM,
                    CLUNUM,
                    seed,
                    CORE_NUM,
                    RATIO_NOISE,
                    G,
                    distr,
                    n=1000,
                    distr_index=i,
                )

                candidates["seed"] = seed

                candidates.to_csv(
                    candidate_path / f"distr_index={i}.csv",
                    index=False,
                )

    # Stage 2: Load archived candidates and select rows.
    for seed in SEEDS:
        seed_path = base_path / str(seed)
        candidate_path = seed_path / "candidates"

        candidate_files = sorted(
            candidate_path.glob("distr_index=*.csv")
        )

        if not candidate_files:
            raise FileNotFoundError(
                f"No candidate files found in {candidate_path}. "
                "Run with run=True first."
            )

        candidates = pd.concat(
            [pd.read_csv(file) for file in candidate_files],
            ignore_index=True,
        )

        selected = select_best_candidates(
            candidates,
            weightings=weightings,
        )

        selected.to_csv(
            seed_path / "results.csv",
            index=False,
        )

        #if plot:
        #    plot_line(
        #        results=selected,
        #        x="distr_index",
        #        path=str(seed_path) + "/",
        #        categorical=True,
        #    )

    # Stage 3: Aggregate the selected results across seeds.
    avg_results, gt_results = aggregate_over_seeds(
        base_path,
        seeds=SEEDS,
        group_cols=["distr_index", "method", "criterion"],
        metrics=["disco", "dcsi", "balance", "minbal", "runtime"],
    )

    x_split = 5

    avg_A = avg_results[avg_results["distr_index"] < x_split].copy()
    avg_B = avg_results[avg_results["distr_index"] >= x_split].copy()

    gt_results = gt_results.loc[
        :, ~gt_results.columns.duplicated()
    ]
    gt_A = gt_results[gt_results["distr_index"] < x_split].copy()
    gt_B = gt_results[gt_results["distr_index"] >= x_split].copy()

    plot_line(
        avg_A,
        x="distr_index",
        tick_labels=["A0", "A1", "A2", "A3", "A4"],
        path=f"{base_path}/A_",
        categorical=True,
        groundtruth_results=gt_A,
        include_std=True,
        xlabel="Subgroup Imbalance Level",
        metrics=["disco", "dcsi", "balance", "minbal", "runtime"],
    )

    plot_line(
        avg_B,
        x="distr_index",
        tick_labels=["B0", "B1", "B2", "B3", "B4"],
        path=f"{base_path}/B_",
        categorical=True,
        groundtruth_results=gt_B,
        include_std=True,
        xlabel="Subgroup Imbalance Level",
        metrics=["disco", "dcsi", "balance", "minbal", "runtime"],
    )

    ablation_selected, ablation_summary = run_weighting_ablation(
        base_path=base_path,
        seeds=SEEDS,
        balance_weights=np.linspace(0, 1, 11),
        plot_metrics=("balance", "disco"),
    )
    print(ablation_summary.head())
if __name__ == "__main__":
    main_rev01()