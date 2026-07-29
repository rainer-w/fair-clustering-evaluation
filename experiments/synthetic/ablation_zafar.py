
from src.generators.debris import DEBRIS

from src.generators.zafar_generator import ZafarGenerator

from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.search import search_dbscan_all, search_hdbscan_all, search_fairden_all, search_kmeans_all,search_fairsc_all,search_backurs_all
from src.utils.plot import plot_line
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth
def run_abl_zafar(path,seed,n, phi,i):
    
    fairdegen = ZafarGenerator([[2,2], [-2,-2]], [[[5, 1],[1, 5]],[[10, 1],[1, 3]]], phi)
    fairdata = fairdegen.generate_data(n, random_state=seed)
    fairdata.to_csv(f"{path}_{i}_data.csv",index=False)
    features_wo = fairdegen.get_features_wo_sensitive()
    features_w = fairdegen.get_features_w_sensitive()

    sensitive = fairdegen.get_sensitive()
    y_true = fairdegen.get_groundtruth()

    y_unfair = fairdegen.get_unfair_groundtruth()
    k_unfair = len(np.unique(y_unfair[y_unfair != -1]))
    
    dbscan_df = search_dbscan_all(features_w, y_true, sensitive)
    hdbscan_df = search_hdbscan_all(features_w, y_true,sensitive)
    fairden_df = search_fairden_all(features_wo, y_true, sensitive, k_unfair=k_unfair)
    kmeans_df = search_kmeans_all(features_w, y_true, sensitive, k_unfair=k_unfair)

    fairsc_df = search_fairsc_all(features_w, y_true, sensitive, k_unfair=k_unfair)
    backurs_df = search_backurs_all(features_wo, y_true, sensitive,pq=(1,2),k_unfair=k_unfair)
    stacked = pd.concat(
        [dbscan_df, hdbscan_df, fairden_df, kmeans_df, 
         fairsc_df, backurs_df]
    )
    
    stacked["score"] = stacked["disco"] + stacked["balance"]
    stacked["n"] = n
    stacked["phi"] = phi
    stacked["i"] = i
    rows = []

    for method in stacked["method"].unique():
        subdf = stacked[stacked["method"]==method]
        if subdf.empty: 
            continue
        best_disco_row = subdf.loc[ subdf["disco"].idxmax() ].copy()
        best_disco_row["criterion"] = "disco"
        best_score_row = subdf.loc[ subdf["score"].idxmax() ].copy()
        best_score_row["criterion"] = "score"
        rows.append(best_disco_row.to_dict())
        rows.append(best_score_row.to_dict())
    groundtruth_eval = evaluate_groundtruth(fairdegen)
    groundtruth_eval["n"] = n
    groundtruth_eval["phi"] = phi
    groundtruth_eval["i"] = i
    rows.append(groundtruth_eval)
    return pd.DataFrame(rows), stacked
def main_abl_zafar(run=True,plot=True):

    N = 2000
    SEEDS = [11,22,33,44,55] 
    base_path = "results/synthetic/ZafarGenerator/Ablation/"
    if run:
        for seed in SEEDS:
            path = f"{base_path}{seed}/"
            Path(path).mkdir(parents=True,exist_ok=True)
            opt_dfs = []
            for i,phi in enumerate([0, np.pi/4, np.pi/2, np.pi]):
                opt_df, all_df = run_abl_zafar(path,seed,n=N,phi=phi,i=i)
                opt_dfs.append(opt_df)
                all_df.to_csv(f"{path}i={i}.csv", index=False)

            results = pd.concat(opt_dfs, ignore_index=True)
            results.to_csv(f"{path}results.csv", index=False)

            from src.utils.plot import plot_line

            if plot:
                plot_line(results=results, x="phi", path=path, categorical=True)
    avg_results, gt_results = aggregate_over_seeds(
    base_path, seeds = SEEDS, group_cols= ["phi", "method", "criterion"]
    )

    plot_line(avg_results, x="phi", path=base_path, categorical=True, groundtruth_results=gt_results, include_std=True)
if __name__ == "__main__":
    main_abl_zafar()