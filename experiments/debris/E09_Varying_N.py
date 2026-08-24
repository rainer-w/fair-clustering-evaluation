
from src.generators.debris import DEBRIS
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.search import search_dbscan_all, search_hdbscan_all, search_fairden_all, search_kmeans_all,search_fairlets_all,search_fairsc_all,search_backurs_all
from src.utils.plot import plot_line
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth
def run_9(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n):
    
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
    fairdata.to_csv(f"{path}_{n}_data.csv",index=False)
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
        #fairlets_df = search_fairlets_all(features_wo, y_true,sensitive, k_unfair=k_unfair)
    fairsc_df = search_fairsc_all(features_w, y_true, sensitive, k_unfair=k_unfair)
    backurs_df = search_backurs_all(features_wo, y_true, sensitive,pq=(1,2),k_unfair=k_unfair)
    stacked = pd.concat(
        [dbscan_df, hdbscan_df, fairden_df, kmeans_df, 
        # fairlets_df, 
         fairsc_df, backurs_df]
    )
    
    stacked["score"] = stacked["disco"] + stacked["balance"]
    stacked["dim"] = dim 
    stacked["n"] = n
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
    rows.append(groundtruth_eval)
    return pd.DataFrame(rows), stacked
def main_9(run=True,plot=True):
    
    SEEDS = [11,22,33,44,55] 
    RATIO_NOISE = 0.05
    G = 2
    DISTR =  [[0.9,0.1], [0.1,0.9]] 
    CLUNUM = 10
    CORE_NUM = [15] * CLUNUM
    DIM = 10
    base_path = "results/debris/Experiment9/"
    if run:
        for seed in SEEDS:
            path = f"{base_path}{seed}/"
            Path(path).mkdir(parents=True,exist_ok=True)
            opt_dfs = []
            for n in [500, 1000, 2000, 5000, 10000]:
                opt_df, all_df = run_9(path,DIM,CLUNUM,seed,CORE_NUM,RATIO_NOISE,G,DISTR,n=n)
                opt_dfs.append(opt_df)
                all_df.to_csv(f"{path}n={n}.csv", index=False)

            results = pd.concat(opt_dfs, ignore_index=True)
            results.to_csv(f"{path}results.csv", index=False)

            #new_results = pd.concat(opt_dfs, ignore_index=True)
            #existing = pd.read_csv(f"{path}results.csv")
            #results = pd.concat([existing, new_results], ignore_index=True)
            #results.to_csv(f"{path}results.csv", index=False)

            if plot:
                plot_line(results=results, x="n", path=path, categorical=True)
    avg_results, gt_results = aggregate_over_seeds(
    base_path, seeds = SEEDS, group_cols= ["n", "method", "criterion"]
    )
 
    plot_line(avg_results, x="n", path=base_path, categorical=False, logaxis="x", groundtruth_results=gt_results, include_std=True)
if __name__ == "__main__":
    main_9()