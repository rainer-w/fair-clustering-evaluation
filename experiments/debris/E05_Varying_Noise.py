
from src.generators.debris import DEBRIS
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.search import search_dbscan_all, search_hdbscan_all, search_fairden_all, search_kmeans_all,search_fairlets_all,search_fairsc_all,search_backurs_all
from src.utils.plot import plot_line
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth
def run_5(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n):
    
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
    fairdata.to_csv(f"{path}_{ratio_noise}_data.csv",index=False)
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
    fairlets_df = search_fairlets_all(features_wo, y_true,sensitive, k_unfair=k_unfair)
    fairsc_df = search_fairsc_all(features_w, y_true, sensitive, k_unfair=k_unfair)
    backurs_df = search_backurs_all(features_wo, y_true, sensitive,pq=(1,2),k_unfair=k_unfair)
    stacked = pd.concat(
        [dbscan_df, hdbscan_df, fairden_df, kmeans_df, fairlets_df, fairsc_df,backurs_df]
    )
    
    stacked["score"] = stacked["disco"] + stacked["balance"]
    stacked["ratio_noise"] = ratio_noise 
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
    groundtruth_eval["ratio_noise"]=ratio_noise
    rows.append(groundtruth_eval)
    return pd.DataFrame(rows), stacked
def main_5(run=True,plot=True):

    SEEDS = [11,22,33,44,55] 
    DIM = 10 
    G = 2
    DISTR =  [[0.9,0.1], [0.1,0.9]] 
    CLUNUM = 10
    CORE_NUM = [15] * CLUNUM
    
    base_path = "results/debris/Experiment5/"
    if run:
        for seed in SEEDS: 
            path = f"{base_path}{seed}/"
            Path(path).mkdir(parents=True,exist_ok=True)
            opt_dfs = []
        
            for ratio_noise in [0.0, 0.05, 0.1,  0.15, 0.2, 0.3, 0.4, 0.5] :
    
                opt_df, all_df = run_5(path,DIM,CLUNUM,seed,CORE_NUM,ratio_noise,G,DISTR,n=1000)
                opt_dfs.append(opt_df)
                all_df.to_csv(f"{path}ratio_noise={ratio_noise}.csv",index=False)
            results = pd.concat(opt_dfs, ignore_index=True)
            results.to_csv(f"{path}results.csv",index=False)

            #new_results = pd.concat(opt_dfs, ignore_index=True)
            #existing = pd.read_csv(f"{path}results.csv")
            #results = pd.concat([existing, new_results], ignore_index=True)
            #results.to_csv(f"{path}results.csv", index=False)

            from src.utils.plot import plot_line

            if plot:
                plot_line(results=results, x="ratio_noise", path=path, categorical=True, tick_labels=["0", ".05", ".1", ".15", ".2", ".3", ".4", ".5"])
    avg_results, gt_results = aggregate_over_seeds(
    base_path, seeds = SEEDS, group_cols= ["ratio_noise", "method", "criterion"]
    )
    print("GT RESULTS , ", gt_results)
    plot_line(avg_results, x="ratio_noise", path=base_path, categorical=True, tick_labels=["0", ".05", ".1", ".15", ".2", ".3", ".4", ".5"], groundtruth_results=gt_results, include_std=True)
if __name__ == "__main__":
    main_5()