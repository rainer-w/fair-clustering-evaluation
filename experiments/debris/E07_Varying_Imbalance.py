
from src.generators.debris import DEBRIS
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.search import search_dbscan_all, search_hdbscan_all, search_fairden_all, search_kmeans_all,search_fairlets_all,search_fairsc_all,search_backurs_all
from src.utils.plot import plot_line
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth
def run_7(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n, distr_index):
    
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
    t = 2
    if distr_index > 4: # global distribution for dataset indices 3-5 != 50:50 -> update parameter of fairlets to allow "imbalance"
        t = 5
    elif distr_index >= 6: 
        t = 9
    fairlets_df = search_fairlets_all(features_wo, y_true,sensitive, t = t, k_unfair=k_unfair)
    fairsc_df = search_fairsc_all(features_w, y_true, sensitive, k_unfair=k_unfair)
    PQ = (1,2)
    if distr_index > 4: 
        PQ = (1,5)
    if distr_index >= 6: 
        PQ = (1,6)
    backurs_df = search_backurs_all(features_wo, y_true, sensitive,pq=PQ,k_unfair=k_unfair)
    stacked = pd.concat(
        [dbscan_df, hdbscan_df, fairden_df, kmeans_df, fairlets_df, fairsc_df,backurs_df]
    )

    stacked["score"] = stacked["disco"] + stacked["balance"]
    stacked["distr_index"] =  distr_index
    stacked["distr"] = str(distr)
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
    groundtruth_eval["distr_index"] = distr_index
    groundtruth_eval["distr"] = str(distr)
    rows.append(groundtruth_eval)
    
    return pd.DataFrame(rows), stacked
def main_7(run=True,plot=True):

    
    SEEDS = [11,22,33,44,55]
    DIM = 10
    RATIO_NOISE = 0.05
    G = 2
    CLUNUM = 10
    CORE_NUM = [15] * CLUNUM
    base_path = "results/debris/Experiment7/"
    if run:
        for seed in SEEDS: 
            path = f"{base_path}{seed}/"
            Path(path).mkdir(parents=True,exist_ok=True)
            opt_dfs = []
        
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

                opt_df,all_df = run_7(path,DIM,CLUNUM,seed,CORE_NUM,RATIO_NOISE,G,distr,n=1000, distr_index=i)
                opt_dfs.append(opt_df)
                all_df.to_csv(f"{path}distr_index={i}.csv", index=False)

            results = pd.concat(opt_dfs, ignore_index=True)
            results.to_csv(f"{path}results.csv",index=False)

            #new_results = pd.concat(opt_dfs, ignore_index=True)
            #existing = pd.read_csv(f"{path}results.csv")
            #results = pd.concat([existing, new_results], ignore_index=True)
            #results.to_csv(f"{path}results.csv", index=False)

            # plot per seed
            if plot:
                plot_line(results=results, x="distr_index", path=path, categorical=True)
    avg_results, gt_results = aggregate_over_seeds(
    base_path, seeds = SEEDS, group_cols= ["distr_index", "method", "criterion"]
    )

    x_split = 5
    avg_A = avg_results[avg_results["distr_index"] < x_split].copy()
    avg_B = avg_results[avg_results["distr_index"] >= x_split].copy()
    gt_results = gt_results.loc[:, ~gt_results.columns.duplicated()]
    gt_A = gt_results[gt_results["distr_index"] < x_split].copy() 
    gt_B = gt_results[gt_results["distr_index"] >= x_split].copy() 
    plot_line(avg_A, x="distr_index", tick_labels=["A0", "A1", "A2", "A3", "A4"], path=f"{base_path}A_", categorical=True, groundtruth_results=gt_A, include_std=True, xlabel="Subgroup Imbalance Level")
    plot_line(avg_B, x="distr_index",tick_labels=["B0", "B1", "B2", "B3", "B4"], path=f"{base_path}B_", categorical=True, groundtruth_results=gt_B, include_std=True, xlabel="Subgroup Imbalance Level")
if __name__ == "__main__":
    main_7()