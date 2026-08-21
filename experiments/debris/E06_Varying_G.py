from src.generators.debris import DEBRIS
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.search import search_dbscan_all, search_hdbscan_all, search_fairden_all, search_kmeans_all,search_fairlets_all,search_fairsc_all
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth
from src.utils.plot import plot_line
def run_6(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n, distr_index):
    
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
    fairdata.to_csv(f"{path}{distr_index}_data.csv",index=False)

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

    fairsc_df = search_fairsc_all(features_w, y_true, sensitive, k_unfair=k_unfair)
    stacked = pd.concat(
        [dbscan_df, hdbscan_df, fairden_df, kmeans_df, fairsc_df]
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
def main_6(run=True,plot=True):

    SEEDS = [11,22,33,44,55]
    DIM = 10
    RATIO_NOISE = 0.05
    base_path = "results/debris/Experiment6/"
    CLUNUM = 10
    CORE_NUM = [15] * CLUNUM
    if run:
        
        for seed in SEEDS: 
            path = f"{base_path}{seed}/"
            Path(path).mkdir(parents=True,exist_ok=True)
            opt_dfs = []
        
            for i,distr in enumerate([
                [[0.9,0.1], [0.1,0.9]], 
                [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05,0.05,0.9]], 
                [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05],  [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85] ], 
                [[0.8, 0.05, 0.05, 0.05, 0.05], [0.05, 0.8, 0.05, 0.05, 0.05], [0.05, 0.05, 0.8, 0.05, 0.05], 
                 [0.05, 0.05, 0.05, 0.8, 0.05], [0.05, 0.05, 0.05, 0.05, 0.8] 
                 ]
            ]) :
                g = len(distr)
                #core_num = CORE_NUM + g*10
                opt_df,all_df = run_6(path,DIM,CLUNUM,seed,CORE_NUM,RATIO_NOISE,g,distr,n=1000, distr_index=i)
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
                plot_line(results=results, x="distr_index", path=path, tick_labels=[2,3,4,5], xlabel="g", categorical=True, include_std=False)
    avg_results, gt_results = aggregate_over_seeds(
        base_path, seeds = SEEDS, group_cols= ["distr_index", "method", "criterion"]
    )

    plot_line(avg_results, x="distr_index", path=base_path, tick_labels=[2,3,4,5], xlabel="g", categorical=True, groundtruth_results=gt_results, include_std=True)
if __name__ == "__main__":
    main_6()