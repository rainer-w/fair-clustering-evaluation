
from src.generators.debris import DEBRIS
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.search import search_dbscan_all, search_hdbscan_all, search_fairden_all, search_kmeans_all,search_fairlets_all, search_fairsc_all
from src.utils.helpers import aggregate_over_seeds, evaluate_groundtruth
from src.utils.plot import plot_line
from sklearn.metrics import silhouette_score
def main(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n, i):
    
    fairdegen = DEBRIS(
        dim = dim, 
        clunum=clunum,
        seed=seed,
        core_num= core_num, 
        ratio_noise = ratio_noise,
        g=g,
        distr = distr
	)
    fairdata = fairdegen.generate_data(n,seed=seed)

    features_wo = fairdegen.get_features_wo_sensitive()
    features_w = fairdegen.get_features_w_sensitive()

    sensitive = fairdegen.get_sensitive()
    y_true = fairdegen.get_groundtruth()
    y_unfair = fairdegen.get_unfair_groundtruth()
    k_unfair = len(np.unique(y_unfair[y_unfair != -1]))

    sil_fair_wsens = silhouette_score(features_w, y_true)
    sil_unf_wsens = silhouette_score(features_w, y_unfair)

    sil_fair_wosens = silhouette_score(features_wo, y_true)
    sil_unf_wosens = silhouette_score(features_wo, y_unfair)

    results = {
        "index" : [seed],
     #   "seed" : seed,
        "Fair Wsens" : sil_fair_wsens, 
        "Fair WOsens" : sil_fair_wosens, 
        "Unf Wsens" : sil_unf_wsens, 
        "Unf WOsens" : sil_unf_wosens
    }

   # print("K : ", clunum)
    #print(results)
    return pd.DataFrame(results)
if __name__ == "__main__":
    run = True
    plot = True
    import random
    SEEDS = []
    DIM = 10 
    RATIO_NOISE = 0.05
    G = 2
    DISTR =  [[0.9,0.1], [0.1,0.9]] 
    CORES_PER_CLUSTER = 15
    base_path = "results/debris/Experiment2/"
    if run:
        res_dfs = []
        for i in range(100): 
            seed = random.randint(0,2147483647)

            distr = [[0.5,0.5], [0.5,0.5]]
            # print("distr i", i)
            clunum = 20
            CORE_NUM = [CORES_PER_CLUSTER] * clunum
            res_df = main("",DIM,clunum,seed,CORE_NUM,RATIO_NOISE,G,distr,n=1000, i=i)
            res_dfs.append(res_df)
        all_df = pd.concat(res_dfs, ignore_index=True)
        print(all_df)
        print(all_df.describe())
