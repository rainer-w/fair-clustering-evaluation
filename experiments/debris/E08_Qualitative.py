from src.utils.search import search_dbscan_all,search_hdbscan_all, search_fairden_all, search_kmeans_all, search_fairlets_all, search_fairsc_all, search_backurs_all
from src.utils.plot import plot_fair_clusters, plot_filtered_skyline
from src.generators.debris import DEBRIS
import pandas as pd
import numpy as np
from pathlib import Path
def run_8(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n, fairlets_t, gap=2,alpha=1, run = True):
  #  cluster_ratios = np.random.dirichlet(np.ones(clunum) * alpha)
    fairdegen = DEBRIS(
        dim = dim, 
        clunum=clunum,
        seed=seed,
        core_num= core_num, 
        ratio_noise = ratio_noise,
        g=g,
        distr = distr,
     #   clu_ratios=cluster_ratios,
    #    gap = gap
	)
    fairdata = fairdegen.generate_data(n)

    if run == False:
        return [], fairdegen, fairdata
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
    
    dfs = [
        dbscan_df, hdbscan_df, fairden_df, kmeans_df, fairsc_df
    ]
    if g == 2: 
       # fairlets_df = search_fairlets_all(features_wo, y_true,sensitive, t = fairlets_t, k_unfair=k_unfair)
       # dfs.append(fairlets_df)
        backurs_df = search_backurs_all(features_wo, y_true, sensitive,pq=(1,fairlets_t),k_unfair=k_unfair)
        dfs.append(backurs_df)
    stacked = pd.concat(
        dfs, ignore_index=True
    )
    stacked["score"] = stacked["disco"] + stacked["balance"]
    stacked["dim"] = dim
    stacked["n"] = n
    stacked["clunum"] = clunum
    stacked["seed"] = seed
    stacked["core_num"] = str(core_num)
    stacked["ratio_noise"] = ratio_noise
    stacked["g"] = g
    stacked["distr"] = str(distr)
    stacked["gap"] = gap
    stacked["alpha"] = alpha
    
    plot_fair_clusters(
        X=fairdegen.get_features_wo_sensitive(),
        semantic_cluster=fairdata["groundtruth"],
        subgroup_ids=fairdata["subgroup_id"],
        sensitive_attribute=fairdata["sensitive_value"],
        subgroup_type=fairdegen.subgroup_type,
        cores=fairdegen.cores,
        core_subgroups=fairdegen.core_subgroup,
        title="", 
        path = path
    )
    plot_filtered_skyline(stacked,x="disco", path=path)
    return stacked, fairdegen, fairdata

def main_8(run=True,plot=True):
    path = "results/debris/Experiment8/"

    """
    
    result1 = pd.read_csv(f"{path}Setting_1_result.csv")
    result2 = pd.read_csv(f"{path}Setting_2_result.csv")
    result3 = pd.read_csv(f"{path}Setting_3_result.csv")
    data1 = pd.read_csv(f"{path}Setting_1_data.csv")
    data2 = pd.read_csv(f"{path}Setting_2_data.csv")
    data3 = pd.read_csv(f"{path}Setting_3_data.csv")
    stacked_map = {
        1: result1, 
        2: result2, 
        3: result3
    }
    for i in [1,2,3]: 
        plot_filtered_skyline(stacked_map[i],x="disco",x_threshold=-0.5 , path=f"{path}Setting_{i}_")
        """


    Path(path).mkdir(parents=True,exist_ok=True)
    stacked_map = {1:None,2:None,3:None}
    result1,fde1,fda1 = run_8(f"{path}Setting_1_",dim=10,clunum=2,seed=27,core_num=[15,15],ratio_noise=0.05,g=2,distr=[[0.3,0.7], [0.1,0.9]],n=1000, fairlets_t=5, run=run)
    if  run:
        result1.to_csv(f"{path}Setting_1_result.csv",index=False)
        fda1.to_csv(f"{path}Setting_1_data.csv", index=False)
        stacked_map[1] = result1
    result2, fde2,fda2 = run_8(f"{path}Setting_2_",dim=10, clunum=3, seed=174, core_num=[3,9,25], ratio_noise=0.05, g=2, distr=[[0.3,0.7], [0.1,0.9]],n=1000, fairlets_t=5,run=run)
    if  run:
        result2.to_csv(f"{path}Setting_2_result.csv",index=False)
        fda2.to_csv(f"{path}Setting_2_data.csv", index=False)
        stacked_map[2] = result2
    import random 
    CLUNUM_3 = 20
    random.seed(11)
    core_num_setting_3 = [random.randint(3, 30) for _ in range(CLUNUM_3)]
    print("core nums of clusters : ", core_num_setting_3)
    result3,fde3,fda3 = run_8(f"{path}Setting_3_", dim=10, clunum=CLUNUM_3, seed=11, core_num=core_num_setting_3, ratio_noise=0.05, g=2, distr=[[0.2,0.8], [0.05,0.95]],n=1000, fairlets_t=8, run=run)
    if  run:
        result3.to_csv(f"{path}Setting_3_result.csv",index=False)
        fda3.to_csv(f"{path}Setting_3_data.csv", index=False)
        stacked_map[3] = result3
    if plot: 
        if not run: 
            result1 = pd.read_csv(f"{path}Setting_1_result.csv")
            result2 = pd.read_csv(f"{path}Setting_2_result.csv")
            result3 = pd.read_csv(f"{path}Setting_3_result.csv")
            data1 = pd.read_csv(f"{path}Setting_1_data.csv")
            data2 = pd.read_csv(f"{path}Setting_2_data.csv")
            data3 = pd.read_csv(f"{path}Setting_3_data.csv")
            stacked_map = {
                1: result1, 
                2: result2, 
                3: result3
            }
        id_fde_dict = {
            1:fde1,
            2:fde2,
            3:fde3
        }
        for i,fd in id_fde_dict.items():
        #for i,fd in enumerate([fde1,fde2,fde3]):
            plot_fair_clusters(fd.get_features_wo_sensitive(), 
                               fd.get_groundtruth(),
                               fd.get_unfair_groundtruth(), 
                               fd.get_sensitive(), 
                               fd.get_subgroup_type(), 
                               fd.get_cores(), 
                               fd.get_core_subgroups(), 
                               alpha_points=0.4,
                               path=f"{path}Setting_{i}_" )
            plot_filtered_skyline(stacked_map[i],x="disco",x_threshold=-1.0, path=f"{path}Setting_{i}_",
                                  xlim = (-1,1), ylim = (0,1))

if __name__ == "__main__":
    main_8()