from src.utils.plot import plot_fair_clusters, _project_if_needed, plot_cluster_labels
from src.generators.debris import DEBRIS

if __name__ =="__main__": 
    NUM_CLUSTERS = 5
    RATIO_NOISE = 0.15
    D = 2
    N = 500
    G = 2
    SEED = 33
    DISTR = [[0.9,0.1],[0.1,0.9]]
    CORE_NUM = 90
    N=500
    fairdegen = DEBRIS(
        dim=D,
        clunum=NUM_CLUSTERS,
        seed=SEED,
        core_num=CORE_NUM,
        ratio_noise=RATIO_NOISE,
        g=G,
        distr = DISTR
    )
    fairdata = fairdegen.generate_data(N)
    X_pca, pca = _project_if_needed(fairdegen.get_features_wo_sensitive())


    plot_fair_clusters(
        X = X_pca,
        semantic_cluster=fairdegen.get_groundtruth(), 
        subgroup_ids=fairdegen.get_unfair_groundtruth(), 
        sensitive_attribute= fairdegen.get_sensitive(),
        subgroup_type=fairdegen.get_subgroup_type(),
        cores=fairdegen.get_cores(),
        core_subgroups=fairdegen.get_core_subgroups(), 
        path = "figures/"
    )
    plot_cluster_labels(Xp =X_pca, labels=fairdegen.get_groundtruth(), sensitive = fairdegen.get_sensitive(),
                        path = "figures/", filename="gt_clusters.pdf",
                        title="", show_hulls=True, show_legend=False, hull_alpha=0.5)
    plot_cluster_labels(Xp =X_pca, labels=fairdegen.get_unfair_groundtruth(),sensitive = fairdegen.get_sensitive(),
                        path = "figures/", filename="density_clusters.pdf",
                        title="", show_hulls=True, show_legend=False, hull_alpha=0.5)