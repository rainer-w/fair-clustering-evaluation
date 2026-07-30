
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, HDBSCAN
from src.methods.FairDen import FairDen
from src.methods.FairSC import FairSC
from src.utils.helpers import evaluate_clustering


def generate_eps_candidates(
    X,
    k=5,
   # quantiles=np.linspace(0.05, 0.95, 10),
   quantiles = [0.1, 0.25, 0.5, 0.75, 0.9],
   discard_zeros = False
):
    """
    Generate DBSCAN epsilon candidates
    from kNN distances.
    """

    nbrs = NearestNeighbors(
        n_neighbors=k
    ).fit(X)

    distances, _ = nbrs.kneighbors(X)
   # print("distances ", distances)
    kth_distances = distances[:, -1]
    if discard_zeros:
        kth_distances = kth_distances[kth_distances > 0]
    eps_values = np.quantile(
        kth_distances,
        quantiles
    )

    return np.unique(eps_values)
def search_dbscan_all(
    X,
    y_true,
    sensitive,
    min_samples_values = None,
    discard_zeros = False
):
    results = []
    if min_samples_values is None: 
        min_samples_values = [
            5
            ,min(2* X.shape[1] - 1, X.shape[0]-1)
        ]
    print("minsamples values", min_samples_values)
    for min_samples in min_samples_values:

        eps_values = generate_eps_candidates(
            X,
            k=min_samples,
            quantiles=[0.1,0.25,0.5,0.75,0.9],
            discard_zeros=discard_zeros
        )
        #print("eps values : ", eps_values)
        for eps in eps_values:
            if eps <= 0.0: 
                continue
            t = time.perf_counter()
            model = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )
            
            y_pred = model.fit_predict(X)
            runtime = time.perf_counter() - t 

            row = evaluate_clustering(
                method="DBSCAN",
                X=X,
                y_pred=y_pred,
                y_true=y_true,
                sensitive=sensitive,
                params={
                    "eps": eps,
                    "min_samples": min_samples
                }
            )
            row["runtime"] = runtime
            results.append(row)

    return pd.DataFrame(results)
def search_hdbscan_all(
    X,
    y_true,
    sensitive,
    min_cluster_sizes = None
):
    results = []
    if min_cluster_sizes is None:
        k_true = len(np.unique(y_true[y_true != -1]))
        even_size = len(X) / k_true
        min_cluster_sizes = [
            5,
            10,
            20,
            30,
            50,
            100,
            int(even_size * 0.1),
            int(even_size * 0.25),
            int(even_size * 0.5),
            int(even_size * 0.75),
        ] 
    for mcs in min_cluster_sizes:
        t = time.perf_counter()
        model = HDBSCAN(
            min_cluster_size=mcs
        )
        
        y_pred = model.fit_predict(X)
        runtime = time.perf_counter() - t
        row = evaluate_clustering(
            method="HDBSCAN",
            X=X,
            y_pred=y_pred,
            y_true=y_true,
            sensitive=sensitive,
            params={
                "min_cluster_size": mcs
            }
        )
        row["runtime"] = runtime
        results.append(row)

    return pd.DataFrame(results)
def search_fairden_all(
    X,
    y_true,
    sensitive,
   # k_values=range(2, 22,2),
    k_values = None,
    min_pts_values= [5],
    k_unfair = None
):
    if k_values is None: 
        k_fair = len(np.unique(y_true[y_true!=-1]))
        k_values = [2,4,6,10,12,14,16,20, k_fair]
        if k_unfair is not None: 
            k_values.append(k_unfair)
        k_values =  sorted(set(k_values))

    results = []
    for min_pts in min_pts_values:
        t_0 = time.perf_counter()
        try:
            model = FairDen(
                min_pts=min_pts,
                n_clusters=None,
                data_wo_sensitive=X,
                sens_columns=np.array(sensitive).reshape(-1,1),
                sens_mixed=None
            )
            
        except np.linalg.LinAlgError as e:
            print(f"Skipping FairDen params min_pts={min_pts}: {e}")
            continue
        t_preprocess = time.perf_counter() - t_0
        for k in k_values:
            
            t_1 = time.perf_counter()
            y_pred = model.run(k=k)
            t_run = time.perf_counter() - t_1

            if y_pred is None: 
                y_pred = np.ones_like(y_true) * -1
            row = evaluate_clustering(
                method="FairDen",
                X=X,
                y_pred=y_pred,
                y_true=y_true,
                sensitive=sensitive,
                params={
                    "k": k,
                    "min_pts": min_pts
                }
            )
            
            row["runtime"] = t_preprocess + t_run
            row["t_preprocess"] = t_preprocess
            row["t_run"] = t_run
            results.append(row)

    return pd.DataFrame(results)

def search_fairsc_all(
        X, y_true, sensitive, k_values= None, k_unfair = None
):
    if k_values is None: 
        k_fair = len(np.unique(y_true[y_true!=-1]))
        k_values = [2,4,6,10,12,14,16,20, k_fair]
        if k_unfair is not None: 
            k_values.append(k_unfair)
        k_values =  sorted(set(k_values))

    results = []
    t_0 = time.perf_counter()
    model = FairSC(data_w_sensitive=X, sens_columns=np.array(sensitive).reshape(-1,1))
    t_preprocess = time.perf_counter() - t_0
    for k in k_values: 

        t_1 = time.perf_counter()
        y_pred = model.run(k=k)
        t_run = time.perf_counter() - t_1
        #y_pred = model.labels_
        row = evaluate_clustering(
            "FairSC", 
            X, 
            y_pred,
            y_true, 
            sensitive,
            params = {
                "k":k
            }
        )
        row["runtime"] = t_preprocess + t_run
        row["t_preprocess"] = t_preprocess
        row["t_run"] = t_run
        results.append(row)
    return pd.DataFrame(results)
from sklearn.cluster import KMeans
def search_kmeans_all(
        X, y_true, sensitive,k_values = None, k_unfair = None
): 
    if k_values is None: 
        k_fair = len(np.unique(y_true[y_true!=-1]))
        k_values = [2,4,6,10,12,14,16,20, k_fair]
        if k_unfair is not None: 
            k_values.append(k_unfair)
        k_values =  sorted(set(k_values))

    results = []
    for k in k_values: 
        t = time.perf_counter()
        model = KMeans(n_clusters= k).fit(X)
        runtime = time.perf_counter() - t
        y_pred = model.labels_
        row = evaluate_clustering(
            "KMeans", 
            X, 
            y_pred,
            y_true, 
            sensitive,
            params = {
                "k":k
            }
        )
        row["runtime"] = runtime
        results.append(row)
    return pd.DataFrame(results)

from src.methods.Fairlet import Fairlet


def compute_fairlets_threshold(X,sensitive,k=5):
    blue = X[sensitive==0]
    red = X[sensitive==1]
    if len(blue)==0 or len(red)==0:
        return np.inf

    nbrs = NearestNeighbors(n_neighbors=k).fit(red)
    distances,_ = nbrs.kneighbors(blue)
    kth = distances[:,-1]
    return np.median(kth)
def compute_fairlets_threshold_candidates(X, sensitive, k=5, quantiles = [0.5]):
    blue = X[sensitive == 0]
    red = X[sensitive == 1]

    if len(blue) == 0 or len(red) == 0:
        return [np.inf]

    nbrs = NearestNeighbors(n_neighbors=k).fit(red)
    distances, _ = nbrs.kneighbors(blue)

    kth = distances[:, -1]

    return np.unique(np.quantile(kth, quantiles))
import networkx as nx
def search_fairlets_all(
        X,y_true,sensitive,
        degrees = None,
        t = 2,
        k_unfair = None
):
    if degrees is None: 
        k_fair = len(np.unique(y_true[y_true!=-1]))
        degrees = [2,4,6,10,12,14,16,20, k_fair]
        if k_unfair is not None: 
            degrees.append(k_unfair)
        degrees =  sorted(set(degrees))

    results = []
    thresholds = compute_fairlets_threshold_candidates(X,sensitive)
  #  thresholds = np.append(thresholds,22500)
    for distance_threshold in thresholds:
        try:
            t_0 = time.perf_counter()
            model = Fairlet(X,sensitive,distance_threshold,t)
            t_preprocess = time.perf_counter() - t_0
        except nx.NetworkXUnfeasible:
 
            for degree in degrees: 
                y_pred = np.full(len(y_true), -1)
                row = evaluate_clustering("Fairlets",X, y_pred, y_true, sensitive, params = {"k":degree, "distance_threshold": distance_threshold, "t": t})

                row["except"]= "NXUnfeasible"

                results.append(row)
            continue

        for degree in degrees: 
            start = time.perf_counter()
            clusters,centers = model.run_experiment(degree)

            t_run = time.perf_counter() - start

            y_pred = np.full(len(y_true),-1)
            for point,cluster in clusters: 
                y_pred[point]=cluster

            row = evaluate_clustering(
                "Fairlets", 
                X, 
                y_pred,
                y_true,
                sensitive, 
                params = {
                    "k":degree,
                    "distance_threshold":distance_threshold
                }
            )
            row["runtime"] = t_preprocess + t_run
            results.append(row)
    return pd.DataFrame(results)


from src.methods.BackursFair import BackursFairClustering

def search_backurs_all(
        X,
        y_true,
        sensitive,
        k_values=None,
        pq=(1,1),
        k_unfair = None
):

    if k_values is None:
        k_true = len(
            np.unique(
                y_true[y_true!=-1]
            )
        )

        k_values = [
            2,4,6,10,12,14,16,20,k_true
        ]
        if k_unfair is not None:
            k_values.append(k_unfair)
        k_values = sorted(set(k_values))


    results=[]


    model = BackursFairClustering(
        X,
        sensitive,
        pq=pq
    )


    for k in k_values:

        start = time.perf_counter()

        y_pred = model.run(k)

        runtime = time.perf_counter()-start
        if y_pred is None: 
            continue

        row = evaluate_clustering(
            method="SFC",
            X=X,
            y_pred=y_pred,
            y_true=y_true,
            sensitive=sensitive,
            params={
                "k":k,
                "pq":pq
            }
        )

        row["runtime"] = runtime

        results.append(row)


    return pd.DataFrame(results)