from src.utils.search import search_dbscan_all,search_hdbscan_all, search_fairden_all, search_kmeans_all, search_fairlets_all, search_fairsc_all, search_backurs_all
from src.utils.plot import plot_filtered_skyline
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.load_uci import load_diabetes, load_adult, load_bank, load_census, load_creditcard
from src.evaluation.balance import balance_score
from src.evaluation.disco import disco_score
from experiments.realworld.table_uci import main_table_uci
dataset_map = {
    "Adult" : load_adult, 
    "Bank": load_bank, 
    "Census":load_census, 
    "Creditcard": load_creditcard,
    "Diabetes" : load_diabetes
    
}
from fractions import Fraction

def get_backurs_balance(sens, max_denominator=20):
    _, counts = np.unique(sens, return_counts=True)

    ratio = counts.min() / counts.max()

    candidates = []

    for q in range(1, max_denominator + 1):
        for p in range(1, q + 1):
            if p / q <= ratio:
                candidates.append((p, q))

    # choose the strongest feasible constraint
    return max(candidates, key=lambda x: x[0] / x[1])
def run_uci(base_path:str, dataset_name : str): 
    print("running : ", dataset_name)
    Path(base_path).mkdir(parents=True,exist_ok=True)
    SEEDS = [11,22,33,44,55]
    N = 2000
    X_w, X_wo, sens, y_true = dataset_map[dataset_name]()
    sens  = sens.iloc[:, 0]
    global_distribution_full = (
        sens.value_counts(normalize=True)
            .sort_index()
            .round(3)
                .tolist()
        )
    opt_dfs = []
    all_dfs = []
    for seed in SEEDS:
        print("running seed : ", seed)
        path = f"{base_path}{seed}/"
        Path(path).mkdir(parents=True,exist_ok=True)
        rng = np.random.default_rng(seed)

        sampled_idx = rng.choice(
            X_w.index,
            size=N,
            replace=False
        )

        sub_X_w = X_w.loc[sampled_idx].reset_index(drop=True).to_numpy()
        sub_X_wo = X_wo.loc[sampled_idx].reset_index(drop=True).to_numpy()
        sub_sens = sens.loc[sampled_idx].reset_index(drop=True).to_numpy()
        sub_y = y_true.loc[sampled_idx].reset_index(drop=True).to_numpy()

        global_distribution_sub = (
            pd.Series(sub_sens).value_counts(normalize=True)
                .sort_index()
                .round(3)
                    .tolist()
        )
        print("global distr. of subset : ", global_distribution_sub)
        dbscan_df = search_dbscan_all(sub_X_w, sub_y, sub_sens)
        hdbscan_df = search_hdbscan_all(sub_X_w, sub_y,sub_sens)
        fairden_df = search_fairden_all(sub_X_wo, sub_y, sub_sens)
        kmeans_df = search_kmeans_all(sub_X_w, sub_y, sub_sens)
       # fairlets_df = search_fairlets_all(sub_X_wo, sub_y,sub_sens)
        fairsc_df = search_fairsc_all(sub_X_w, sub_y, sub_sens)


        backurs_df = search_backurs_all(sub_X_wo, sub_y, sub_sens, pq=get_backurs_balance(sub_sens))
        stacked = pd.concat(
            [dbscan_df, hdbscan_df, fairden_df, kmeans_df, #fairlets_df, 
             fairsc_df, backurs_df]
        )
        stacked["score"] = stacked["disco"] + stacked["balance"]
        
        gt_balance = balance_score("test",["sensitive_value"], sub_y, sub_sens)
        stacked["gt_balance"] = gt_balance
        gt_disco = disco_score(sub_X_wo, sub_y)
        stacked["gt_disco"] = gt_disco
        stacked["seed"] = seed


        stacked["global_distribution_full"] = str(global_distribution_full)
        stacked["global_distribution_sub"] = str(global_distribution_sub)

        rows = []

        for method in stacked["method"].unique():
            subdf = stacked[stacked["method"]==method]
            if subdf.empty: 
                continue
            best_disco_row = subdf.loc[ subdf["disco"].idxmax() ].copy()
            best_disco_row["criterion"] = "disco"
            best_score_row = subdf.loc[ subdf["score"].idxmax() ].copy()
            best_score_row["criterion"] = "score"
            best_balance_row = subdf.loc[ subdf["balance"].idxmax() ].copy()
            best_balance_row["criterion"] = "balance"
            rows.append(best_disco_row.to_dict())
            rows.append(best_score_row.to_dict())
            rows.append(best_balance_row.to_dict())
        opt_df = pd.DataFrame(rows)
        opt_df["seed"] = seed
        opt_df.to_csv(f"{path}opt_results.csv",index=False)
        stacked.to_csv(f"{path}all_results.csv", index=False)

        opt_dfs.append(opt_df)
        all_dfs.append(stacked)

    opt_results = pd.concat(opt_dfs, ignore_index=True)
    opt_results.to_csv(f"{base_path}opt_results.csv",index=False)
    all_results = pd.concat(all_dfs, ignore_index=True)
    all_results.to_csv(f"{base_path}all_results.csv")

    group_cols = ["method", "criterion"]
    grouped = opt_results.groupby(group_cols, as_index=False)
    metrics=["disco", "balance", "score", "runtime"]
    mean_df = grouped[metrics].mean()
    std_df = grouped[metrics].std()

    std_df = std_df.rename(columns={m: f"{m}_std" for m in metrics})

    merged = mean_df.merge(
        std_df,
        on=group_cols,
        how="left"
    )

    merged.to_csv(f"{base_path}mean_std_results.csv", index=False)

def main_all_uci(run=True,plot=True): 

    for dataset_name in ["Adult","Bank","Census", "Creditcard","Diabetes"]:
        if run:
            run_uci(base_path=f"results/realworld/{dataset_name}/", dataset_name=dataset_name)

        if plot:
            df = pd.read_csv(f"results/realworld/{dataset_name}/opt_results.csv")
            score_df = df[df["criterion"] == "score"].copy()
            plot_filtered_skyline(score_df, x="n_clusters", path=f"results/realworld/{dataset_name}/optscore_n_clusters-")
            plot_filtered_skyline(score_df, x="disco", path=f"results/realworld/{dataset_name}/optscore_disco-")
            disco_df = df[df["criterion"] == "disco"].copy()
            plot_filtered_skyline(disco_df, x="n_clusters", path=f"results/realworld/{dataset_name}/optdisco_n_clusters-")
            plot_filtered_skyline(disco_df, x="disco", path=f"results/realworld/{dataset_name}/optdisco_disco-")

if __name__ == "__main__": 
    main_all_uci()
    main_table_uci()