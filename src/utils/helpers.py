from src.evaluation.balance import balance_score
from src.evaluation.disco import disco_score

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)
import numpy as np

def evaluate_groundtruth( 
    fairdegen
) : 
    features = fairdegen.get_features_wo_sensitive()
    sensitive = fairdegen.get_sensitive()
    fair_gt = fairdegen.get_groundtruth()
    unfair_gt = fairdegen.get_unfair_groundtruth()

    balance_fair = balance_score("test",["sensitive_value"], fair_gt, sensitive)
    balance_unfair = balance_score("test", ["sensitive_value"], unfair_gt, sensitive)
    disco_fair = disco_score(features, fair_gt)
    disco_unfair = disco_score(features, unfair_gt)

    row = {
        "method" : "GroundTruth",
        "balance_fair" : balance_fair, 
        "balance_unfair" : balance_unfair, 
        "disco_fair" : disco_fair, 
        "disco_unfair" : disco_unfair
    }
    return row
def evaluate_groundtruth_separate(
        fairdegen
): 
    features = fairdegen.get_features_wo_sensitive()
    sensitive = fairdegen.get_sensitive()
    fair_gt = fairdegen.get_groundtruth()
    unfair_gt = fairdegen.get_unfair_groundtruth()
    balance_fair = balance_score("test",["sensitive_value"], fair_gt, sensitive)
    balance_unfair = balance_score("test", ["sensitive_value"], unfair_gt, sensitive)
    disco_fair = disco_score(features, fair_gt)
    disco_unfair = disco_score(features, unfair_gt)

    row_fair = {
        "method" : "GT_Fair",
        "balance" : balance_fair, 
        "disco" : disco_fair,
    }
    row_unfair = {
        "method" : "GT_Unfair",
        "balance" : balance_unfair,
        "disco" : disco_unfair
    }
    rows = [row_fair,row_unfair]
    return rows
def add_gt_deviations(df, gt_rows):
    gt_fair = next(
        row for row in gt_rows
        if row["method"] == "GT_Fair"
    )

    gt_unfair = next(
        row for row in gt_rows
        if row["method"] == "GT_Unfair"
    )

    df = df.copy()

    df["delta_balance_fair"] = (
        df["balance"] - gt_fair["balance"]
    )

    df["delta_disco_fair"] = (
        df["disco"] - gt_fair["disco"]
    )

    df["delta_balance_unfair"] = (
        df["balance"] - gt_unfair["balance"]
    )

    df["delta_disco_unfair"] = (
        df["disco"] - gt_unfair["disco"]
    )

    return df
def summarize_gt_deviations(df, gt_rows):
    # Extract GT values
    gt_fair = next(
        row for row in gt_rows
        if row["method"] == "GT_Fair"
    )

    gt_unfair = next(
        row for row in gt_rows
        if row["method"] == "GT_Unfair"
    )

    # Work on a copy
    df = df.copy()

    # Calculate per-row deviations
    df["deviation_balance_fair"] = (
        df["balance"] - gt_fair["balance"]
    )

    df["deviation_disco_fair"] = (
        df["disco"] - gt_fair["disco"]
    )

    df["deviation_balance_unfair"] = (
        df["balance"] - gt_unfair["balance"]
    )

    df["deviation_disco_unfair"] = (
        df["disco"] - gt_unfair["disco"]
    )

    # Aggregate over all parameter settings per method
    summary = (
        df.groupby("method")
        .agg(
            deviation_balance_fair=("deviation_balance_fair", "mean"),
            deviation_balance_fair_std=("deviation_balance_fair", "std"),

            deviation_disco_fair=("deviation_disco_fair", "mean"),
            deviation_disco_fair_std=("deviation_disco_fair", "std"),

            deviation_balance_unfair=("deviation_balance_unfair", "mean"),
            deviation_balance_unfair_std=("deviation_balance_unfair", "std"),

            deviation_disco_unfair=("deviation_disco_unfair", "mean"),
            deviation_disco_unfair_std=("deviation_disco_unfair", "std"),
        )
        .reset_index()
    )

    return summary
def evaluate_clustering(
    method,
    X,
    y_pred,
    y_true,
    sensitive,
    params,
):
	
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    balance = balance_score("notneeded", ["sensitive_value"], y_pred, sensitive)
   # print("y_pred = ", y_pred)
    y_pred_arr = np.asarray(y_pred)
  #  print("y_pred_arr = ", y_pred_arr)
    disco = disco_score(X,y_pred)
    row = {
        "method": method,
        "ari": ari,
        "nmi": nmi,
        "disco": disco,
        "balance": balance,
        **params,
        "labels": y_pred,
        #"n_clusters" : len(set(y_pred)) - (1 if -1 in y_pred else 0)
        "n_clusters": len(np.unique(y_pred[y_pred != -1])),
        "noise_fraction" : np.mean(y_pred_arr == -1)#(y_pred == -1).mean()
    }

    return row
import pandas as pd
import json
def load_dataset(path,prefix):

    df = pd.read_csv(f"{path}{prefix}_data.csv")

    with open(f"{prefix}_metadata.json", "r") as f:
        metadata = json.load(f)
    metadata["subgroup_type"] = {
        int(k):v for k,v in metadata["subgroup_type"].items()
	}

    return df, metadata

from pathlib import Path
import pandas as pd


def backup_aggregate_over_seeds(
    base_path,
    seeds,
    group_cols,
    metrics=["disco", "balance", "score", "runtime"]
):

    dfs = []

    base_path = Path(base_path)

    missing = []

    for seed in seeds:

        result_file = base_path / str(seed) / "results.csv"

        if not result_file.exists():
            missing.append(seed)
            continue

        df = pd.read_csv(result_file)

        df["seed"] = seed
        dfs.append(df)


    if len(dfs) == 0:
        raise FileNotFoundError(
            f"No results.csv files found in {base_path} for seeds {seeds}"
        )


    if missing:
        print(
            f"Warning: skipped missing seeds: {missing}"
        )


    all_results = pd.concat(
        dfs,
        ignore_index=True
    )


    avg_results = (
        all_results
        .groupby(
            group_cols,
            as_index=False
        )[metrics]
        .mean()
    )


    print(
        f"Aggregated {len(dfs)}/{len(seeds)} seeds"
    )


    return avg_results

def aggregate_over_seeds(
    base_path,
    seeds,
    group_cols,
    metrics=["disco", "balance", "score", "runtime"]
):

    dfs = []
    base_path = Path(base_path)
    missing = []

    for seed in seeds:
        result_file = base_path / str(seed) / "results.csv"

        if not result_file.exists():
            missing.append(seed)
            continue

        df = pd.read_csv(result_file)
        df["seed"] = seed
        dfs.append(df)

    if len(dfs) == 0:
        raise FileNotFoundError(
            f"No results.csv files found in {base_path} for seeds {seeds}"
        )

    if missing:
        print(f"Warning: skipped missing seeds: {missing}")

    all_results = pd.concat(dfs, ignore_index=True)
    gt_df = all_results[all_results["method"] == "GroundTruth"]
    model_df = all_results[all_results["method"] != "GroundTruth"]

    grouped = model_df.groupby(group_cols, as_index=False)

    mean_df = grouped[metrics].mean()
    std_df = grouped[metrics].std()

    std_df = std_df.rename(columns={m: f"{m}_std" for m in metrics})

    merged = mean_df.merge(
        std_df,
        on=group_cols,
        how="left"
    )

    gt_metrics = ["balance_fair", "balance_unfair", "disco_fair", "disco_unfair"]
    main_col = group_cols[0]
    gt_grouped = gt_df.groupby([main_col], as_index=False)
    mean_gt = gt_grouped[gt_metrics].mean()
    std_gt = gt_grouped[gt_metrics].std()
    std_gt = std_gt.rename(columns={c: f"{c}_std" for c in gt_metrics})
    gt_summary = mean_gt.merge( 
        std_gt, on=[main_col], how="left"
    )
   # gt_summary = pd.concat([mean_gt, std_gt], axis=1)

    return merged, gt_summary