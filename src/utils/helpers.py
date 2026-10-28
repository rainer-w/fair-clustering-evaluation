from src.evaluation.balance import balance_score
from src.evaluation.disco import disco_score
from src.evaluation.dcsi import dcsiscore

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

    balance_fair, clbal_fair, gbal_fair = balance_score("test",["sensitive_value"], fair_gt, sensitive, per_cluster=True)
    balance_unfair, clbal_unfair, gbal_unfair = balance_score("test", ["sensitive_value"], unfair_gt, sensitive, per_cluster=True)
    minbal_fair = (
        min(clbal_fair.values())
        if isinstance(clbal_fair, dict) and clbal_fair
        else 0
    )
    minbal_unfair = (
        min(clbal_unfair.values())
        if isinstance(clbal_unfair, dict) and clbal_unfair
        else 0
    )
    disco_fair = disco_score(features, fair_gt)
    disco_unfair = disco_score(features, unfair_gt)
    dcsi_fair = dcsiscore(features,fair_gt)
    dcsi_unfair = dcsiscore(features, unfair_gt)

    row = {
        "method" : "GroundTruth",
        "balance_fair" : balance_fair, 
        "minbal_fair" : minbal_fair,
        "balance_unfair" : balance_unfair,
        "minbal_unfair" : minbal_unfair, 
        "disco_fair" : disco_fair, 
        "disco_unfair" : disco_unfair,
        "dcsi_fair" : dcsi_fair, 
        "dcsi_unfair" : dcsi_unfair,
    }
    return row
def evaluate_groundtruth_separate(
        fairdegen
): 
    features = fairdegen.get_features_wo_sensitive()
    sensitive = fairdegen.get_sensitive()
    fair_gt = fairdegen.get_groundtruth()
    unfair_gt = fairdegen.get_unfair_groundtruth()
    balance_fair, clbal_fair, gbal_fair = balance_score("test",["sensitive_value"], fair_gt, sensitive, per_cluster=True)
    balance_unfair, clbal_unfair, gbal_unfair = balance_score("test", ["sensitive_value"], unfair_gt, sensitive, per_cluster=True)
    if isinstance(clbal_fair, dict):
        minbal_fair = min(clbal_fair.values())
        minbal_fair = (
            min(clbal_fair.values())
            if isinstance(clbal_fair, dict) and clbal_fair
            else 0
        )
    else: 
        minbal_fair = None
    if isinstance(clbal_fair, dict):
        minbal_unfair = (
            min(clbal_unfair.values())
            if isinstance(clbal_unfair, dict) and clbal_unfair
            else 0
        )
    else: 
        minbal_unfair = None

    disco_fair = disco_score(features, fair_gt)
    disco_unfair = disco_score(features, unfair_gt)
    dcsi_fair = dcsiscore(features,fair_gt)
    dcsi_unfair = dcsiscore(features, unfair_gt)
    row_fair = {
        "method" : "GT_Fair",
        "balance" : balance_fair, 
        "minbal" : minbal_fair,
        "disco" : disco_fair,
        "dcsi" : dcsi_fair,
    }
    row_unfair = {
        "method" : "GT_Unfair",
        "balance" : balance_unfair,
        "minbal" : minbal_unfair,
        "disco" : disco_unfair,
        "dcsi" : dcsi_unfair,
    }
    rows = [row_fair,row_unfair]
    return rows


def add_gt_deviations(df, gt_rows):
    gt_fair = next(row for row in gt_rows if row["method"] == "GT_Fair")
    gt_unfair = next(row for row in gt_rows if row["method"] == "GT_Unfair")

    df = df.copy()

    metrics = ["balance", "minbal", "disco", "dcsi"]

    for metric in metrics:
        # Only calculate deviations if the metric is available
        # in the results and both ground-truth rows.
        if (
            metric in df.columns
            and metric in gt_fair
            and metric in gt_unfair
        ):
            df[f"delta_{metric}_fair"] = (
                df[metric] - gt_fair[metric]
            )
            df[f"delta_{metric}_unfair"] = (
                df[metric] - gt_unfair[metric]
            )

    return df


def summarize_gt_deviations(df, gt_rows):
    gt_fair = next(row for row in gt_rows if row["method"] == "GT_Fair")
    gt_unfair = next(row for row in gt_rows if row["method"] == "GT_Unfair")

    df = df.copy()

    metrics = ["balance", "minbal", "disco", "dcsi"]
    available_metrics = [
        metric for metric in metrics
        if (
            metric in df.columns
            and metric in gt_fair
            and metric in gt_unfair
        )
    ]

    # Calculate deviations for available metrics.
    for metric in available_metrics:
        df[f"deviation_{metric}_fair"] = (
            df[metric] - gt_fair[metric]
        )
        df[f"deviation_{metric}_unfair"] = (
            df[metric] - gt_unfair[metric]
        )

    # Build aggregation dynamically so missing metrics are skipped.
    agg_dict = {}

    for metric in available_metrics:
        for reference in ["fair", "unfair"]:
            col = f"deviation_{metric}_{reference}"
            agg_dict[col] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, "std")

    if agg_dict:
        summary = (
            df.groupby("method")
            .agg(**agg_dict)
            .reset_index()
        )
    else:
        summary = df[["method"]].drop_duplicates().reset_index(drop=True)

    # Add the GT_Unfair vs GT_Fair difference for available metrics.
    gt_diff_row = {"method": "GT_Unfair"}

    for metric in available_metrics:
        gt_diff_row[f"deviation_{metric}_fair"] = (
            gt_unfair[metric] - gt_fair[metric]
        )

    gt_diff = pd.DataFrame([gt_diff_row])

    summary = pd.concat([summary, gt_diff], ignore_index=True)

    return summary


def OLD_add_gt_deviations(df, gt_rows):
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
    if "minbal" in df.columns:
        df["delta_minbal_fair"] = (
            df["minbal"] - gt_fair["minbal"]
        )
    df["delta_disco_fair"] = (
        df["disco"] - gt_fair["disco"]
    )
    if "dcsi" in df.columns:
        df["delta_dcsi_fair"] = (
            df["dcsi"] - gt_fair["dcsi"]
        )
    df["delta_balance_unfair"] = (
        df["balance"] - gt_unfair["balance"]
    )
    if "minbal" in df.columns:
        df["delta_minbal_unfair"] = (
            df["minbal"] - gt_unfair["minbal"]
        )
    df["delta_disco_unfair"] = (
        df["disco"] - gt_unfair["disco"]
    )
    if "dcsi" in df.columns:
        df["delta_dcsi_unfair"] = (
            df["dcsi"] - gt_unfair["dcsi"]
        )

    return df
def OLD_summarize_gt_deviations(df, gt_rows):
    gt_fair = next(
        row for row in gt_rows
        if row["method"] == "GT_Fair"
    )

    gt_unfair = next(
        row for row in gt_rows
        if row["method"] == "GT_Unfair"
    )

    df = df.copy()

    df["deviation_balance_fair"] = (
        df["balance"] - gt_fair["balance"]
    )
    if "minbal" in df.columns:
        df["deviation_minbal_fair"] = (
            df["minbal"] - gt_fair["minbal"]
        )
    df["deviation_disco_fair"] = (
        df["disco"] - gt_fair["disco"]
    )
    if "dcsi" in df.columns:
        df["deviation_dcsi_fair"] = (
            df["dcsi"] - gt_fair["dcsi"]
        )
    df["deviation_balance_unfair"] = (
        df["balance"] - gt_unfair["balance"]
    )
    if "minbal" in df.columns:
        df["deviation_minbal_unfair"] = (
            df["minbal"] - gt_unfair["minbal"]
        )
    df["deviation_disco_unfair"] = (
        df["disco"] - gt_unfair["disco"]
    )
    if "dcsi" in df.columns:
        df["deviation_dcsi_unfair"] = (
            df["dcsi"] - gt_unfair["dcsi"]
        )
    summary = (
        df.groupby("method")
        .agg(
            deviation_balance_fair=("deviation_balance_fair", "mean"),
            deviation_balance_fair_std=("deviation_balance_fair", "std"),

            deviation_minbal_fair=("deviation_minbal_fair", "mean"),
            deviation_minbal_fair_std=("deviation_minbal_fair", "std"),

            deviation_disco_fair=("deviation_disco_fair", "mean"),
            deviation_disco_fair_std=("deviation_disco_fair", "std"),

            deviation_dcsi_fair=("deviation_dcsi_fair", "mean"),
            deviation_dcsi_fair_std=("deviation_dcsi_fair", "std"),

            deviation_balance_unfair=("deviation_balance_unfair", "mean"),
            deviation_balance_unfair_std=("deviation_balance_unfair", "std"),

            deviation_minbal_unfair=("deviation_minbal_unfair", "mean"),
            deviation_minbal_unfair_std=("deviation_minbal_unfair", "std"),

            deviation_disco_unfair=("deviation_disco_unfair", "mean"),
            deviation_disco_unfair_std=("deviation_disco_unfair", "std"),
            
            deviation_dcsi_unfair=("deviation_dcsi_unfair", "mean"),
            deviation_dcsi_unfair_std=("deviation_dcsi_unfair", "std"),
        )
        .reset_index()
    )
    gt_diff = pd.DataFrame([{
        "method" : "GT_Unfair",
        "deviation_balance_fair" : gt_unfair["balance"] - gt_fair["balance"], 
        "deviation_minbal_fair" : gt_unfair["minbal"] - gt_fair["minbal"],
        "deviation_disco_fair" : gt_unfair["disco"] - gt_fair["disco"],
        "deviation_dcsi_fair" : gt_unfair["dcsi"] - gt_fair["dcsi"],
    }])
    summary = pd.concat(
        [summary,gt_diff], ignore_index=True
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
    balance, clbal, grpbal = balance_score("notneeded", ["sensitive_value"], y_pred, sensitive, per_cluster=True)
    minbal = (
        min(clbal.values())
        if isinstance(clbal, dict) and clbal
        else 0
    )

   # print("y_pred = ", y_pred)
    y_pred_arr = np.asarray(y_pred)
  #  print("y_pred_arr = ", y_pred_arr)
    disco = disco_score(X,y_pred)
    #print("Before dcsiscore:", type(X), getattr(X, "shape", None))

    dcsi = dcsiscore(X,y_pred)

    row = {
        "method": method,
        "ari": ari,
        "nmi": nmi,
        "disco": disco,
        "dcsi": dcsi,
        "balance": balance,
        "minbal" : minbal,
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