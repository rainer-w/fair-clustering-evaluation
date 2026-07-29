import pandas as pd
from src.utils.helpers import aggregate_over_seeds

base_path = "results/debris/Experiment2/"
seeds = [11,22,33,44,55]
group_cols= ["clunum", "method", "criterion"]
metrics=["balance", "disco", "ari","n_clusters"]
avg_results, gt_results = aggregate_over_seeds(base_path,seeds,group_cols,metrics)
print(avg_results)
print(gt_results)

latex_df = avg_results.copy()


for metric in metrics:
    latex_df[metric] = (
        latex_df[metric].map(lambda x: f"{x:.3f}")
        + r" $\pm$ "
        + latex_df[f"{metric}_std"].map(lambda x: f"{x:.3f}")
    )


latex_df = latex_df.drop(columns=[f"{m}_std" for m in metrics])

print(latex_df.to_latex(index=False, escape=False))