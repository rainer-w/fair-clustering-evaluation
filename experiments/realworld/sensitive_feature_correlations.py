from src.utils.plot import plot_feature_sensitive_correlations
from src.utils.load_uci import load_diabetes, load_adult, load_bank, load_census, load_creditcard
import pandas as pd
from pathlib import Path
from experiments.realworld.sensitive_feature_correlations_plot import main_corr_legend_plot
dataset_map = {
    "Adult" : load_adult, 
    "Bank": load_bank, 
    "Census":load_census, 
    "Creditcard": load_creditcard,
    "Diabetes" : load_diabetes
    
}

def main_feat_corr(run=True):
    run = True

    if run:
        for dataset_name in ["Adult","Bank","Census", "Creditcard","Diabetes"]:
            print("running ", dataset_name)
            base_path=f"results/realworld/correlations/{dataset_name}/"
            Path(base_path).mkdir(parents=True,exist_ok=True)
            X_w, X_wo, sens, y_true = dataset_map[dataset_name]()
            pearson, spearman, mi = plot_feature_sensitive_correlations(
                X_wo,
                sens,
                save_path=f"{base_path}", 
                title_inline= dataset_name
            )
            pearson.to_csv(f"{base_path}pearson.csv")
            spearman.to_csv(f"{base_path}spearman.csv")
            mi.to_csv(f"{base_path}mutual_information.csv")
        main_corr_legend_plot()
if __name__ == "__main__": 
    main_feat_corr()