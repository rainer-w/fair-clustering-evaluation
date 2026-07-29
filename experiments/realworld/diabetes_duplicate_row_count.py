from src.utils.plot import plot_feature_sensitive_correlations
from src.utils.load_uci import load_diabetes, load_adult, load_bank, load_census, load_creditcard
import pandas as pd
from pathlib import Path
dataset_map = {
    "Adult" : load_adult, 
    "Bank": load_bank, 
    "Census":load_census, 
    "Creditcard": load_creditcard,
    "Diabetes" : load_diabetes
    
}
def main_diabetes_duplicates(run=True,plot=True):
    run = True
    plot = True
    
    if run:
        for dataset_name in ["Diabetes"]:
            #base_path=f"results/correlations/{dataset_name}/"
           # Path(base_path).mkdir(parents=True,exist_ok=True)
            X_w, X_wo, sens, y_true = dataset_map[dataset_name]()
            print(X_wo["time_in_hospital"].unique(), "unq hos time vals")
            print(X_wo["age"].unique(), "unq age vals")
            n_total = len(X_wo)
            n_unique = X_wo.drop_duplicates().shape[0]

            print(f"{n_unique}/{n_total} unique ({100*n_unique/n_total:.1f}%)")
            print("\nMost common feature vectors:")
            print(X_wo.value_counts().head(10))
            
if __name__ == "__main__": 
    main_diabetes_duplicates()