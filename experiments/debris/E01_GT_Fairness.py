
from src.generators.debris import DEBRIS
from pathlib import Path
from src.evaluation.balance import balance_score
from src.evaluation.disco import disco_score
import pandas as pd
from src.utils.plot import plot_experiment_1

def run_1(path,dim,clunum,seed,core_num,ratio_noise,g,distr, n):
    
    fairdegen = DEBRIS(
        dim = dim, 
        clunum=clunum,
        seed=seed,
        core_num= core_num, 
        ratio_noise = ratio_noise,
        g=g,
        distr = distr
	)
    fairdata = fairdegen.generate_data(n)
    features = fairdegen.get_features_wo_sensitive()
    sensitive = fairdegen.get_sensitive()
    fair_gt = fairdegen.get_groundtruth()
    unfair_gt = fairdegen.get_unfair_groundtruth()

    balance_fair = balance_score("test",["sensitive_value"], fair_gt, sensitive)
    balance_unfair = balance_score("test", ["sensitive_value"], unfair_gt, sensitive)
    disco_fair = disco_score(features, fair_gt)
    disco_unfair = disco_score(features, unfair_gt)
    row = {
            "balance_fair":balance_fair, 
            "balance_unfair":balance_unfair, 
            "disco_fair": disco_fair, 
            "disco_unfair":disco_unfair, 
            "seed":seed, 
            "dim/k" : dim
        }
    return row
def main_1():
    
    gtodistr = {
        2 : [[0.9,0.1], [0.1,0.9]] , 
        3 : [ [0.7, 0.2, 0.1], [0.5,0.2,0.3], [0.6,0.2,0.2] ]
    }
    rows = []
    for seed in [11,22,33,44,55]: 
        for dim_clunum in [10,20,50,100]: 
            for distr in [
                [[0.5,0.5], [0.5,0.5]],
                [[0.9,0.1], [0.1,0.9]],
                [[0.8,0.2], [0.8,0.2]], 
                [[0.1,0.9], [0.3,0.7]] 
            ]:
                g = 2 
               # distr = gtodistr[g]
                dim = dim_clunum
                clunum = dim_clunum
                path = f"results/debris/Experiment1/seed{seed}_dim{dim}_clunum{clunum}/"
            # Path(path).mkdir(parents=True,exist_ok=True)
                
                row = run_1(path,dim,clunum,seed,[15]*clunum,0.0,g,distr,n=8000)
                rows.append(row)
    results = pd.DataFrame(rows)
    
    path = "results/debris/Experiment1/"
    Path(path).mkdir(parents=True,exist_ok=True)
    results.to_csv(f"{path}results.csv")
    
    plot_experiment_1(results, path)

if __name__ == "__main__":
    main_1()