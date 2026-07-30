from experiments.debris.E01_GT_Fairness import main_1
from experiments.debris.E02_Varying_K import main_2
from experiments.debris.E03_Varying_Distr import main_3
from experiments.debris.E04_Varying_Dim import main_4
from experiments.debris.E05_Varying_Noise import main_5
from experiments.debris.E06_Varying_G import main_6
from experiments.debris.E07_Varying_Imbalance import main_7
from experiments.debris.E08_Qualitative import main_8
from experiments.debris.E09_Varying_N import main_9
from experiments.debris.E10_Varying_Sizes import main_10
from experiments.debris.E11_Varying_Densities import main_11
from experiments.debris.E12_Varying_NumGap import main_12

exp_map = {
    "1 GT Fairness": main_1, 
    "2 Varying Number of Clusters":main_2,
    "3 Varying Distribution":main_3, 
    "4 Varying Dimensions":main_4,
    "5 Varying Ratio of Noise":main_5, 
    "6 Varying Number of sensitive Groups":main_6,
    "7 Varying Imbalance between subgroups":main_7, 
    "8 Qualitative":main_8,
    "9 Varying Number of Samples":main_9,
    "10 Varying Sizes of subgroups":main_10,
    "Varying Densities of Clusters":main_11,
    "12 Varying Number of Gap-Cores between subgroups":main_12
}
if __name__ == "__main__":
    for name, experiment in exp_map.items():
        print("=" * 60)
        print(f"Running Experiment: {name}")
        print("=" * 60)

        try:
            experiment()
        except Exception as e:
            print(f"Experiment '{name}' failed:")
            print(e)