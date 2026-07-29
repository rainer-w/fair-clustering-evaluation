import pandas as pd

def main_table_uci():
    datasets = ["Adult", "Bank", "Census", "Creditcard", "Diabetes"]

    unfair_methods = ["DBSCAN", "HDBSCAN", "KMeans"]
    fair_methods = ["FairDen", "FairSC", "SFC"]

    latex = r"""\begin{table*}[ht!]
    \caption{Mean balance $\pm$ standard deviation for aware/unaware settings across five random subsets of UCI datasets. The mean and standard deviation are taken over the optimal solutions of the parameter search on each subset. Fairness-aware settings exclude the sensitive attribute from the feature space and instead use it only within the fairness mechanism, whereas fairness-unaware settings include the sensitive attribute as an ordinary feature.}
    \centering
    \begin{tabularx}{\textwidth}{l|rr|rr|rr|rr|rr}
    Method"""

    # ----------------------------
    # dataset header
    # ----------------------------
    for i, d in enumerate(datasets):
        if i < len(datasets) - 1:
            latex += f" & \\multicolumn{{2}}{{c|}}{{{d}}}"
        else:
            latex += f" & \\multicolumn{{2}}{{c}}{{{d}}}"

    latex += r"""\\
    """

    # ----------------------------
    # aware / unaware header
    # ----------------------------
    latex += " "
    for _ in datasets:
        latex += " & Aware & Unaware"
    latex += r"""\\ \hline
    """

    # ======================================================
    # Vanilla methods
    # ======================================================

    for method in unfair_methods:

        row = method

        for dataset in datasets:

            df = pd.read_csv(f"results/realworld/{dataset}/mean_std_results.csv")

            aware = df[
                (df["criterion"] == "score")
                & (df["method"] == method)
            ]

            unaware = df[
                (df["criterion"] == "disco")
                & (df["method"] == method)
            ]

            if aware.empty or unaware.empty:
                row += " & -- & --"
                continue

            aware_str = (
                f"${aware['balance'].iloc[0]:.2f}\\pm{aware['balance_std'].iloc[0]:.2f}$"
            )

            unaware_str = (
                f"${unaware['balance'].iloc[0]:.2f}\\pm{unaware['balance_std'].iloc[0]:.2f}$"
            )

            row += f" & {aware_str} & {unaware_str}"

        latex += row + r"""\\
    """

    latex += r"""\hline
    """

    # ======================================================
    # Fair methods
    # ======================================================

    for method in fair_methods:

        row = method

        for dataset in datasets:

            df = pd.read_csv(f"results/realworld/{dataset}/mean_std_results.csv")

            aware = df[
                (df["criterion"] == "score")
                & (df["method"] == method)
            ]

            if aware.empty:
                row += " & -- & --"
                continue

            aware_str = (
                f"${aware['balance'].iloc[0]:.2f}\\pm{aware['balance_std'].iloc[0]:.2f}$"
            )

            # no unaware setting
            row += f" & {aware_str} & {{}}"

        latex += row + r"""\\
    """

    latex += r"""\end{tabularx}
    \label{tab:aware-unaware}
    \end{table*}
    """

    print(latex)

if __name__ == "__main__":
    main_table_uci()