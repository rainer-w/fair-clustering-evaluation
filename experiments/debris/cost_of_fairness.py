

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Catppuccin Mocha palette (https://catppuccin.com/palette)

CATPPUCCIN_MOCHA = {
    "green": "#40a02b",
    "teal": "#179299",
    "sapphire": "#209fb5",
    "blue": "#1e66f5",
    "lavender": "#7287fd",
}
CATPPUCCIN_HEADROOM_CMAP = LinearSegmentedColormap.from_list(
    "catppuccin_headroom",
    [
        CATPPUCCIN_MOCHA["green"],
        CATPPUCCIN_MOCHA["teal"],
        CATPPUCCIN_MOCHA["sapphire"],
        CATPPUCCIN_MOCHA["blue"],
        CATPPUCCIN_MOCHA["lavender"],
    ],
)



def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["disco_loss"] = df["disco_unfair"] - df["disco_fair"]
    df["delta_balance"] = df["balance_fair"] - df["balance_unfair"]
    df["headroom"] = 1 - df["balance_unfair"]
    return df


def plot_cost_of_fairness(
    df: pd.DataFrame,
    out_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    scatter = ax.scatter(
        df["delta_balance"].abs(),
        df["disco_loss"].abs(),
        c=df["headroom"],
        cmap=CATPPUCCIN_HEADROOM_CMAP,
        s=60,
        alpha=0.8,
        edgecolors="none",
    )

    ax.set_xlabel(r"$|\Delta \mathrm{Balance}|$")
    ax.set_ylabel(r"$|\Delta \mathrm{DISCO}|$")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Achievable Improvement (1-$\text{Balance}_{\text{unfair}}$)")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main_CoF():
    csv_path = "results/debris/Experiment1/results.csv"
    output = "results/debris/Experiment1/cost_of_fairness.pdf"


    df = pd.read_csv(csv_path, index_col=0)
    df = compute_metrics(df)

    print(
        df[
            [
                "balance_fair",
                "balance_unfair",
                "disco_fair",
                "disco_unfair",
                "delta_balance",
                "disco_loss",
                "headroom",
            ]
        ].describe()
    )

    plot_cost_of_fairness(df, out_path=output)


if __name__ == "__main__":
    main_CoF()

