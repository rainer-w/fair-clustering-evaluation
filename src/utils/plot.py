import matplotlib.pyplot as plt

def plot_experiment_1(df, path, annotate = False):
    plt.rcParams.update({
        "font.size": 25,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 25,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22
        })
    fig, ax = plt.subplots(figsize=(14, 14))

    size = 200

    for _, row in df.iterrows():
        ax.plot(
            [row["disco_fair"], row["disco_unfair"]],
            [row["balance_fair"], row["balance_unfair"]],
            color="gray",
            alpha=0.4,
            linewidth=1.5,
            zorder=1
        )

    # fair points
    ax.scatter(
        df["disco_fair"],
        df["balance_fair"],
        s=size,
        color="green",
        marker="o",
        label="Fair",
        zorder=2
    )

    # unfair points
    ax.scatter(
        df["disco_unfair"],
        df["balance_unfair"],
        s=size,
        color="red",
        marker="X",
        label="Unfair",
        zorder=3
    )

    # annotate points with dim/k
    if annotate:
        for _, row in df.iterrows():

            ax.annotate(
                str(row["dim/k"]),
                (row["disco_fair"], row["balance_fair"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10
            )

            ax.annotate(
                str(row["dim/k"]),
                (row["disco_unfair"], row["balance_unfair"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10
            )

    ax.set_xlabel("DISCO")
    ax.set_ylabel("Balance")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"{path}FairVsUnfairLabels.pdf",
        bbox_inches="tight"
    )

    return fig
def plot_filtered_skyline(
    results_df,
    x,
    x_threshold=-1,
    balance_threshold=0,
    path="plots/",
    annotate=False, 
    xlim=None, 
    ylim=None,
    title_inline=None,
    y="balance"
):

    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 40,
        "axes.labelsize": 50,
        "axes.titlesize": 50,
        "legend.fontsize": 40,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40
    })


    fig, ax = plt.subplots(
        figsize=(14, 12),
        constrained_layout=True
    )

    filtered = results_df[
        (results_df[x] >= x_threshold) &
        (results_df[y] >= balance_threshold)
    ]

    method_order = [
            "KMeans", 
            "Fairlets", 
            "SFC",
            "DBSCAN", 
            "FairDen", 
            "HDBSCAN", 
            "FairSC",
            "GT_Fair",
            "GT_Unfair" 
        ]
    method_markers = {
        "DBSCAN": "^",
        "HDBSCAN": "v",
       "FairDen" : "D",
       "Fairlets" : "8",
       "SFC" : "H",
        "KMeans": "o",
        "FairSC": "s", 
        "GT_Fair" : "*", 
        "GT_Unfair" : "X"
    }


    method_colors = {
        "DBSCAN": "#9467bd",      
        "HDBSCAN": "#6a51a3",     
        "KMeans": "#1f77b4",      
        "FairDen": "#2ca02c",     
        "FairSC": "#66a61e",      
        "Fairlets": "tab:olive",    
        "SFC":       "#1b9e77",  
        "GT_Fair": "#666666",
        "GT_Unfair": "#bbbbbb",
    }
    for method in method_order:
        subdf = filtered[
            filtered["method"] == method
        ]
        if subdf.empty:
            continue

        ax.scatter(
            subdf[x],
            subdf[y],
            label=method,
            alpha=0.8,
            s=700,          
            color=method_colors[method],
            marker=method_markers[method]
        )


        if annotate:
            for _, row in subdf.iterrows():
                ax.annotate(
                    str(row["n_clusters"]),
                    (row[x], row["balance"]),
                    fontsize=30
                )

    metric_to_label = {"disco":"DISCO", "n_clusters":"N Clusters", "balance":"Balance",
                       "deviation_disco_fair":"$\Delta DISCO$", "deviation_balance_fair":"$\Delta Balance$"}
    xlabel = metric_to_label[x]
    ax.set_xlabel(rf"\texttt{{{xlabel}}}")
    ylabel = metric_to_label[y]
    ax.set_ylabel(rf'\texttt{{{ylabel}}}')
   # ax.set_xlabel(x)
    if xlim is not None: 
        ax.set_xlim(xlim)
    if ylim is not None: 
        ax.set_ylim(ylim)


    
    handles, labels = ax.get_legend_handles_labels()

   # fig_legend = plt.figure(figsize=(10,2))
    # --
    handle_dict = dict(zip(labels,handles))
    ordered_handles = [handle_dict[m] for m in method_order if m in handle_dict]
    ordered_labels = [m for m in method_order if m in handle_dict]
    # --

    fig_legend = plt.figure(figsize=(2,1))
    # fig_legend.legend(ordered_handles, ordered_labels, loc="center", ncol=2, frameon=False)
    fig_legend.legend(
        ordered_handles,
        ordered_labels,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=16,
        markerscale=0.25,
        handlelength=1,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.2,
    )


    fig_legend.savefig(
        f"{path}legend_skyline.pdf",
        bbox_inches="tight"
    )

    plt.close(fig_legend)
    if title_inline: 
        ax.text(
            0.02, 0.02, 
            title_inline, 
            transform=ax.transAxes,
            ha="left",
            va= "bottom"
        )

    fig.savefig(
        f"{path}skyline.pdf",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    return fig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_line(
    results,
    x,
    path,
    tick_labels=None,
    xlabel=None,
    categorical=False, 
    groundtruth_results = None, 
    include_std = False, 
    logaxis = None, # can be "x", "y", "xy"
    x_filter = None
):

    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 40,
        "axes.labelsize": 50,
        "axes.titlesize": 50,
        "legend.fontsize": 40,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40
    })
    method_order = [
        "GT_Fair", 
        "GT_Unfair",
        "DBSCAN", 
        "HDBSCAN", 
        "KMeans", 
        "FairDen", 
        "FairSC", 
        "Fairlets", 
        "SFC",
    ]
    method_markers = {
        "DBSCAN": "^",
        "HDBSCAN": "v",
       "FairDen" : "D",
       "Fairlets" : "8",
       "SFC" : "H",
        "KMeans": "o",
        "FairSC": "s", 
        "GT_Fair" : "*", 
        "GT_Unfair" : "X"
    }

    method_linestyles = {
        "DBSCAN": "-",
        "HDBSCAN": "--",
        "FairDen": ":",
        "KMeans": "-.",
        "Fairlets": (0, (4, 1,  1)),
        "SFC" : "-",
        "FairSC": (0, (3, 1, 1, 1)),
        "GT_Fair" : None, 
        "GT_Unfair" : None
    }

    method_colors = {
        "DBSCAN": "#9467bd",
        "HDBSCAN": "#6a51a3",
        "KMeans": "#1f77b4",
        "FairDen": "#2ca02c",
        "FairSC": "#66a61e",
        "Fairlets": "tab:olive",
        "SFC":       "#1b9e77", 
        "GT_Fair":  "#666666",
        "GT_Unfair": "#bbbbbb",
    }
    results = results.copy()

    if x_filter is not None:
        results = results[results[x].isin(x_filter)]

        if groundtruth_results is not None:
            groundtruth_results = groundtruth_results[
                groundtruth_results[x].isin(x_filter)
            ]
    #define ordering
    x_values = sorted(results[x].unique())

    # map x → index if categorical mode
    if categorical:
        x_map = {v: i for i, v in enumerate(x_values)}
    else:
        x_map = None

    if xlabel is None:
        xlabel = x

    for criterion in ["disco", "score"]:
        for metric in [ "runtime", "disco","balance"]:

            fig, ax = plt.subplots(figsize=(14, 12), constrained_layout = True)
            plt_map = {
                None : ax.plot, 
                "x" : ax.semilogx, 
                "y" : ax.semilogy, 
                "xy" : ax.loglog
            }
            custom_plot = plt_map[logaxis]
            if metric == "runtime": 
                custom_plot = plt_map["y"]
            #for method in results["method"].unique():
            for method in method_order:
                if method == "GroundTruth" or method not in results["method"].values: #
                    continue
                subdf = results[
                    (results["method"] == method) &
                    (results["criterion"] == criterion)
                ].copy()

                subdf = subdf.sort_values(x)

                if categorical:
                    xs = subdf[x].map(x_map).values
                else:
                    xs = subdf[x].values

                ys = subdf[metric].values
                custom_plot(
                    xs,
                    ys,
                    label=method,
                    marker=method_markers[method],
                    linestyle=method_linestyles[method],
                    color=method_colors[method],
                    markersize=32,
                    linewidth=3,
                    alpha=0.9
                )
                if include_std:
                    std = subdf[f"{metric}_std"].values
                    # variance band
                    ax.fill_between(
                        xs,
                        ys - std,
                        ys + std,
                        color=method_colors[method],
                        alpha=0.15
                    )
               # print(xs)
              #  print("gt results ", groundtruth_results)
            if groundtruth_results is not None and metric in ["disco","balance"]: #metric == "balance": 
                custom_plot(
                    xs, 
                    groundtruth_results[f"{metric}_fair"],
                    marker=method_markers["GT_Fair"],
                    label="GT_Fair",
                    linestyle=method_linestyles["GT_Fair"],
                    color=method_colors["GT_Fair"],
                    markersize=32,
                    linewidth=3,
                    alpha=0.7
                )
                if include_std:
                    std = groundtruth_results[f"{metric}_fair_std"].values
                    # variance band
                    ax.fill_between(
                        xs,
                        groundtruth_results[f"{metric}_fair"]- std,
                        groundtruth_results[f"{metric}_fair"] + std,
                        color=method_colors[method],
                        alpha=0.15
                    )
                custom_plot(
                    xs, 
                    groundtruth_results[f"{metric}_unfair"],
                    marker=method_markers["GT_Unfair"],
                    label="GT_Unfair",
                    linestyle=method_linestyles["GT_Unfair"],
                    color=method_colors["GT_Unfair"],
                    markersize=32,
                    linewidth=3,
                    alpha=0.7
                )
                if include_std:
                    std = groundtruth_results[f"{metric}_unfair_std"].values
                    # variance band
                    ax.fill_between(
                        xs,
                        groundtruth_results[f"{metric}_unfair"]- std,
                        groundtruth_results[f"{metric}_unfair"] + std,
                        color=method_colors[method],
                        alpha=0.15
                    )

            # ---- axis labels ----
            metric_to_label = {"disco":"DISCO", "balance":"Balance", "runtime":"Runtime (s)"}
            ax.set_xlabel(rf"\texttt{{{xlabel}}}")
            metric_label = metric_to_label[metric]
            ax.set_ylabel(rf'\texttt{{{metric_label}}}')
            #print("debug print: ")
           # print(sorted(results["distr_index"].unique()))
          #  print(len(sorted(results["distr_index"].unique())))
  
            if categorical:
                ax.set_xticks(range(len(x_values)))

                if tick_labels is not None:
                    ax.set_xticklabels(tick_labels)
                else:
                    ax.set_xticklabels([str(v) for v in x_values]) # ha = "right"?

            else:
                if tick_labels is not None: 
                    ax.set_xticks(tick_labels)
                else:
                    ax.set_xticks(x_values)
        #    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
           # ax.legend(loc="upper center",bbox_to_anchor=(0.5, -0.15),frameon=False)
            handles, labels = ax.get_legend_handles_labels()

            # --
            handle_dict = dict(zip(labels,handles))
            ordered_handles = [handle_dict[m] for m in method_order if m in handle_dict]
            ordered_labels = [m for m in method_order if m in handle_dict]


            # --

            fig_legend = plt.figure(figsize=(2,1))
           # fig_legend.legend(ordered_handles, ordered_labels, loc="center", ncol=2, frameon=False)
            fig_legend.legend(
                ordered_handles,
                ordered_labels,
                loc="center",
                ncol=5,
                frameon=False,
                fontsize=16,
                markerscale=0.25,
                handlelength=1,
                handletextpad=0.4,
                columnspacing=1.0,
                labelspacing=0.2,
            )

            fig_legend.canvas.draw()

            fig_legend.savefig(f"{path}legend.pdf", bbox_inches="tight")
            plt.close(fig_legend)
            plt.savefig(
                f"{path}{metric}_for_opt_{criterion}.pdf",
                bbox_inches=None
            )
            plt.close()
from sklearn.decomposition import PCA
def _project_if_needed(X):

    if X.shape[1] == 2:
        return X, None

    pca = PCA(n_components=2)
    return pca.fit_transform(X), pca

def _compute_cluster_centers(X, labels):
    centers = {}

    for c in np.unique(labels):
        if c == -1:
            continue

        pts = X[labels == c]
        centers[c] = pts.mean(axis=0)

    return centers
import numpy as np
from matplotlib.patches import Ellipse
def plot_fair_clusters(
    X,
    semantic_cluster,
    subgroup_ids,
    sensitive_attribute,
    subgroup_type,
    cores=None,
    core_subgroups=None,
    title="Fair Density-Based Clusters",
    figsize=(14,12),
    alpha_points=0.6,
    show_semantic_outlines=False,
    show_core_paths=True,
    path=""
):

    plt.rcParams.update({
        "font.size": 40,
        "axes.labelsize": 50,
        "axes.titlesize": 50,
        "legend.fontsize": 40,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40
    })


    fig, ax = plt.subplots(
        figsize=figsize,
        constrained_layout=True
    )


    Xp, pca = _project_if_needed(X)

    # CORE VISUALIZATION
    # =========================

    if cores is not None:
        core_cmap = plt.cm.viridis
        unique_types = []
        for sg in subgroup_type:
            t = subgroup_type[sg]
            if t not in unique_types:
                unique_types.append(t)

        type_to_color = {}
        for i, t in enumerate(unique_types):
            type_to_color[t] = core_cmap(
                i / max(1, len(unique_types)-1)
            )

        added_labels = set()
        for clu in cores:
            if clu == -1:
                continue
            core_pts = np.asarray(cores[clu])

            if pca is not None:
                core_pts = pca.transform(core_pts)
            subgroup_assign = np.asarray(
                core_subgroups[clu]
            )
            # core trajectories
            if show_core_paths and len(core_pts) > 1:
                ax.plot(
                    core_pts[:,0],
                    core_pts[:,1],
                    color="black",
                    linewidth=2,
                    alpha=0.5
                )
            # core points
            for i, pt in enumerate(core_pts):
                sg = subgroup_assign[i]
                if sg not in subgroup_type.keys():
                    continue
                t = subgroup_type[sg]
                color = type_to_color[t]
                label = None
                if t not in added_labels:
                    label = f"core type={t}"
                    added_labels.add(t)

                ax.scatter(
                    pt[0],
                    pt[1],
                    color=color,
                  #  edgecolor="black",
                    marker="o",
                    s=3200,
                    linewidth=1.5,
                    label=label,
                    alpha=0.8
                )
    # NORMAL DATA POINTS
    # =========================

    point_cmap = plt.cm.coolwarm
    unique_sens = np.unique(
        sensitive_attribute
    )
    sens_colors = point_cmap(
        np.linspace(
            0.1,
            0.9,
            len(unique_sens)
        )
    )
    markers = {
        0:"o",
        1:"^",
        2:"D",
        3:"v",
        4:"s",
        5:"P",
        6:"h"
    }

    for idx, s in enumerate(unique_sens):
        mask = sensitive_attribute == s
        ax.scatter(
            Xp[mask,0],
            Xp[mask,1],
            color=sens_colors[idx],
            alpha=alpha_points,
            s=700,          # increased marker size
            label=f"sensitive={s}",
            marker=markers[s]
        )

    # SEMANTIC CLUSTER OUTLINES
    # =========================
    if show_semantic_outlines:
        centers = _compute_cluster_centers(
            Xp,
            semantic_cluster
        )
        for clu in centers:
            pts = Xp[
                semantic_cluster == clu
            ]
            center = centers[clu]
            cov = np.cov(
                pts.T
            )
            eigvals, eigvecs = np.linalg.eigh(
                cov
            )
            order = eigvals.argsort()[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:,order]
            angle = np.degrees(
                np.arctan2(
                    *eigvecs[:,0][::-1]
                )
            )
            width, height = (
                4*np.sqrt(eigvals)
            )
            ellipse = Ellipse(
                xy=center,
                width=width,
                height=height,
                angle=angle,
                edgecolor="black",
                facecolor="none",
                linestyle="--",
                linewidth=3,
                alpha=0.5
            )

            ax.add_patch(
                ellipse
            )

            ax.text(
                center[0],
                center[1],
                f"C{clu}",
                fontsize=35,
                weight="bold"
            )
    # STYLE
    # =========================

    ax.set_title("")

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_box_aspect(None)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # EXTERNAL LEGEND
    # =========================

    handles, labels = ax.get_legend_handles_labels()
    fig_legend = plt.figure(
        figsize=(2,1)
    )
    fig_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=16,
        markerscale=0.25,
        handlelength=1,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.2,
    )


    fig_legend.savefig(
        f"{path}legend_data.pdf",
        bbox_inches="tight"
    )

    plt.close(fig_legend)
    fig.savefig(
        f"{path}fair_plot.pdf",
        bbox_inches="tight", dpi = 300
    )


    plt.close(fig)


    return fig


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def plot_cluster_labels(
    Xp,
    labels,
    sensitive=None,
    path="",
    filename="cluster_labels.pdf",
    title="Cluster visualization",
    figsize=(14, 14),
    alpha_points=0.85,
    show_hulls=False,
    hull_alpha=0.15,
    hull_edgecolor="black",
    cmap_name="tab20",
    show_legend=True,
    s=256,
    noise_label=-1,
):

    labels = np.asarray(labels)
    Xp = np.asarray(Xp)

    if sensitive is not None:
        sensitive = np.asarray(sensitive)


    plt.rcParams.update({
        "font.size": 25,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "legend.fontsize": 25,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22
    })

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap(cmap_name)

    colors = {
        lab: cmap(i / max(1, len(unique_labels) - 1))
        for i, lab in enumerate(unique_labels)
    }

    def get_marker_map(sensitive):
        sens_vals = np.unique(sensitive)
        if len(sens_vals) == 1:
            return {sens_vals[0]: "o"}
        if len(sens_vals) == 2:
            return {sens_vals[0]: "o", sens_vals[1]: "^"}
        markers = ["o", "^", "s", "D", "P", "X"]
        return {sv: markers[i % len(markers)] for i, sv in enumerate(sens_vals)}

    marker_map = None
    if sensitive is not None:
        marker_map = get_marker_map(sensitive)

    for lab in unique_labels:
        mask = labels == lab
        pts = Xp[mask]

        if len(pts) == 0:
            continue

        cluster_color = colors[lab]

        if lab == noise_label:
            if sensitive is not None:
                for sv in np.unique(sensitive):
                    smask = mask & (sensitive == sv)

                    ax.scatter(
                        Xp[smask, 0],
                        Xp[smask, 1],
                        c="lightgray",
                        s=s * 0.85,
                        alpha=0.5,
                        marker=marker_map.get(sv, "x"),
                        label=f"noise, s={sv}" if show_legend else None
                    )
            else:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    c="lightgray",
                    s=s * 0.85,
                    alpha=0.5,
                    marker="x",
                    label="noise" if show_legend else None
                )
            continue


        if sensitive is not None:
            for sv in np.unique(sensitive):
                smask = mask & (sensitive == sv)

                ax.scatter(
                    Xp[smask, 0],
                    Xp[smask, 1],
                    c=[cluster_color],
                    s=s,
                    alpha=alpha_points,
                    marker=marker_map.get(sv, "o"),
                    edgecolor="black",
                    linewidth=0.3,
                    label=f"c{lab}, s={sv}" if show_legend else None
                )
        else:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=[cluster_color],
                s=s,
                alpha=alpha_points,
                edgecolor="black",
                linewidth=0.3,
                label=f"cluster {lab}" if show_legend else None
            )

        if show_hulls and len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]

                ax.fill(
                    hull_pts[:, 0],
                    hull_pts[:, 1],
                    color=cluster_color,
                    alpha=hull_alpha,
                    edgecolor=hull_edgecolor,
                    linewidth=1.2
                )
            except Exception:
                pass

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_frame_on(False)

    if show_legend:
        ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(f"{path}{filename}", dpi=300, bbox_inches="tight")
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr, spearmanr

sens_attr_markers = {
        "sex" : "v", "race":"^", "marital":"<", "age":">"
    }
sens_colors = {
    "sex": "#1f77b4",
    "race": "#d62728",
    "marital": "#2ca02c",
    "age": "#9467bd",
    }
def plot_feature_sensitive_correlations(
    X,
    sensitive,
    feature_names=None,
    save_path=None,
    title_inline = None
):

    plt.rcParams.update({
        "font.size": 40,
        "axes.labelsize": 50,
        "axes.titlesize": 50,
        "legend.fontsize": 40,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40
    })

    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)

    if isinstance(sensitive, pd.Series):
        sensitive = sensitive.to_frame()

    elif isinstance(sensitive, np.ndarray):
        if sensitive.ndim == 1:
            sensitive = pd.DataFrame({"Sensitive": sensitive})
        else:
            sensitive = pd.DataFrame(
                sensitive,
                columns=[f"S{i}" for i in range(sensitive.shape[1])]
            )

    # Pearson
    # ===============================

    pearson = pd.DataFrame(
        index=X.columns,
        columns=sensitive.columns,
        dtype=float
    )

    for s in sensitive.columns:
        y = sensitive[s]

        for f in X.columns:
            try:
                r, _ = pearsonr(X[f], y)
            except Exception:
                r = np.nan

            pearson.loc[f, s] = r

    fig, ax = plt.subplots(figsize=(14, 12))

    x = np.arange(len(X.columns))
    width = 0.8 / len(sensitive.columns)

    for i, s in enumerate(sensitive.columns):
        ax.plot(
            x,
            pearson[s].values,
            marker=sens_attr_markers[s],
            linewidth=3,
            markersize=32,
            label=s,
            color = sens_colors[s]
        )

    ax.set_xticks(x + width * (len(sensitive.columns) - 1) / 2)
    ax.set_xticklabels(X.columns, rotation=45, ha="right")
    ax.set_ylabel("Pearson correlation")
    ax.set_ylim(-1, 1)
    ax.axhline(0, linewidth=1)
   # ax.set_title("Feature ↔ Sensitive Attribute (Pearson)")
   # ax.legend()
    if title_inline:
        ax.text(
            0.02, 0.98, 
            title_inline, 
            transform=ax.transAxes,
            ha="left",
            va= "top"
        )


    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/pearson.pdf", bbox_inches=None)
        save_separate_legend(fig, ax, save_path, "legend.pdf")

  #  plt.show()

    # Spearman
    # ===============================

    spearman = pd.DataFrame(
        index=X.columns,
        columns=sensitive.columns,
        dtype=float
    )

    for s in sensitive.columns:
        y = sensitive[s]

        for f in X.columns:
            try:
                r, _ = spearmanr(X[f], y)
            except Exception:
                r = np.nan

            spearman.loc[f, s] = r

    fig, ax = plt.subplots(figsize=(14, 12))

    for i, s in enumerate(sensitive.columns):
        ax.plot(
            x,
            spearman[s].values,
            marker=sens_attr_markers[s],
            linewidth=3,
            markersize=32,
            label=s,
            color = sens_colors[s]
        )

    ax.set_xticks(x + width * (len(sensitive.columns) - 1) / 2)
    ax.set_xticklabels(X.columns, rotation=45, ha="right")
    ax.set_ylabel("Spearman correlation")
    ax.set_ylim(-1, 1)
    ax.axhline(0, linewidth=1)
    #ax.set_title("Feature ↔ Sensitive Attribute (Spearman)")
   # ax.legend()
    if title_inline:
        ax.text(
            0.02, 0.98, 
            title_inline, 
            transform=ax.transAxes,
            ha="left",
            va= "top"
        )
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/spearman.pdf", bbox_inches=None)

   # plt.show()

    # Mutual Information
    # ===============================

    mi = pd.DataFrame(
        index=X.columns,
        columns=sensitive.columns,
        dtype=float
    )

    for s in sensitive.columns:

        values = mutual_info_classif(
            X,
            sensitive[s],
            random_state=0
        )

        mi[s] = values

    fig, ax = plt.subplots(figsize=(14, 12))

    for i, s in enumerate(sensitive.columns):
        ax.plot(
            x,
            mi[s].values,
            marker=sens_attr_markers[s],
            linewidth=3,
            markersize=32,
            label=s,
            color = sens_colors[s]
        )

    ax.set_xticks(x + width * (len(sensitive.columns) - 1) / 2)
    ax.set_xticklabels(X.columns, rotation=45, ha="right")
    ax.set_ylabel("Mutual Information")
    ax.set_ylim(0, mi.max().max() * 1.1)
    if title_inline:
        ax.text(
            0.02, 0.98, 
            title_inline, 
            transform=ax.transAxes,
            ha="left",
            va= "top"
        )
    #ax.set_title("Feature ↔ Sensitive Attribute (Mutual Information)")
   # ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/mutual_information.pdf", bbox_inches=None)

   # plt.show()
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=sens_colors["sex"], marker="v", linewidth=3, label="Sex", markersize=32),
        Line2D([], [], color=sens_colors["race"], marker="^", linewidth=3, label="Race", markersize=32),
        Line2D([], [], color=sens_colors["marital"], marker="<", linewidth=3, label="Marital status", markersize=32),
        Line2D([], [], color=sens_colors["age"], marker=">", linewidth=3, label="Age", markersize=32),
        ]

    fig = plt.figure(figsize=(4, 2))
    fig.legend(handles=handles,
            loc="center",
            ncol=4,
            frameon=False)

    fig.savefig(f"{save_path}/legend_all_sensitive.pdf", bbox_inches="tight")
    plt.close(fig)

    return pearson, spearman, mi
def save_separate_legend(fig, ax, path, filename="legend.pdf"):
    handles, labels = ax.get_legend_handles_labels()

    fig_legend = plt.figure(figsize=(4, 2))
    fig_legend.legend(
        handles,
        labels,
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=16
    )

    fig_legend.savefig(
        f"{path}/{filename}",
        bbox_inches="tight"
    )
    plt.close(fig_legend)

