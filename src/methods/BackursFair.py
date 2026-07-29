# Adaptation of Backurs et al. implementation
# does not require matlab and/or python<=3.6

import numpy as np
from sklearn.metrics import pairwise_distances
import kmedoids

from src.methods.backurs_fairlet import (
    TreeNode,
    build_quadtree,
    tree_fairlet_decomposition,
    FAIRLETS,
    FAIRLET_CENTERS
)


class BackursFairClustering:
    """
    Scalable fair clustering based on Backurs et al.
    Fairlet decomposition + k-median clustering.

    Parameters
    ----------
    data_wo_sensitive : np.ndarray
        Numerical feature matrix without sensitive attributes.

    sensitive : np.ndarray
        Binary sensitive attribute.

    balance : tuple
        Desired balance ratio (p,q).
        Example:
            (1,1) perfectly balanced
            (1,2) allows 1:2 imbalance
    """

    def __init__(
        self,
        data_wo_sensitive,
        sensitive,
        pq=(1,1),
        random_state=0
    ):

        self.X = np.asarray(
            data_wo_sensitive,
            dtype=float
        )

        self.sensitive = np.asarray(
            sensitive
        )

        self.p, self.q = sorted(pq)

        self.random_state = random_state

        self.fitted = False

        self.fairlets = None
        self.fairlet_centers = None


    def fit(self):

        # clear old global state from original implementation
        FAIRLETS.clear()
        FAIRLET_CENTERS.clear()

        np.random.seed(self.random_state)

        # build quadtree
        root = build_quadtree(
            self.X,
            random_shift=True
        )


        # fairlet decomposition
        tree_fairlet_decomposition(
            self.p,
            self.q,
            root,
            self.X,
            self.sensitive
        )


        # copy because original code uses globals
        self.fairlets = FAIRLETS.copy()

        self.fairlet_centers = FAIRLET_CENTERS.copy()


        self.fitted = True

        return self



    def run(self, k):

        if not self.fitted:
            self.fit()
        if len(self.fairlet_centers) < k:
            return None

        fairlet_center_points = self.X[
            self.fairlet_centers
        ]


        # pairwise distances for fasterpam
        D = pairwise_distances(
            fairlet_center_points,
            metric="euclidean"
        )


        result = kmedoids.fasterpam(
            D,
            k
        )


        medoid_indices = result.medoids


        # map medoids back to original point indices
        cluster_centers = [
            self.fairlet_centers[i]
            for i in medoid_indices
        ]


        # assign each fairlet to closest medoid
        labels = np.zeros(
            len(self.X),
            dtype=int
        )


        medoid_points = self.X[
            cluster_centers
        ]


        for fairlet_id, fairlet in enumerate(self.fairlets):

            fairlet_center = self.X[
                self.fairlet_centers[fairlet_id]
            ]


            distances = np.linalg.norm(
                medoid_points - fairlet_center,
                axis=1
            )

            cluster = np.argmin(distances)


            for point in fairlet:
                labels[point] = cluster


        return labels