import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import PCG64
from scipy.stats import bernoulli, multivariate_normal


class ZafarGenerator:

    def __init__(self, centers, covs, phi):
        self.centers = centers
        self.covs = covs
        self.phi = phi
        self.df = None

        self.multi_normals = [multivariate_normal(mean=self.centers[i], cov=self.covs[i]) for i in range(len(self.centers))]

    def generate_data(self, num, random_state=42):
        main_generator = np.random.Generator(PCG64(random_state))
        center_count = len(self.centers)
        labels = main_generator.integers(0, center_count, num)
        _, indices, cluster_counts = np.unique(labels, return_counts=True, return_inverse=True)
        data = np.zeros((num, len(self.centers[0])))
        for cluid in range(len(cluster_counts)):
            cluster_data = self.multi_normals[cluid].rvs(cluster_counts[cluid], random_state=main_generator.integers(low=0, high=100, size=1)[0])
            #print(np.where(indices == cluid))
            data[np.where(indices == cluid)] = cluster_data

        rotation = [[math.cos(self.phi), -1*math.sin(self.phi)], [math.sin(self.phi), math.cos(self.phi)]]
        rotation_np = np.array(rotation)
        data_np = np.array(data)
        rotated_data = np.matmul(rotation_np, data_np.T).T

        prob1 = self.multi_normals[0].pdf(rotated_data)
        prob2 = self.multi_normals[1].pdf(rotated_data)
        sensitivty_prob = prob1 / (prob1 + prob2)
        rands = main_generator.random(len(sensitivty_prob))
        sensitive = [1 if rands[i] <= sensitivty_prob[i] else 0 for i in range(len(sensitivty_prob))]
        feature_cols = {
            f"x{i}": data[:, i]
            for i in range(data.shape[1])
        }
        data_df = pd.DataFrame({
            **feature_cols,
            "groundtruth": labels,
            "subgroup_id": sensitive,
            "sensitive_value": sensitive,
        })

        self.df = data_df
        return data_df

    def get_features_wo_sensitive(self):
        return self.df.filter(regex="^x").values
    def get_features_w_sensitive(self):
        return self.df.drop(columns=["groundtruth", "subgroup_id"])
    def get_groundtruth(self):
        return self.df["groundtruth"]
    def get_unfair_groundtruth(self):
        return self.df["subgroup_id"]
    def get_sensitive(self):
        return self.df["sensitive_value"]


"""Specifically, we generate 4,000 binary class labels
uniformly at random and assign a 2-dimensional
user feature vector per label by drawing sam-
ples from two different Gaussian distributions:
p(x|y=1) = N ([2; 2], [5, 1; 1, 5]) and p(x|y=−1) =
N ([−2; −2], [10, 1; 1, 3]). Then, we draw each user’s
sensitive attribute z from a Bernoulli distribution:
p(z = 1) = p(x′|y = 1)/(p(x′|y = 1) + p(x′|y = −1)),
where x′ = [cos(φ), − sin(φ); sin(φ), cos(φ)]x is sim-
ply a rotated version of the feature vector, x.
We generate datasets with two values for the parame-
ter φ, which controls the correlation between the sen-
sitive attribute and the class labels (and hence, the
resulting disparate impact). Here, the closer φ is to
zero, the higher the correlation. Finally, we trained
both types of constrained classifiers on each dataset.
"""

if __name__ == "__main__":
    zafar_generator = ZafarGenerator([[2,2], [-2,-2]], [[[5, 1],[1, 5]],[[10, 1],[1, 3]]], 180)
    zafar_generator.generate_data(500)

    zafar_data = zafar_generator.get_features_wo_sensitive()
    zafar_labels = zafar_generator.get_groundtruth()
    zafar_sensitive = zafar_generator.get_unfair_groundtruth()

    marker_dict = {1: "o", 0: "*"}
    colors = {1: "blue", 0: "orange"}
    color_np = np.array([colors[l] for l in zafar_labels])
    print(zafar_labels, zafar_sensitive)
    marker_list = [marker_dict[s] for s in zafar_sensitive]
    indices_1 = np.where(zafar_sensitive == 1)[0]
    indices_0 = np.where(zafar_sensitive == 0)[0]
    plt.figure(figsize=(5,5))
    plt.scatter(zafar_data[indices_0,0], zafar_data[indices_0,1], cmap="viridis", c=zafar_labels[indices_0], marker=marker_dict[0])
    plt.scatter(zafar_data[indices_1,0], zafar_data[indices_1,1], cmap="viridis", c=zafar_labels[indices_1], marker=marker_dict[1])
    plt.show()

