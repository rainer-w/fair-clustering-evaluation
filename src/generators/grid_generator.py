import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import PCG64
from scipy.spatial.distance import euclidean


class GridGenerator:

    def __init__(self, grid_len, num_sensitive=2, radius= 0.5):
        self.grid_len = grid_len
        self.centers = []
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                self.centers.append([i,j])
        self.centers = np.array(self.centers)
        self.radius = radius
        self.num_sensitive = num_sensitive


    def generate_data(self, num, random_state=42):


        main_generator = np.random.Generator(PCG64(random_state))
        center_count = len(self.centers)
        labels = main_generator.integers(0, center_count, num)
        _, indices, cluster_counts = np.unique(labels, return_counts=True, return_inverse=True)
        data = np.zeros((num, len(self.centers[0])))
        for cluid in range(len(cluster_counts)):
            cluster_data = []
            for j in range(cluster_counts[cluid]):
                if j == 0:
                    instance = self.centers[cluid]
                else:
                    instance0 = main_generator.uniform(self.centers[cluid][0] - self.radius, self.centers[cluid][0] + self.radius )
                    instance1 = main_generator.uniform(self.centers[cluid][1] - self.radius, self.centers[cluid][1] + self.radius)
                    instance = np.array([instance0, instance1])
                    while euclidean(instance, self.centers[cluid]) > self.radius:
                        instance0 = main_generator.uniform(self.centers[cluid][0] - self.radius, self.centers[cluid][0] + self.radius )
                        instance1 = main_generator.uniform(self.centers[cluid][1] - self.radius, self.centers[cluid][1] + self.radius)
                        instance = np.array([instance0, instance1])
                cluster_data.append(instance)
            data[np.where(indices == cluid)] = cluster_data

        sensitive = main_generator.integers(0, self.num_sensitive, num)

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

if __name__ == "__main__":
    grid_generator = GridGenerator(grid_len=3)
    grid_generator.generate_data(num=1000)

    grid_data = grid_generator.get_features_wo_sensitive()
    grid_labels = grid_generator.get_groundtruth()
    grid_sensitive = grid_generator.get_unfair_groundtruth()

    marker_dict = {1: "o", 0: "*"}
    colors = {1: "blue", 0: "orange"}
    print(grid_labels, grid_sensitive)
    marker_list = [marker_dict[s] for s in grid_sensitive]
    indices_1 = np.where(grid_sensitive == 1)[0]
    indices_0 = np.where(grid_sensitive == 0)[0]
    plt.figure(figsize=(5,5))
    plt.scatter(grid_data[indices_0,0], grid_data[indices_0,1], cmap="viridis", c=grid_labels[indices_0], marker=marker_dict[0])
    plt.scatter(grid_data[indices_1,0], grid_data[indices_1,1], cmap="viridis", c=grid_labels[indices_1], marker=marker_dict[1])
    plt.show()
