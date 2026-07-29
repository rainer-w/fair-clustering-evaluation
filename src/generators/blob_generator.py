import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import PCG64
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_blobs


class BlobGenerator:

    def __init__(self, blob_count = 5):
        self.blob_count = blob_count


    def generate_data(self, num, random_state=42):

        data, labels = make_blobs(n_features=2, n_samples=num, centers=self.blob_count, random_state=random_state)
        main_generator = np.random.Generator(PCG64(random_state))
        sensitive = main_generator.integers(0, 2, num)

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
    blob_generator = BlobGenerator(blob_count = 5)
    blob_generator.generate_data(num=200)

    blob_data = blob_generator.get_features_wo_sensitive()
    blob_labels = blob_generator.get_groundtruth()
    blob_sensitive = blob_generator.get_unfair_groundtruth()

    marker_dict = {1: "o", 0: "*"}
    colors = {1: "blue", 0: "orange"}
    print(blob_labels, blob_sensitive)
    marker_list = [marker_dict[s] for s in blob_sensitive]
    indices_1 = np.where(blob_sensitive == 1)[0]
    indices_0 = np.where(blob_sensitive == 0)[0]
    plt.figure(figsize=(5,5))
    plt.scatter(blob_data[indices_0,0], blob_data[indices_0,1], cmap="viridis", c=blob_labels[indices_0], marker=marker_dict[0])
    plt.scatter(blob_data[indices_1,0], blob_data[indices_1,1], cmap="viridis", c=blob_labels[indices_1], marker=marker_dict[1])
    plt.show()
