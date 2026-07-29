# Implementation of Fair Spectral clustering by
# - Author: Matthaus Kleindessner, Samira Samadi, Pranjal Awasthi, and Jamie Morgenstern
# - Source: https://github.com/matthklein/fair_spectral_clustering
# - License: -

# Paper: Guarantees for Spectral Clustering with Fairness Constraints
# Authors: Matthaus Kleindessner, Samira Samadi, Pranjal Awasthi, and Jamie Morgenstern
# Link: https://arxiv.org/pdf/1901.08668

# Our modifications (Krieger et al.):
#    (1) translated from matlab

# modified by Rainer Wöß 
# (2) adapt input parameters to not use dataloader class but directly receive data and sensitive columns
import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import null_space
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh

class FairSC:
    """
        Fair-Spectral-Clustering algorithm object.
    """
    def __init__(self, data_w_sensitive, sens_columns):

        self.sens_columns = sens_columns
        self.data = data_w_sensitive
        #self.data_loader = data_loader
        connectivity = kneighbors_graph(
            data_w_sensitive, n_neighbors=15, mode="distance", include_self=False
        )
        connectivity = connectivity.toarray()
        self.affinity_matrix_ = np.asarray(0.5 * (connectivity + connectivity.T))

    def run(self, k):
        """
            Run unnormalized spectral clustering and return the labels.

                Parameters:
                        k (int): number of clusters.

                Returns:
                        labels (np.ndarray): clustering labels.
        """
        fairSC_H, fairSC_labels = self.unnormalized_sc_with_fairness_constraints(
            self.affinity_matrix_,
            k,
            laplacian=True,
        )
        return fairSC_labels

    def unnormalized_sc_with_fairness_constraints(self, W, k, laplacian=True):
        """
            Implementation of fair unnormalized SC with fairness constraints translated from MATLAB to Python.
            Original matlab code from paper: https://github.com/matthklein/fair_spectral_clustering/blob/master/Fair_SC_unnormalized.m

                Parameters:
                        W (np.ndarray) : weigthed adjacency matrix.
                        k (int): number of clusters.

                Returns:
                        H : eigenvectors drawn out of the subspace.
                        labels.
        """
        # n = number of datapoints
        # h = number of subgroups resp. number of sensitive attributes
        ################### Prepare G
        # number of datapoints
        #n = self.data_loader.get_data().shape[0]
        n = self.data.shape[0]
        # group-membership vector
        G = np.array(self.calculate_group_membership())

        """Step 1: compute L"""
        if laplacian == True:
            # compute degree matrix
            degrees = np.sum(W, axis=0)
            D = np.diag(degrees)
            # print(D)

            # compute Laplacian from D and W (nxn)
            L = D - W
        else:
            L = W

        """Step 2: compute F """
        ones_matrix = np.ones(n)
        # number of sensitive attributes
        h = G.shape[1]
        # subtrahend = (h / n) * ones_matrix
        # fs are the columns
        fs = []
        # per sensitive attribute we calculate feature vector - number of elements in s divided by number of datapoints* ones
        for s in range(h - 1):
            fs.append(G[:, s] - (sum(G[:, s])) / n * ones_matrix)
        # F is the matrix composed of the columns in fs
        F = np.transpose(np.array(fs))

        """Step 3: compute Z"""
        Z = null_space(F.conj().T)

        """Step 5: compute k smallest eigenvectors"""
        Z_D = Z.conj().T @ L @ Z
        Z_lap = (Z_D + Z_D.conj().T) / 2  # ensuring symmetry
        try:
            eigValues, Y = eigh(
                Z_lap
            )
            # eigValues, Y = eigsh(
            #    Z_lap, k, which="SM", maxiter=1000, ncv=min(Z_lap.shape[0], max(2 * k, 25))
            # )
        except:
            eigValues, Y = eigsh(
                Z_lap,
                k,
                which="SA",
                maxiter=1000,
                tol=0.01,
                ncv=min(Z_D.shape[0], max(2 * k, 25)),
            )
        # eig val already sorted
        index_k_smallest = [i for i in range(0, k)]
        Y = Y[:, index_k_smallest]

        H = Z @ Y
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(H)
        labels = kmeans.labels_

        return H, labels

    def calculate_group_membership(self):
        """
                Calculates the group membership vector.

                    Parameters:
                        data_loader : DataLoader object encapsulating the data.

                    Returns:
                        group_membership_vector.
        """
        # we get columns with sensitive values encoded as numbers
        # empty groupmembership- vector
        G = []
        # load dataframe including the sensitive attributes
        sensitive_columns = self.sens_columns
       # print(self.sens_columns)
        #print(self.sens_columns.shape)
        if self.sens_columns.shape[1] > 1:
            sensitive_columns = self.sens_mixed()

        # print(sensitive_columns)
        # for each sensitive column
        for sensitive_column in sensitive_columns:
            # take the column
            column = sensitive_columns[sensitive_column]
            # create a 2D array filled with 0's (size is number of samples * number of sensitive attributes in this column)
            encoded_array = np.zeros((column.size, column.max() + 1), dtype=int)
            # one hot encoding
            encoded_array[np.arange(column.size), column] = 1
            G.append(encoded_array)
        G = np.concatenate(G, axis=1)
        return G