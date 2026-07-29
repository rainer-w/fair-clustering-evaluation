# Copyright 2025 Forschungszentrum Juelich GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implementation of FairDen by
# - Author: us (Lena Krieger*, Anna Beer*, Pernille Matthews, Anneka Myrup Thiesson, and Ira Assent)
# - Source: this repo
# - License: Apache 2.0

# Paper: FairDen: Fair Density-based Clustering
# Authors: Lena Krieger*, Anna Beer*, Pernille Matthews, Anneka Myrup Thiesson, and Ira Assent
# Link: https://openreview.net/pdf?id=aPHHhnZktB

import scipy

import numpy as np

from scipy.linalg import pinv
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans
from src.methods.dctree import DCTree

class FairDen(object):

    def __init__(self, min_pts, n_clusters, data_wo_sensitive, sens_columns, sens_mixed, alpha='0',encoded_data=None, num_categorical=0 ):
        """
            Construct FairDEN object and precompute, dc_distance, matrices.

            Parameters:
                data_loader: DataLoader object encapsulating the dataset.
                min_pts: min_pts parameter for dc_distance.
                alpha: alpha parameter showing inclusion of categorical when set to 'avg' (weighted avg).

        """
        self.n_clusters = n_clusters
        self.n, self.d = data_wo_sensitive.shape
        self.data_wo_sensitive = data_wo_sensitive
        if num_categorical != 0:
            self.encoded_data = encoded_data
        self.num_categorical = num_categorical
        self.alpha = alpha
        self.sens_columns = sens_columns
        self.sens_mixed = sens_mixed


        self.min_points = min_pts
        # calculate the DC distance
        self.__setup()

    def get_similarity(self):
        # get precomputed distances
        self.dc_dist = np.array(DCTree(self.data_wo_sensitive, min_points=self.min_points, no_fastindex=True,
                                       use_less_memory=True).dc_distances(), dtype='f')
        # normalized
        sim = np.array(self.dc_dist / np.linalg.norm(self.dc_dist))
        # get similiarity scores
        sim = 1 - sim
        # scale similarity scores to be in range ]0, 1]
        sim = (np.array(sim) - np.min(np.array(sim))) / (np.max(np.array(sim)) - np.min(np.array(sim)))
        np.fill_diagonal(sim, 0)
        del self.dc_dist
        return sim

    def __setup(self):
        import faulthandler
        faulthandler.enable()
        # number of data points
        n = self.n
        # group-membership vector
        group_membership_vector = np.array(self.calculate_group_membership())
        # dc_distance precomputed
        scaled = self.get_similarity()
        # if categorical data include weighted average
        if self.alpha == 'avg':
            num_features = self.data_wo_sensitive.shape[1]
            scaled = np.add(num_features * scaled, self.num_categorical * np.array(self.encoded_data))
            scaled = scaled / (num_features + self.num_categorical)
            del self.encoded_data
            scaled = (scaled - np.min(scaled)) / (np.max(scaled) - np.min(scaled))
            np.fill_diagonal(scaled, 0)
        del self.data_wo_sensitive
        # Degree matrix
        D = np.sum(scaled, axis=0)
        D = np.diag(D)
        # compute laplacian L
        self.L = D - scaled
        del scaled
        # compute Fairness matrix F
        ones_matrix = np.ones(n)
        # number of sensitive attributes
        h = group_membership_vector.shape[1]
        # fs are the columns
        fs = []
        # per sensitive attribute we calculate feature vector - number of elements in s divided by number of
        # datapoints* ones
        fs = [group_membership_vector[:, s] - (((sum(group_membership_vector[:, s])) / n) * ones_matrix) for s in range(h-1)]
        # F is the matrix composed of the columns in fs
        self.F = np.transpose(np.array(fs))
        # Z is nullspace of F^T
        self.Z = scipy.linalg.null_space(self.F.conj().T, rcond=0.001)
        # Q sqrtm
        Q = sqrtm(self.Z.conj().T @ D @ self.Z)
        self.Q_inv = np.linalg.pinv(Q)
        del Q

        Z_D = (self.Z.conj().T @ self.L @ self.Z)

        Z_lap = self.Q_inv.conj().T @ Z_D @ self.Q_inv
        del Z_D
        Z_lap = (Z_lap + Z_lap.conj().T) / 2  # ensuring symmetry
        # predict eigenvalues
        #self.eig_val, self.eig_vec = np.linalg.eigh(Z_lap)
        self.eig_val, self.eig_vec = scipy.sparse.linalg.eigsh(Z_lap, k=10, which='SA', return_eigenvectors=True)
        del Z_lap

    def run(self, k=2):
        """
            Run the FairDEN algorithm and calculate the clustering for a given number of clusters.

            Parameters:
                k (int): Number of clusters.

            Returns:
                clustering (np.array): Clustering.
        """
        degree = k
        repeat = True
        # eig val already sorted
        Y = self.eig_vec[:, 0:k]
        H = self.Z @ self.Q_inv @ Y
        if np.iscomplex(H).any():
            # ADJUSTED FROM ORIGINAL FAIRDEN TO NOT RETURN "None"
            H = np.real(H)
            print("real value of H taken by FairDen")
        # repeat is True when there are less than k non noise clusters
        while repeat:
            clustering = KMeans(n_clusters=k, n_init=10).fit(H)
            clustering = clustering.labels_
            labels, counts = np.unique(clustering, return_counts=True)
            for label, count in zip(labels, counts):
                # if cluster is smaller than minpoints the points are considered noise
                if count < self.min_points:
                    clustering[clustering == label] = -1
            labels = np.unique(clustering)
            # check if there are k non noise clusters
            if -1 in labels:
                if len(labels) == degree + 1 or len(labels) > degree + 1:
                    repeat = False
                else:
                    self.k = k + 1
                    k = self.k
                    repeat = True
            else:
                if len(labels) == degree or len(labels) > degree:
                    repeat = False
                else:
                    self.k = k + 1
                    k = self.k
                    repeat = True
            # ADJUSTED FROM ORIGINAL FAIRDEN -> STOP CONDITION
            if k == self.n+1: 
                repeat = False
        labels = np.unique(clustering)
        i = 0
        # adjust labels to be in order without missing ones
        for label in labels:
            if label == -1:
                pass
            else:
                clustering[clustering == label] = i
                i = i + 1
        return clustering


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