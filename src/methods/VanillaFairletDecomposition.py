# Implementation of Fairlets by
# - Author: Akhil Gupta, Anunay Sharma, Ayush Rajput, Badrinarayanan Rajasekaran, and Vishnu Pratheek Challa
# - Source: https://github.com/guptakhil/fair-clustering-fairlets
# - License: MIT

# Paper: Fair Clustering Through Fairlets
# Authors: Flavio Chierichetti, Ravi Kumar, Silvio Lattanzi, Sergei Vassilvitskii
# Link: https://arxiv.org/abs/1802.05733

import numpy as np
import random
import time
from math import gcd
import matplotlib.pyplot as plt
from src.methods.utils import distance


class VanillaFairletDecomposition(object):
    """
    Computes vanilla fairlet decomposition that ensures fair clusters. It might not give the optimal cost value.
    """

    def __init__(self, p, q, blues, reds, data):
        """
        p (int) : First balance parameter
        q (int) : Second balance parameter
        blues (list) : Index of the points corresponding to first class
        reds (list) : Index of the points corresponding to second class
        data (list) : Contains actual data points
        """
        self.p = p
        self.q = q
        self.blues = blues
        self.reds = reds
        self.data = data

    def balanced(self, r, b):
        """
        Checks for initial balance and feasibility.

        Args:
                r (int) : Total length of majority class
                b (int) : Total length of minority class

        Returns:
                bool value indicating whether the balance is possible
        """
        if r == 0 and b == 0:
            return True
        if r == 0 or b == 0:
            return False
        return min(float(r / b), float(b / r)) >= float(self.p / self.q)

    def make_fairlet(self, points, dataset, fairlets, fairlet_centers, costs):
        """
        Adds fairlet to the fairlet decomposition and returns the k-center cost for the fairlet.

        Args:
                points (list) : Index of the points that comprise the fairlet
                dataset (list) : Original data
                fairlets (list)
                fairlet_centers (list)
                costs (list)
        """
        # Finding the point as center whose maximum distance from any point is minimum

        cost_list = [
            (i, max([distance(dataset[i], dataset[j]) for j in points])) for i in points
        ]
        cost_list = sorted(cost_list, key=lambda x: x[1], reverse=False)
        center, cost = cost_list[0][0], cost_list[0][1]

        # Adding the shortlisted points to the fairlets
        fairlets.append(points)
        fairlet_centers.append(center)
        costs.append(cost)

        return

    def decompose(self):
        """
        Computes vanilla (p , q) - fairlet decomposition of given points as per Lemma 3 in NeurIPS 2017 paper.
        Assumes that balance parameters are non-negative integers such that gcd(p, q) = 1.
        Also assumes that balance of reds and blues is atleast p/q.

        Returns:
                fairlets (list)
                fairlet_centers (list)
                costs (list)
        """
        assert (
            gcd(self.p, self.q) == 1
        ), "Please ensure that the GCD of balance parameters is 1."
        assert self.p <= self.q, "Please use balance parameters such that p <= q."

        fairlets = []
        fairlet_centers = []
        fairlet_costs = []

        if len(self.reds) < len(
            self.blues
        ):  # We want the reds to be bigger in size as they correspond to 'q' parameter
            temp = self.blues
            self.blues = self.reds
            self.reds = temp

        R = len(self.reds)
        B = len(self.blues)

        assert self.balanced(R, B), (
            "Input sets are unbalanced: " + str(R) + " , " + str(B)
        )

        # If both reds and blues are empty, return empty results
        if R == 0 and B == 0:
            return fairlets, fairlet_centers, fairlet_costs

        b = 0
        r = 0

        # random.seed(42)
        random.shuffle(self.reds)
        random.shuffle(self.blues)

        while (
            ((R - r) - (B - b)) >= (self.q - self.p)
            and (R - r) >= self.q
            and (B - b) >= self.p
        ):
            self.make_fairlet(
                self.reds[r : (r + self.q)] + self.blues[b : (b + self.p)],
                self.data,
                fairlets,
                fairlet_centers,
                fairlet_costs,
            )
            r += self.q
            b += self.p
        if ((R - r) + (B - b)) >= 1 and ((R - r) + (B - b)) <= (self.p + self.q):
            self.make_fairlet(
                self.reds[r:] + self.blues[b:],
                self.data,
                fairlets,
                fairlet_centers,
                fairlet_costs,
            )
            r = R
            b = B
        elif ((R - r) != (B - b)) and ((B - b) >= self.p):
            self.make_fairlet(
                self.reds[r : r + (R - r) - (B - b) + self.p]
                + self.blues[b : (b + self.p)],
                self.data,
                fairlets,
                fairlet_centers,
                fairlet_costs,
            )
            r += (R - r) - (B - b) + self.p
            b += self.p
        assert (R - r) == (B - b), "Error in computing fairlet decomposition."
        for i in range(R - r):
            self.make_fairlet(
                [self.reds[r + i], self.blues[b + i]],
                self.data,
                fairlets,
                fairlet_centers,
                fairlet_costs,
            )

        print("%d fairlets have been identified." % (len(fairlet_centers)))
        assert len(fairlets) == len(fairlet_centers)
        assert len(fairlet_centers) == len(fairlet_costs)

        return fairlets, fairlet_centers, fairlet_costs
