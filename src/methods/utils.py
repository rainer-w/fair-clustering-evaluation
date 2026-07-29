# Implementation of Fairlets by
# - Author: Akhil Gupta, Anunay Sharma, Ayush Rajput, Badrinarayanan Rajasekaran, and Vishnu Pratheek Challa
# - Source: https://github.com/guptakhil/fair-clustering-fairlets
# - License: MIT

# Paper: Fair Clustering Through Fairlets
# Authors: Flavio Chierichetti, Ravi Kumar, Silvio Lattanzi, Sergei Vassilvitskii
# Link: https://arxiv.org/abs/1802.05733

import numpy as np
import matplotlib.pyplot as plt


def distance(a, b, order=2):
    """
    Calculates the specified norm between two vectors.

    Args:
            a (list) : First vector
            b (list) : Second vector
            order (int) : Order of the norm to be calculated as distance

    Returns:
            Resultant norm value
    """
    assert len(a) == len(b), "Length of the vectors for distance don't match."
    return np.linalg.norm(x=np.array(a) - np.array(b), ord=order)


def plot_analysis(degrees, costs, balances, step_size):
    """
    Plots the curves for costs and balances.

    Args:
            degrees (list)
            costs (list)
            balances (list)
            step_size (int)
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    ax[0].plot(costs, marker=".", color="blue")
    ax[0].set_xticks(list(range(0, len(degrees), step_size)))
    ax[0].set_xticklabels(
        list(range(min(degrees), max(degrees) + 1, step_size)), fontsize=12
    )
    ax[1].plot(balances, marker="x", color="saddlebrown")
    ax[1].set_xticks(list(range(0, len(degrees), step_size)))
    ax[1].set_xticklabels(
        list(range(min(degrees), max(degrees) + 1, step_size)), fontsize=12
    )
    plt.show()
