# Implementation of the dc-distance with a DCTree by
# - Author: Pascal Weber (pasiweber)
# - Source: https://github.com/pasiweber/SHADE
# License: BSD 3 clause

# Paper: Connecting the Dots -- Density-Connectivity Distance unifies DBSCAN, k-Center and Spectral Clustering
# Authors: Anna Beer, Andrew Draganov, Ellen Hohma, Philipp Jahn, Christian M.M. Frey, and Ira Assent
# Link: https://doi.org/10.1145/3580305.3599283

import faiss
import gzip
import json
import numpy as np

from dcdist import serialize_DCTree
from dcdist import deserialize_DCTree
from dcdist import DCTree
from typing import Optional, Sequence, Tuple, Union


class SampleDCTree:
    """
    The SampleDCTree computes the dc_distances of a given sample with the DCTree and builds
    a faiss index structures over them. Hence, the index structure can retrieve the nearest
    sampled points of unsampled ones. The dc_distance of two points `x` and `y` is then
    approximated by `max(dc_dist(x',y'), eucl(x,x'), eucl(y,y'))`, where `x'` and `y'` are the
    nearest neighbors within the given sample, `dc_dist` is the dc_distance, and `eucl` is the
    euclidean distance.

    The function `dc_dist(i, j)` returns the approx. dc_distance between `points[i]` and
    `points[j]`.
    The function `dc_distance(X, Y=None)` returns a dc_distance matrix with the approx.
    dc_distance from each pair of `points[X]` and `points[Y]`.

    The SampleDCTree provides `serialize` and `serialize_compressed` functions to serialize
    the SampleDCTree. With `deserialize` or `deserialize_compressed` the SampleDCTree can be
    deserialized again.

    The SampleDCTree also provides `save` and `load` functions to save / load the SampleDCTree
    to / from disk.


    Parameters
    ----------
    points : np.ndarray
        points, of which the approx. dc_distances should be computed of.

    samples : Union[Sequence[int], np.ndarray, int, None] = None
        List of indexes of the points which should be sampled. A DCTree with `points[samples]`
        will be constructed and a faiss index structure on them initialized.

        If set to an integer, a uniform and random selection of points with size `samples`
        is chosen as sample.

        If not set or set to None, all points are used as sample. This can lead to a large
        memory consumption and a very long initialization.

    min_points : int, optional
        min_points parameter used for the computation of the dc_distances, by default 5.

    use_cache: bool
        Save the dc_dists matrix of all sampled points `points[samples]` for faster lookups.
        Usually, this occupies `nr_of_samples**2 * 8` bytes in the RAM.
        By default True.


    Functions
    ---------
    SampleDCTree[x] or SampleDCTree[i,j] or SampleDCTree[X,Y]:
        Returns the point(s) at the given index if `arg` is an integer or Sequence.

        If SampleDCTree[i,j] is used, the approx. dc_dist of `points[i]` and `points[j]`
        is returned (i and j are integer).

        If SampleDCTree[X,Y] is used, the approx. dc_dist matrix between each pair of
        `points[X]` and `points[Y]` is returned (X and Y are np.ndarray or Sequences).

    dc_dist : (self, i: int, j: int, exact_dists: bool = False) -> float
        Returns the approx. dc_distance between points[i] and points[j].

        `exact_dists` (by default: False):
            If True: Calculate exact euclidean distance from sample points to non sample points
            If False: Use faiss approx. euclidean distance from sample points to non sample points

    dc_distances : (self, X = None, Y = None, exact_dists = False, access_method = "tree") -> np.ndarray
        Computes the approx. dc_distance matrix between each pair of points[X] and points[Y].
        If X=None, X=range(n) is used.
        If Y=None, Y=X is used.

        `access_method` (by default: "tree"):
            `access_method` of the DCTree.dc_distances function. Is only used if
            self.use_cache = False.

        `exact_dists` (by default: False):
            If True: Calculate exact euclidean distance from sample points to non sample points
            If False: Use faiss approx. euclidean distance from sample points to non sample points

        Returns dc_dists: ndarray of shape (n_samples_X, n_samples_Y)


    serialize : (sample_dc_tree: SampleDCTree) -> str
        Serializes the SampleDCTree `sample_dc_tree` to a string.

    serialize_compressed : (sample_dc_tree: SampleDCTree) -> bytes
        Serializes the SampleDCTree `sample_dc_tree` to a compressed byte array.

    save : (sample_dc_tree: SampleDCTree, file_path: str) -> save to disk
        Saves the SampleDCTree `sample_dc_tree` to disk at `file_path`.


    deserialize : (data: str, n_jobs = None) -> SampleDCTree
        Deserializes a string `str` to a SampleDCTree.

    deserialize_compressed : (data: bytes, n_jobs = None) -> SampleDCTree
        Deserializes a compressed byte array `bytes` to a SampleDCTree.

    load : (file_path: str, n_jobs = None) -> SampleDCTree
        Loads a SampleDCTree from disk at `file_path`.


    Examples
    --------
    >>> import dcdist
    >>> points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])
    >>> sample_dc_tree = dcdist.SDCTree(points, 5, [0,1,4,5])
    >>> print(sample_dc_tree.dc_dist(2,5))
    >>> print(sample_dc_tree.dc_distances(range(len(points))))
    >>> print(sample_dc_tree.dc_distances([0,1], [2,3]))

    >>> print(sample_dc_tree[2,5])
    >>> print(sample_dc_tree[range(len(points), range(len(points))])
    >>> print(sample_dc_tree[[0,1], [2,3]])

    >>> s = dcdist.serialize_SDCTree(sample_dc_tree)
    >>> sample_dc_tree_new = dcdist.deserialize_SDCTree(s)

    >>> b = dcdist.serialize_SDCTree_compressed(sample_dc_tree)
    >>> sample_dc_tree_new = dcdist.deserialize_SDCTree_compressed(b)

    >>> dcdist.save_SDCTree(sample_dc_tree, "./data.sdctree")
    >>> sample_dc_tree_new = dcdist.load_SDCTree("./data.sdctree")
    """

    points: np.ndarray
    samples: Union[Sequence[int], np.ndarray]
    lookup_sample: np.ndarray
    min_points: int = 5
    faiss_index: faiss.Index
    use_cache: bool = True

    dc_tree: DCTree
    dc_dists: np.ndarray

    def __init__(
        self,
        points: np.ndarray,
        samples: Union[Sequence[int], np.ndarray, int, None] = None,
        min_points: int = 5,
        use_cache: bool = True,
        n_jobs: Optional[int] = None,
    ):
        self.points = points

        if samples is None:
            self.samples = range(points.shape[0])
        elif isinstance(samples, int):
            samples = min(points.shape[0], samples)
            self.samples = np.random.choice(a=points.shape[0], size=samples, replace=False)
        else:
            self.samples = samples

        self.lookup_sample = np.full(points.shape[0], -1)
        self.lookup_sample[self.samples] = np.arange(len(self.samples))

        self.dc_tree = DCTree(points=points[self.samples], min_points=min_points, n_jobs=n_jobs)

        if use_cache:
            self.dc_dists = self.dc_tree.dc_distances(range(len(self.samples)))
            del self.dc_tree

        self._init_faiss()

    def _init_faiss(self):
        self.faiss_index = faiss.IndexFlatL2(self.points.shape[1])
        self.faiss_index.add(self.points[self.samples].astype(np.float32))

    def __getitem__(
        self,
        arg: Union[
            int,
            Sequence[int],
            np.ndarray,
            Tuple[int, int],
            Tuple[int, np.ndarray, Sequence[int]],
            Tuple[np.ndarray, Sequence[int], int],
            Tuple[np.ndarray, Sequence[int], np.ndarray, Sequence[int]],
        ],
    ) -> Union[int, float, np.ndarray]:
        """
        Returns the point(s) at the given index if `arg` is an integer or Sequence.

        If SampleDCTree[i,j] is used, the approx. dc_dist of `points[i]` and `points[j]`
        is returned (i and j are integer).

        If SampleDCTree[X,Y] is used, the approx. dc_dist matrix between each pair of
        `points[X]` and `points[Y]` is returned (X and Y are np.ndarray or Sequences).
        """

        index_error_msg = f"`{arg}` needs to be an integer, Sequence, np.ndarray, tuple of integer, or tuple of np.ndarray / Sequence!"

        if isinstance(arg, tuple):
            if len(arg) != 2:
                raise IndexError(index_error_msg)
            (i, j) = arg
            if isinstance(i, int) and isinstance(j, int):
                return self.dc_dist(i, j)
            if isinstance(i, int) and isinstance(j, (np.ndarray, Sequence)):
                if isinstance(j, np.ndarray):
                    j = j.flatten()
                return self.dc_distances([i], j)
            if isinstance(i, (np.ndarray, Sequence)) and isinstance(j, int):
                if isinstance(i, np.ndarray):
                    i = i.flatten()
                return self.dc_distances(i, [j])
            elif isinstance(i, (np.ndarray, Sequence)) and isinstance(j, (np.ndarray, Sequence)):
                if isinstance(i, np.ndarray):
                    i = i.flatten()
                if isinstance(j, np.ndarray):
                    j = j.flatten()
                return self.dc_distances(i, j)
            else:
                raise IndexError(index_error_msg)

        elif isinstance(arg, (int, Sequence, np.ndarray)):
            return self.points[arg]

        else:
            raise IndexError(index_error_msg)

    def _query_faiss(
        self, X: np.ndarray, exact_dists: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find for all `X` the nearest-neighbor in `sample_points` and the distance to it"""
        x = X.astype(np.float32).reshape((-1, self.points.shape[1]))
        dists, labels = self.faiss_index.search(x=x, k=1)
        labels = labels[:, 0]
        if exact_dists:
            dists = np.linalg.norm(X - self.points[self.samples][labels], axis=1)
        else:
            dists = dists[:, 0]
        return (labels, dists)

    def dc_dist(self, i: int, j: int, exact_dists: bool = False) -> float:
        """Returns the approx. dc_distance from points[i] to points[j]."""

        if i == j:
            return 0

        (i_tree, i_dist) = (self.lookup_sample[i], 0)
        if i_tree == -1:
            # i is not in sample_points
            (labels, dists) = self._query_faiss(self.points[i][np.newaxis, :], exact_dists)
            (i_tree, i_dist) = (labels[0], dists[0])

        (j_tree, j_dist) = (self.lookup_sample[j], 0)
        if j_tree == -1:
            # j is not in sample_points
            (labels, dists) = self._query_faiss(self.points[j][np.newaxis, :], exact_dists)
            (j_tree, j_dist) = (labels[0], dists[0])

        if self.use_cache:
            sample_dc_dist = self.dc_dists[i_tree, j_tree]
        else:
            sample_dc_dist = self.dc_tree.dc_dist(i_tree, j_tree)

        return max(i_dist, j_dist, sample_dc_dist)

    def dc_distances(
        self,
        X: Union[Sequence[int], np.ndarray, None] = None,
        Y: Union[Sequence[int], np.ndarray, None] = None,
        exact_dists: bool = False,
        access_method: str = "tree",
    ) -> np.ndarray:
        """
        Computes the approx. dc_distance matrix between each pair of points[X] and points[Y].
        If X=None, X=range(n) is used.
        If Y=None, Y=X is used.

        Returns dc_dists: ndarray of shape (n_samples_X, n_samples_Y)
        """

        if X is None:
            X = range(self.points.shape[0])

        if Y is None:
            Y = X

        if X is Y:
            X = np.array(X)

            i_trees = self.lookup_sample[X]
            i_trees_not_sample_idx = np.where(i_trees == -1)[0]
            (i_labels, i_trees_not_sample_dists) = self._query_faiss(
                self.points[X[i_trees_not_sample_idx]], exact_dists
            )
            i_trees[i_trees_not_sample_idx] = i_labels

            if self.use_cache:
                dc_dists = self.dc_dists[(i_trees[:, np.newaxis], i_trees[np.newaxis, :])]
            else:
                dc_dists = self.dc_tree.dc_distances(i_trees, access_method=access_method)

            dc_dists[i_trees_not_sample_idx, :] = np.maximum(
                dc_dists[i_trees_not_sample_idx, :], i_trees_not_sample_dists[:, np.newaxis]
            )
            dc_dists[:, i_trees_not_sample_idx] = np.maximum(
                dc_dists[:, i_trees_not_sample_idx], i_trees_not_sample_dists[np.newaxis, :]
            )
            np.fill_diagonal(dc_dists, 0)

            return dc_dists

        else:
            X = np.array(X)
            Y = np.array(Y)

            i_trees = self.lookup_sample[X]
            i_trees_not_sample_idx = np.where(i_trees == -1)[0]
            (i_labels, i_trees_not_sample_dists) = self._query_faiss(
                self.points[X[i_trees_not_sample_idx]], exact_dists
            )
            i_trees[i_trees_not_sample_idx] = i_labels

            j_trees = self.lookup_sample[Y]
            j_trees_not_sample_idx = np.where(j_trees == -1)[0]
            (j_labels, j_trees_not_sample_dists) = self._query_faiss(
                self.points[Y[j_trees_not_sample_idx]], exact_dists
            )
            j_trees[j_trees_not_sample_idx] = j_labels

            if self.use_cache:
                dc_dists = self.dc_dists[i_trees[:, np.newaxis], j_trees[np.newaxis, :]]
            else:
                dc_dists = self.dc_tree.dc_distances(i_trees, j_trees, access_method)

            dc_dists[i_trees_not_sample_idx, :] = np.maximum(
                dc_dists[i_trees_not_sample_idx, :], i_trees_not_sample_dists[:, np.newaxis]
            )
            dc_dists[:, j_trees_not_sample_idx] = np.maximum(
                dc_dists[:, j_trees_not_sample_idx], j_trees_not_sample_dists[np.newaxis, :]
            )

            # Set edges from i -> i to 0
            self_edges = np.zeros((0, 2), dtype=int)
            swapped = False
            if len(X) < len(Y):
                X, Y = Y, X
                swapped = True
            for shift in range(len(X)):
                edge_to_self = np.where(X[0 : len(Y)] == Y)[0]
                if not swapped:
                    self_to_self = np.c_[(edge_to_self + shift) % len(X), edge_to_self]
                else:
                    self_to_self = np.c_[edge_to_self, (edge_to_self + shift) % len(X)]
                self_edges = np.concatenate([self_edges, self_to_self], axis=0)
                X = np.roll(X, -1)
            dc_dists[self_edges[:, 0], self_edges[:, 1]] = 0

            return dc_dists


def serialize(sample_dc_tree: SampleDCTree) -> str:
    """Serializes the SampleDCTree `sample_dc_tree` to a string."""

    samples = sample_dc_tree.samples
    if isinstance(samples, np.ndarray):
        samples = samples.tolist()

    if sample_dc_tree.use_cache:
        dc = json.dumps(sample_dc_tree.dc_dists.tolist())
    else:
        dc = serialize_DCTree(sample_dc_tree.dc_tree)

    sep = "<\1sample_dc_tree>"
    # points, samples, min_points, faiss_index, use_cache, dc_tree or dc_dists
    return (
        json.dumps(sample_dc_tree.points.tolist())
        + sep
        + json.dumps(samples)
        + sep
        + str(sample_dc_tree.min_points)
        + sep
        + json.dumps(faiss.serialize_index(sample_dc_tree.faiss_index).tolist())
        + sep
        + str(sample_dc_tree.use_cache)
        + sep
        + dc
    )


def serialize_compressed(sample_dc_tree: SampleDCTree) -> bytes:
    """Serializes the SampleDCTree `sample_dc_tree` to a gzip-compressed byte array."""
    data = serialize(sample_dc_tree)
    byte_data = bytes(data, "utf-8")
    return gzip.compress(byte_data)


def save(sample_dc_tree: SampleDCTree, file_path: str) -> None:
    """Saves the SampleDCTree `sample_dc_tree` to disk at `file_path`."""
    byte_data = serialize_compressed(sample_dc_tree)
    with open(file_path, "wb") as file:
        file.write(byte_data)


def deserialize(data: str, n_jobs: Optional[int] = None) -> SampleDCTree:
    """Deserializes a string `str` to a SampleDCTree."""
    sample_dc_tree = object.__new__(SampleDCTree)

    sep = "<\1sample_dc_tree>"
    # points, samples, min_points, faiss_index, use_cache, dc_tree or dc_dists
    (points, samples, min_points, faiss_index, use_cache, dc) = data.split(sep)
    sample_dc_tree.points = np.array(json.loads(points))
    sample_dc_tree.samples = np.array(json.loads(samples))

    sample_dc_tree.lookup_sample = np.full(sample_dc_tree.points.shape[0], -1)
    sample_dc_tree.lookup_sample[sample_dc_tree.samples] = np.arange(len(sample_dc_tree.samples))

    sample_dc_tree.min_points = int(min_points)
    sample_dc_tree.faiss_index = faiss.deserialize_index(
        np.array(json.loads(faiss_index), dtype=np.uint8)
    )
    sample_dc_tree.use_cache = bool(use_cache)

    if sample_dc_tree.use_cache:
        sample_dc_tree.dc_dists = np.array(json.loads(dc))
    else:
        sample_dc_tree.dc_tree = deserialize_DCTree(dc, n_jobs)

    return sample_dc_tree


def deserialize_compressed(compressed_data: bytes, n_jobs: Optional[int] = None) -> SampleDCTree:
    """Deserializes a compressed byte array `bytes` to a SampleDCTree."""
    byte_data = gzip.decompress(compressed_data)
    data = str(byte_data, "utf-8")
    return deserialize(data, n_jobs)


def load(file_path: str, n_jobs: Optional[int] = None) -> SampleDCTree:
    """Loads a SampleDCTree from disk at `file_path`."""
    file = open(file_path, "rb")
    byte_data = file.read()
    file.close()
    return deserialize_compressed(byte_data, n_jobs)