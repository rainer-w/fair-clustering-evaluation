# Are We Evaluating Fair Clustering Fairly? Benchmarks, Protocols and Data Generation for Non-Convex Fair Clustering

Fair clustering research has largely relied on benchmark datasets originally developed for supervised learning tasks. Despite the rapid growth of the field, there is still no established benchmark suite or standardized evaluation protocol for fair clustering.

As a result, published results are often based on a small number of reused datasets, inconsistent preprocessing choices, and incomparable experimental settings, making it difficult to assess real progress. In this work, we present a benchmark study of fair clustering evaluation with a focus on density-based methods.

We analyze commonly used benchmark datasets and evaluation protocols and identify major limitations in current evaluation practice. We further introduce a fairness-aware model selection protocol that jointly considers clustering quality and fairness when selecting hyperparameters for standard clustering algorithms. Our experiments show that this stronger baseline frequently achieves fairness comparable to or better than dedicated fair clustering methods while maintaining competitive clustering quality.

To enable systematic evaluation beyond existing benchmarks, we present **DEBRIS**, a synthetic data generator for density-based clusters with controllable sensitive attribute distributions and varying levels of difficulty. DEBRIS generates non-convex cluster structures, supports configurable dimensionality, density, noise, and subgroup organization, and provides both density-based and fairness-oriented reference labelings.

We hope that our benchmark study, evaluation protocol, and synthetic generator contribute to more reliable and reproducible evaluation of future fair clustering methods.

---

# Repository Structure

```
data_uci/       Datasets from the UCI repository (see data_uci/README.md)
experiments/    Scripts for reproducing experiments
figures/        Figure generation scripts
results/        Generated experiment outputs
src/            Core implementation
```

---

# Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Reproducing the Paper

All experiments can be executed from the repository root.

For example, to reproduce the UCI benchmark experiments:

```bash
python -m experiments.realworld.run_all_uci
```

Generated outputs are stored in:

```
results/
```

Figures can then be generated using the scripts in:

```
figures/
```

---

# DEBRIS Synthetic Data Generator

DEBRIS generates synthetic density-based clusters with controllable sensitive attribute distributions.

Available parameters:

| Parameter     | Description                                                      |
| ------------- | ---------------------------------------------------------------- |
| `dim`         | Number of dimensions                                             |
| `clunum`      | Number of clusters                                               |
| `core_num`    | Number of cores used for generation                              |
| `ratio_noise` | Ratio of noise samples                                           |
| `seed`        | Random seed                                                      |
| `g`           | Number of sensitive attribute values                             |
| `distr`       | Distribution of sensitive values within subgroups                |
| `gap`         | Number of cores separating individual subgroups within a cluster |
| `clu_ratios`  | Distribution of points across clusters                           |

Example:

```python
clu_ratios = np.ones(clunum) / clunum
```

creates clusters with equal densities.

The sensitive attribute distribution can be controlled using:

```python
distr = [[0.9, 0.1],
         [0.1, 0.9]]
```

which creates two subgroups dominated by different sensitive attribute values.

---
## Example Usage: DEBRIS Generator

<p align="center">
  <img src="docs/images/fair_clusters.png" width="30%">
  <img src="docs/images/gt_clusters.png" width="30%">
  <img src="docs/images/density_clusters.png" width="30%">
</p>

DEBRIS can be used to generate synthetic density-based clusters with controllable sensitive attribute distributions.

The following example generates a dataset with:

| Parameter | Value |
|-----------|-------|
| Number of samples | 1000 |
| Number of clusters | 5 |
| Dimensionality | 2 |
| Sensitive attribute groups | 2 |
| Noise ratio | 0.15 |
| Number of cores | 15 per cluster |
| Random seed | 36 |
| Sensitive attribute distribution | `[[0.9, 0.1], [0.1, 0.9]]` |
| Cluster ratios (densities) | `[[0.5, 0.1, 0.1, 0.2, 0.1]]` |

Run the example:

```bash
python -m figures.generate_debris_example
```
---
# Citation

If you use this repository or DEBRIS in your research, please cite:

```
TODO: add BibTeX entry
```
