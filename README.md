# t-SNE from Scratch

A Jupyter notebook that implements **t-SNE (t-Distributed Stochastic Neighbor Embedding)** from scratch, applies it to the `digits` dataset, compares it to scikit-learn's reference implementation, and extends it with a custom `transform` step that maps unseen samples into an existing embedding.

This was Exercise 1 of course 3945 (Machine Learning), 2025A.

---

## Table of contents

- [Overview](#overview)
- [What's in the notebook](#whats-in-the-notebook)
- [Algorithm](#algorithm)
- [Dataset](#dataset)
- [Hyperparameter exploration](#hyperparameter-exploration)
- [Comparison vs. scikit-learn](#comparison-vs-scikit-learn)
- [Extension: embedding new samples](#extension-embedding-new-samples)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Notes](#notes)

---

## Overview

[t-SNE](https://lvdmaaten.github.io/tsne/) is a non-linear dimensionality-reduction technique designed primarily for **visualization** of high-dimensional data. It models pairwise similarities in the high-dimensional space with a Gaussian kernel and in the low-dimensional space with a heavy-tailed Student-t distribution, then minimizes the KL divergence between the two distributions via gradient descent.

This notebook walks through:

1. The mathematical foundations (Gaussian neighborhoods, perplexity, KL divergence, the Student-t kernel).
2. A from-scratch implementation organized as a small, modular class.
3. Application to the classic `sklearn.datasets.load_digits` dataset.
4. A systematic hyperparameter sweep over `perplexity` and `learning_rate`.
5. A side-by-side comparison with `sklearn.manifold.TSNE`, evaluated quantitatively with the silhouette score and qualitatively with cluster plots and pairwise-distance histograms.
6. An **extension** that maps fresh test samples into an existing 2-D embedding — something the standard t-SNE algorithm does not natively support.

## What's in the notebook

| Section | Description |
| --- | --- |
| **Theoretical overview** | Discussion of high-dim vs. low-dim probabilities, perplexity, and the crowding problem. |
| **Implementation** | A class with helper methods like `_compute_prob_matrix`, `_find_optimal_sigma` (binary search for per-point bandwidth), `_low_dim_probs`, and a gradient-descent loop with momentum. |
| **Dataset & preprocessing** | Load `digits`, scale features to `[0, 1]` with `MinMaxScaler`, split 60/40 train/test. |
| **Demonstration** | 2-D visualisation of the embedding coloured by digit label. |
| **Hyperparameter selection** | Sweeps over `perplexity ∈ {5, 30, 50}` and `learning_rate ∈ {50, 200, ...}`. |
| **Comparison** | Custom vs. sklearn t-SNE — clusters, silhouette score, pairwise-distance histograms. |
| **Extension** | Mapping new (test) samples into the existing embedding and measuring how well they fit. |
| **Use of generative AI** | Disclosure of how generative AI was used while writing the notebook. |

## Algorithm

For \(N\) high-dimensional points \(x_1, \dots, x_N\) and a target dimension \(d\) (typically 2):

1. **High-dim affinities.** For each pair \((i, j)\), compute a conditional Gaussian probability:
   \[ p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k\neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)} \]
   The bandwidth \(\sigma_i\) is found by **binary search** so that the perplexity of the conditional distribution equals the user's `perplexity` parameter. Symmetrize: \(p_{ij} = (p_{j|i} + p_{i|j}) / (2N)\).
2. **Low-dim affinities.** Place \(N\) points \(y_1, \dots, y_N\) in \(\mathbb{R}^d\). Define heavy-tailed similarities with a Student-t kernel:
   \[ q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k\neq l} (1 + \|y_k - y_l\|^2)^{-1}} \]
3. **Optimization.** Minimize \(\mathrm{KL}(P \,\|\, Q) = \sum_{i\neq j} p_{ij} \log (p_{ij}/q_{ij})\) by gradient descent with momentum.

### Implementation notes

- **Binary search for \(\sigma_i\)** balances accuracy and efficiency while guaranteeing the requested perplexity per point.
- **Symmetric normalization** of \(P\) ensures probabilities are properly scaled.
- **Student-t kernel in low-dim** addresses the *crowding problem* by spreading distant points further apart than a Gaussian would.
- **Gradient descent with momentum** is used in place of more advanced optimizers like Barnes-Hut or FFT-accelerated gradients (which sklearn uses internally).

## Dataset

The notebook uses [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), which contains 8×8 grayscale images of handwritten digits 0–9 (1797 samples, 64 features). It is well-suited to t-SNE demos because:

- It is high-dimensional (64-D) but small enough to run a from-scratch implementation in a reasonable time.
- It has well-defined, semantically meaningful classes that should form visible clusters in 2-D.

## Hyperparameter exploration

- **Perplexity 5** → embeddings focus too much on local neighbourhoods, clusters scatter and lose cohesion.
- **Perplexity 30** → good balance, well-defined clusters.
- **Perplexity 50** → similar to 30 but slightly better silhouette score; chosen as the final value.
- **Learning rate 50** → slow convergence, less compact clusters.
- **Learning rate 200** → compact, well-separated clusters; used as the final value.

## Comparison vs. scikit-learn

| Metric / aspect | Custom t-SNE | `sklearn.manifold.TSNE` |
| --- | --- | --- |
| Silhouette score (digits) | ~0.42 | ~0.53 |
| Cluster compactness | Tighter, locally focused | Slightly more dispersed, better global separation |
| Pairwise-distance distribution | Narrower range | Broader range |
| Optimizations | Plain gradient descent + momentum | Barnes-Hut / FFT, early exaggeration, PCA init |

Sklearn's higher silhouette score is largely attributable to **early exaggeration** and other optimizations that emphasize global cluster separation. The custom version is more transparent and easier to reason about pedagogically, while sklearn's is more efficient and scalable.

## Extension: embedding new samples

Standard t-SNE does not provide a `transform` for unseen points; recomputing from scratch every time new data arrives is impractical. The notebook implements a custom transformation step: it embeds new test points by minimizing the KL divergence between their conditional probabilities (computed against the original training data) and their positions in the existing 2-D embedding, while keeping the training embedding **fixed**.

Evaluation:

- **Silhouette on training set:** ~0.42
- **Silhouette on transformed test set:** ~0.45

The slight increase suggests the new test points slot smoothly into the existing clusters, and the test stars overlay the training dots cleanly when plotted by digit label.

## Project structure

```
t-sne-notebook/
├── t-SNE.ipynb          # The main notebook
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── .gitignore
```

## Requirements

- Python 3.9 or newer
- `numpy >= 1.15.4`
- `scikit-learn`
- `matplotlib`
- `jupyter`

Install everything with:

```bash
pip install -r requirements.txt
```

## How to run

```bash
git clone https://github.com/galsnir/t-sne-notebook.git
cd t-sne-notebook
pip install -r requirements.txt
jupyter notebook t-SNE.ipynb
```

Then run the cells top-to-bottom. The notebook is fully self-contained — no external data files are required (the digits dataset ships with scikit-learn).

## Notes

- A fixed random seed is used for reproducibility.
- The custom implementation is intentionally *simple* (vanilla gradient descent + momentum) — it prioritizes clarity over speed. Expect runtimes of seconds to a couple of minutes depending on the chosen number of iterations and points.
- The notebook discloses where generative AI was used (mainly for rephrasing explanations and helping with plotting boilerplate).
