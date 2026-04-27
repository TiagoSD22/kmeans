"""
Elbow and Silhouette methods to determine optimal K.
All maths hand-rolled with numpy — no sklearn/scipy.
"""

import numpy as np
from kmeans import KMeans


# ---------------------------------------------------------------------------
# Elbow
# ---------------------------------------------------------------------------

def elbow(X: np.ndarray, k_range: range = range(2, 11)):
    """
    Fit KMeans for each k in k_range, collect WCSS (inertia).
    Returns (ks, inertias) as plain lists.
    """
    ks, inertias = [], []
    for k in k_range:
        if k > len(X):
            break
        model = KMeans(k=k, seed=42).fit(X)
        ks.append(k)
        inertias.append(model.inertia_)
    return ks, inertias


def find_knee(ks: list, inertias: list) -> int:
    """
    Detect the elbow as the point of maximum second difference (curvature).
    Returns the optimal k value.
    """
    if len(ks) < 3:
        return ks[0]
    inertias_arr = np.array(inertias, dtype=float)
    second_diff = np.diff(inertias_arr, n=2)   # length = len(ks) - 2
    # index into ks is offset by 1 (second diff starts at ks[1])
    knee_idx = int(np.argmax(np.abs(second_diff))) + 1
    return ks[knee_idx]


# ---------------------------------------------------------------------------
# Silhouette (hand-rolled)
# ---------------------------------------------------------------------------

def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Mean silhouette coefficient over all points.

    For each point i:
      a(i) = mean distance to other points in the same cluster
      b(i) = min over other clusters of mean distance to points in that cluster
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    All distances computed with numpy only.
    """
    n = len(X)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    # Pairwise distance matrix (N x N) — hand-rolled
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]   # (N, N, d)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))       # (N, N)

    scores = np.zeros(n)
    for i in range(n):
        label_i = labels[i]
        same_mask = (labels == label_i)
        same_mask[i] = False  # exclude self

        # a(i): mean intra-cluster distance
        if same_mask.sum() == 0:
            a = 0.0
        else:
            a = dist_matrix[i, same_mask].mean()

        # b(i): min mean distance to any other cluster
        b_values = []
        for lbl in unique_labels:
            if lbl == label_i:
                continue
            other_mask = labels == lbl
            b_values.append(dist_matrix[i, other_mask].mean())
        b = min(b_values)

        denom = max(a, b)
        scores[i] = (b - a) / denom if denom > 0 else 0.0

    return float(scores.mean())


def silhouette(X: np.ndarray, k_range: range = range(2, 11)):
    """
    Compute mean silhouette score for each k in k_range.
    Returns (ks, scores) as plain lists.
    """
    ks, scores = [], []
    for k in k_range:
        if k >= len(X):
            break
        model = KMeans(k=k, seed=42).fit(X)
        score = _silhouette_score(X, model.labels_)
        ks.append(k)
        scores.append(score)
    return ks, scores


def best_silhouette_k(ks: list, scores: list) -> int:
    """Return the k with the highest silhouette score."""
    return ks[int(np.argmax(scores))]
