import numpy as np


class KMeans:
    """
    Hand-rolled K-Means clustering.

    Parameters
    ----------
    k        : number of clusters
    max_iter : maximum number of assign/update iterations
    tol      : convergence threshold — stop when max centroid shift < tol
    seed     : random seed for reproducible centroid initialisation
    """

    def __init__(self, k: int = 3, max_iter: int = 300, tol: float = 1e-4, seed: int = None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self.centroids_: np.ndarray = None   # shape (k, 2)
        self.labels_: np.ndarray = None      # shape (N,)
        self.inertia_: float = None          # WCSS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Euclidean distances from every point in a (N, d) to every point in
        b (M, d).  Returns matrix of shape (N, M).
        Computed entirely with numpy — no scipy or sklearn.
        """
        # diff[i, j] = a[i] - b[j]  →  shape (N, M, d)
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def _assign(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid. Returns labels (N,)."""
        dist = self._euclidean(X, self.centroids_)  # (N, k)
        return np.argmin(dist, axis=1)

    def _update(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recompute centroids as the mean of their assigned points."""
        new_centroids = np.zeros_like(self.centroids_)
        for j in range(self.k):
            members = X[labels == j]
            if len(members) == 0:
                # Empty cluster: reinitialise to a random point
                rng = np.random.default_rng(self.seed)
                new_centroids[j] = X[rng.integers(len(X))]
            else:
                new_centroids[j] = members.mean(axis=0)
        return new_centroids

    def _wcss(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Within-cluster sum of squares (inertia)."""
        total = 0.0
        for j in range(self.k):
            members = X[labels == j]
            if len(members):
                diff = members - self.centroids_[j]
                total += (diff ** 2).sum()
        return float(total)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-Means on X (N, 2). Stores .labels_, .centroids_, .inertia_."""
        if len(X) < self.k:
            raise ValueError(f"Need at least {self.k} points to form {self.k} clusters.")

        rng = np.random.default_rng(self.seed)
        # Random initialisation: pick k distinct rows from X
        indices = rng.choice(len(X), size=self.k, replace=False)
        self.centroids_ = X[indices].copy()

        for _ in range(self.max_iter):
            labels = self._assign(X)
            new_centroids = self._update(X, labels)

            # Convergence check: max shift across all centroids
            shift = np.sqrt(((new_centroids - self.centroids_) ** 2).sum(axis=1)).max()
            self.centroids_ = new_centroids
            if shift < self.tol:
                break

        self.labels_ = self._assign(X)
        self.inertia_ = self._wcss(X, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest fitted centroid."""
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._assign(X)
