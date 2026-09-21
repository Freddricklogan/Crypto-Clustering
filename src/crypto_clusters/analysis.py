"""K-means, PCA and HDBSCAN over the scaled table. Every choice (k, components) is made by a stated
rule and returned with the evidence behind it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class KSweep:
    ks: list[int]
    inertia: list[float]
    silhouette: list[float]  # NaN for k = 1
    davies_bouldin: list[float]  # NaN for k = 1


@dataclass(frozen=True)
class Clustering:
    k: int
    labels: IntArray
    inertia: float
    silhouette: float
    davies_bouldin: float


@dataclass(frozen=True)
class Projection:
    components: int
    scores: FloatArray  # rows = coins
    explained: list[float]
    loadings: FloatArray  # components by features


@dataclass(frozen=True)
class Density:
    labels: IntArray  # -1 = noise
    clusters: int
    noise: int
    silhouette: float  # NaN when fewer than two clusters among non-noise points
    min_cluster_size: int


def _kmeans(x: FloatArray, k: int, seed: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)


def sweep_k(x: FloatArray, ks: range, seed: int = 42) -> KSweep:
    inertia: list[float] = []
    sil: list[float] = []
    dbi: list[float] = []
    for k in ks:
        model = _kmeans(x, k, seed)
        inertia.append(float(model.inertia_))
        if k >= 2 and k < len(x):
            sil.append(float(silhouette_score(x, model.labels_)))
            dbi.append(float(davies_bouldin_score(x, model.labels_)))
        else:
            sil.append(float("nan"))
            dbi.append(float("nan"))
    return KSweep(list(ks), inertia, sil, dbi)


def elbow_k(sweep: KSweep) -> int:
    """Knee rule: after scaling k and inertia to [0, 1], the k whose point lies farthest below the
    chord from the first to the last point (the Kneedle idea, without smoothing)."""
    y = np.asarray(sweep.inertia, dtype=np.float64)
    x = np.asarray(sweep.ks, dtype=np.float64)
    if len(y) < 3 or y[0] == y[-1]:
        return sweep.ks[0]
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[-1]) / (y[0] - y[-1])
    chord = 1 - xn  # the line from (0, 1) to (1, 0)
    return sweep.ks[int(np.argmax(chord - yn))]


def silhouette_k(sweep: KSweep) -> int:
    """k with the highest silhouette among k >= 2."""
    s = np.asarray(sweep.silhouette)
    valid = np.where(~np.isnan(s))[0]
    return sweep.ks[int(valid[np.argmax(s[valid])])]


def cluster(x: FloatArray, k: int, seed: int = 42) -> Clustering:
    model = _kmeans(x, k, seed)
    labels = np.asarray(model.labels_, dtype=np.int64)
    return Clustering(
        k=k,
        labels=labels,
        inertia=float(model.inertia_),
        silhouette=float(silhouette_score(x, labels)) if 2 <= k < len(x) else float("nan"),
        davies_bouldin=float(davies_bouldin_score(x, labels)) if 2 <= k < len(x) else float("nan"),
    )


def project(x: FloatArray, components: int = 3, seed: int = 42) -> Projection:
    pca = PCA(n_components=components, random_state=seed).fit(x)
    return Projection(
        components=components,
        scores=np.asarray(pca.transform(x), dtype=np.float64),
        explained=[float(v) for v in pca.explained_variance_ratio_],
        loadings=np.asarray(pca.components_, dtype=np.float64),
    )


def density(x: FloatArray, min_cluster_size: int = 3) -> Density:
    model = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit(x)
    labels = np.asarray(model.labels_, dtype=np.int64)
    mask = labels >= 0
    n_clusters = len(set(labels[mask].tolist()))
    sil = float("nan")
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = float(silhouette_score(x[mask], labels[mask]))
    return Density(labels, n_clusters, int((~mask).sum()), sil, min_cluster_size)


def agreement(a: IntArray, b: IntArray) -> float:
    """Adjusted Rand index between two labelings (1 = identical partition, ~0 = chance)."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))
