from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from crypto_clusters.analysis import (
    agreement,
    cluster,
    density,
    elbow_k,
    project,
    silhouette_k,
    sweep_k,
)
from crypto_clusters.data import FEATURES, load

DATA = Path(__file__).resolve().parents[1] / "data" / "crypto_market_data.csv"


@pytest.fixture(scope="module")
def table():  # type: ignore[no-untyped-def]
    return load(DATA)


def test_load_and_scale(table) -> None:  # type: ignore[no-untyped-def]
    assert len(table.coins) == 41 and table.coins[0] == "bitcoin"
    assert table.raw.shape == (41, 7) and table.scaled.shape == (41, 7)
    assert np.allclose(table.scaled.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(table.scaled.std(axis=0), 1, atol=1e-9)


def test_load_rejections(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_text("coin_id,price_change_percentage_24h\nx,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load(p)
    head = "coin_id," + ",".join(FEATURES) + "\n"
    q = tmp_path / "b.csv"
    q.write_text(head + "a,1,2,3,4,5,6,\n" + "b,1,2,3,4,5,6,7\n" + "c,1,2,3,4,5,6,7\n")
    with pytest.raises(ValueError, match="missing values"):
        load(q)
    r = tmp_path / "c.csv"
    r.write_text(head + "a,1,2,3,4,5,6,7\nb,2,3,4,5,6,7,8\n")
    with pytest.raises(ValueError, match="at least 3"):
        load(r)
    s = tmp_path / "d.csv"
    s.write_text(head + "a,1,2,3,4,5,6,7\na,2,3,4,5,6,7,8\nc,3,4,5,6,7,8,9\n")
    with pytest.raises(ValueError, match="duplicate"):
        load(s)


def test_sweep_is_monotone_and_rules_pick_within_range(table) -> None:  # type: ignore[no-untyped-def]
    sw = sweep_k(table.scaled, range(1, 12), seed=42)
    assert sw.ks == list(range(1, 12))
    for a, b in pairwise(sw.inertia):
        assert b <= a + 1e-9  # more clusters never increase inertia
    assert np.isnan(sw.silhouette[0]) and np.isnan(sw.davies_bouldin[0])
    assert all(-1 <= s <= 1 for s in sw.silhouette[1:])
    ke, ks = elbow_k(sw), silhouette_k(sw)
    assert 2 <= ke <= 11 and 2 <= ks <= 11
    assert (
        elbow_k(sweep_k(table.scaled, range(1, 3))) == 1
    )  # too few points for a second difference


def test_elbow_rule_on_a_synthetic_curve() -> None:
    from crypto_clusters.analysis import KSweep

    # Inertia falls steadily to k=4, then flattens: the knee is at k=4.
    nan6 = [float("nan")] * 6
    sw = KSweep([1, 2, 3, 4, 5, 6], [100, 80, 60, 20, 18, 17], nan6, nan6)
    assert elbow_k(sw) == 4
    flat = KSweep([1, 2, 3], [5, 5, 5], [float("nan")] * 3, [float("nan")] * 3)
    assert elbow_k(flat) == 1
    sw2 = KSweep([2, 3, 4], [1, 1, 1], [0.1, 0.5, 0.2], [1, 1, 1])
    assert silhouette_k(sw2) == 3


def test_cluster_is_deterministic_and_scored(table) -> None:  # type: ignore[no-untyped-def]
    a = cluster(table.scaled, 4, seed=42)
    b = cluster(table.scaled, 4, seed=42)
    assert (a.labels == b.labels).all()
    assert set(a.labels.tolist()) == {0, 1, 2, 3}
    assert -1 <= a.silhouette <= 1 and a.davies_bouldin > 0
    assert np.isnan(cluster(table.scaled, 1).silhouette)
    assert agreement(a.labels, a.labels) == 1.0
    assert agreement(a.labels, np.roll(a.labels, 1)) < 1.0


def test_projection_variance_and_loadings(table) -> None:  # type: ignore[no-untyped-def]
    p = project(table.scaled, 3, seed=42)
    assert p.scores.shape == (41, 3) and p.loadings.shape == (3, 7)
    assert 0 < sum(p.explained) <= 1
    assert p.explained == sorted(p.explained, reverse=True)
    assert all(abs(np.linalg.norm(row) - 1) < 1e-9 for row in p.loadings)  # unit-length components
    full = project(table.scaled, 7)
    assert abs(sum(full.explained) - 1) < 1e-9


def test_density_finds_clusters_and_noise() -> None:
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(0, 0.2, (15, 2)), rng.normal(5, 0.2, (15, 2)), [[20, 20]]])
    d = density(blobs, min_cluster_size=3)
    assert d.clusters == 2 and d.noise >= 1
    assert d.labels[-1] == -1
    assert 0 < d.silhouette <= 1
    one = density(np.vstack([rng.normal(0, 0.1, (10, 2))]), min_cluster_size=3)
    assert np.isnan(one.silhouette)
