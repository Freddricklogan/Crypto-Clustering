"""Static report — the Pages artefact. Runs the full analysis and renders tables and SVG charts;
every number on the page comes from this run."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template

import numpy as np

from .analysis import (
    Clustering,
    Density,
    KSweep,
    Projection,
    agreement,
    cluster,
    density,
    elbow_k,
    project,
    silhouette_k,
    sweep_k,
)
from .data import DATA_SOURCE, FEATURES, Table, load

PKG = Path(__file__).parent
SHELL_DIR = PKG / "shell"
TEMPLATES = PKG / "templates"
COLOURS = ["#58A6FF", "#3fb950", "#d29922", "#f85149", "#d2a8ff", "#79c0ff", "#ffa657", "#8b98b0"]
HORIZONS = ["24h", "7d", "14d", "30d", "60d", "200d", "1y"]


@dataclass(frozen=True)
class Result:
    table: Table
    sweep: KSweep
    elbow: int
    best_sil: int
    chosen: Clustering
    alt: Clustering
    projection: Projection
    sweep_pca: KSweep
    chosen_pca: Clustering
    ari: float
    hdb: Density
    seed: int


def run(data: Path, seed: int = 42, ks: range = range(1, 12)) -> Result:
    t = load(data)
    sw = sweep_k(t.scaled, ks, seed)
    ke, ks_ = elbow_k(sw), silhouette_k(sw)
    chosen = cluster(t.scaled, ke, seed)
    alt = cluster(t.scaled, ks_, seed) if ks_ != ke else chosen
    proj = project(t.scaled, 3, seed)
    sw_pca = sweep_k(proj.scores, ks, seed)
    chosen_pca = cluster(proj.scores, ke, seed)
    return Result(
        table=t,
        sweep=sw,
        elbow=ke,
        best_sil=ks_,
        chosen=chosen,
        alt=alt,
        projection=proj,
        sweep_pca=sw_pca,
        chosen_pca=chosen_pca,
        ari=agreement(chosen.labels, chosen_pca.labels),
        hdb=density(t.scaled, 3),
        seed=seed,
    )


def _row(cells: list[str], head: bool = False) -> str:
    tag = "th" if head else "td"
    return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"


def _f(v: float, d: int = 3) -> str:
    return "—" if np.isnan(v) else f"{v:.{d}f}"


def _sweep_table(a: KSweep, b: KSweep, elbow: int, best: int) -> str:
    rows = [
        _row(
            ["k", "Inertia", "Silhouette", "Davies-Bouldin", "Inertia (PCA)", "Silhouette (PCA)"],
            head=True,
        )
    ]
    for i, k in enumerate(a.ks):
        mark = " (elbow)" if k == elbow else (" (best silhouette)" if k == best else "")
        rows.append(
            _row(
                [
                    f"{k}{mark}",
                    _f(a.inertia[i], 1),
                    _f(a.silhouette[i]),
                    _f(a.davies_bouldin[i]),
                    _f(b.inertia[i], 1),
                    _f(b.silhouette[i]),
                ]
            )
        )
    return f"<table>{''.join(rows)}</table>"


def _members_table(t: Table, c: Clustering) -> str:
    rows = [_row(["Cluster", "Coins", "Members"], head=True)]
    for k in range(c.k):
        idx = np.where(c.labels == k)[0]
        rows.append(
            _row(
                [
                    f'<span class="cc-swatch" data-c="{k % len(COLOURS)}"></span> {k}',
                    str(len(idx)),
                    html.escape(", ".join(t.coins[i] for i in idx)),
                ]
            )
        )
    return f"<table>{''.join(rows)}</table>"


def _loadings_table(p: Projection) -> str:
    rows = [
        _row(
            ["Horizon", *[f"PC{i + 1} ({v * 100:.1f}%)" for i, v in enumerate(p.explained)]],
            head=True,
        )
    ]
    for j, h in enumerate(HORIZONS):
        rows.append(_row([h, *[f"{p.loadings[i, j]:+.3f}" for i in range(p.components)]]))
    return f"<table>{''.join(rows)}</table>"


def _hdb_table(t: Table, d: Density) -> str:
    rows = [_row(["HDBSCAN cluster", "Coins", "Members"], head=True)]
    for k in sorted(set(d.labels.tolist())):
        idx = np.where(d.labels == k)[0]
        name = "noise (-1)" if k < 0 else str(k)
        rows.append(_row([name, str(len(idx)), html.escape(", ".join(t.coins[i] for i in idx))]))
    return f"<table>{''.join(rows)}</table>"


def _tick(x: float, y: float, text: str, anchor: str = "start") -> str:
    return f'<text class="cc-tick" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{text}</text>'


def _axis(pad: int, width: int, height: int) -> str:
    bottom = height - pad
    return f'<path class="cc-axis" d="M{pad},{pad} L{pad},{bottom} L{width - pad},{bottom}"/>'


def _scatter(t: Table, p: Projection, c: Clustering, width: int = 640, height: int = 360) -> str:
    pad = 36
    xs, ys = p.scores[:, 0], p.scores[:, 1]
    x_lo, x_hi, y_lo, y_hi = xs.min(), xs.max(), ys.min(), ys.max()

    def sx(x: float) -> float:
        return pad + (x - x_lo) / (x_hi - x_lo) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - y_lo) / (y_hi - y_lo) * (height - 2 * pad)

    pts = []
    for i, coin in enumerate(t.coins):
        col = COLOURS[int(c.labels[i]) % len(COLOURS)]
        title = html.escape(f"{coin} - cluster {int(c.labels[i])}")
        pts.append(
            f'<circle cx="{sx(xs[i]):.1f}" cy="{sy(ys[i]):.1f}" r="5" fill="{col}" '
            f'fill-opacity="0.85"><title>{title}</title></circle>'
        )
    labels = _tick(width - pad, height - pad + 14, f"PC1 ({p.explained[0] * 100:.1f}%)", "end")
    labels += _tick(pad, pad - 8, f"PC2 ({p.explained[1] * 100:.1f}%)")
    aria = "Coins on the first two principal components, coloured by K-means cluster"
    return (
        f'<svg class="cc-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{aria}">'
        f"{_axis(pad, width, height)}{''.join(pts)}{labels}</svg>"
    )


def _elbow_chart(a: KSweep, b: KSweep, width: int = 640, height: int = 240) -> str:
    pad = 40
    top = max(a.inertia[0], b.inertia[0])

    def sx(k: float) -> float:
        return pad + (k - a.ks[0]) / (a.ks[-1] - a.ks[0]) * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - v / top * (height - 2 * pad)

    def line(sw: KSweep, cls: str) -> str:
        pts = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in zip(sw.ks, sw.inertia, strict=True))
        return f'<polyline class="{cls}" points="{pts}"/>'

    ticks = "".join(_tick(sx(k), height - pad + 14, str(k), "middle") for k in a.ks)
    legend = (
        f'<text class="cc-tick" x="{width - pad}" y="{pad - 8}" text-anchor="end">'
        '<tspan class="cc-line-a-t">scaled features</tspan>  '
        '<tspan class="cc-line-b-t">PCA scores</tspan></text>'
    )
    return (
        f'<svg class="cc-chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Inertia by k for both feature spaces">{_axis(pad, width, height)}'
        f"{line(a, 'cc-line-a')}{line(b, 'cc-line-b')}{ticks}{legend}</svg>"
    )


def kpi_json(r: Result) -> dict[str, object]:
    return {
        "coins": len(r.table.coins),
        "elbowK": r.elbow,
        "silhouetteK": r.best_sil,
        "silhouetteAtElbow": r.chosen.silhouette,
        "silhouetteAtBest": r.alt.silhouette,
        "explained": r.projection.explained,
        "ari": r.ari,
        "hdbClusters": r.hdb.clusters,
        "hdbNoise": r.hdb.noise,
        "sizes": [int((r.chosen.labels == k).sum()) for k in range(r.chosen.k)],
        "seed": r.seed,
        "source": DATA_SOURCE,
    }


def render_html(r: Result, pages: str) -> str:
    tpl = Template((TEMPLATES / "page.html").read_text(encoding="utf-8"))
    sizes = [int((r.chosen.labels == k).sum()) for k in range(r.chosen.k)]
    singletons = sum(1 for s in sizes if s == 1)
    return tpl.substitute(
        pages=html.escape(pages),
        source=html.escape(DATA_SOURCE),
        coins=str(len(r.table.coins)),
        features=str(len(FEATURES)),
        seed=str(r.seed),
        elbow=str(r.elbow),
        best=str(r.best_sil),
        sil_elbow=_f(r.chosen.silhouette),
        sil_best=_f(r.alt.silhouette),
        singletons=str(singletons),
        sweep_table=_sweep_table(r.sweep, r.sweep_pca, r.elbow, r.best_sil),
        elbow_chart=_elbow_chart(r.sweep, r.sweep_pca),
        members=_members_table(r.table, r.chosen),
        members_alt=_members_table(r.table, r.alt),
        explained=", ".join(f"{v * 100:.1f}%" for v in r.projection.explained),
        explained_total=f"{sum(r.projection.explained) * 100:.1f}%",
        loadings=_loadings_table(r.projection),
        scatter=_scatter(r.table, r.projection, r.chosen),
        ari=_f(r.ari),
        hdb_clusters=str(r.hdb.clusters),
        hdb_noise=str(r.hdb.noise),
        hdb_sil=_f(r.hdb.silhouette),
        hdb_min=str(r.hdb.min_cluster_size),
        hdb_table=_hdb_table(r.table, r.hdb),
        report_json=json.dumps(kpi_json(r)),
    )


def write_report(
    out: Path,
    data: Path,
    seed: int = 42,
    pages: str = "https://freddricklogan.github.io/Crypto-Clustering/",
) -> Path:
    r = run(data, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "src").mkdir(exist_ok=True)
    shutil.copy(SHELL_DIR / "exec-shell.css", out / "src" / "exec-shell.css")
    shutil.copy(SHELL_DIR / "exec-shell.js", out / "src" / "exec-shell.js")
    shutil.copy(PKG / "report.js", out / "src" / "report.js")
    shutil.copy(TEMPLATES / "report.css", out / "src" / "report.css")
    (out / "index.html").write_text(render_html(r, pages), encoding="utf-8")
    (out / "report.json").write_text(json.dumps(kpi_json(r), indent=2), encoding="utf-8")
    return out / "index.html"
