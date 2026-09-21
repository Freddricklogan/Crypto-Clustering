import json
from pathlib import Path

from crypto_clusters.report import run, write_report

DATA = Path(__file__).resolve().parents[1] / "data" / "crypto_market_data.csv"


def test_write_report(tmp_path: Path) -> None:
    out = write_report(tmp_path / "dist", DATA, seed=42)
    assert out.exists()
    page = out.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in page and "onclick" not in page and "style=" not in page
    assert "$" not in page.replace("$pages", "")  # every template field substituted
    data = json.loads((tmp_path / "dist" / "report.json").read_text())
    assert data["coins"] == 41 and 2 <= data["elbowK"] <= 11 and 2 <= data["silhouetteK"] <= 11
    assert sum(data["sizes"]) == 41 and len(data["explained"]) == 3
    assert -1 <= data["ari"] <= 1 and data["hdbClusters"] + data["hdbNoise"] >= 1


def test_run_is_reproducible_for_a_seed() -> None:
    a, b = run(DATA, seed=7), run(DATA, seed=7)
    assert (a.chosen.labels == b.chosen.labels).all()
    assert a.elbow == b.elbow and a.best_sil == b.best_sil
    assert a.chosen_pca.k == a.elbow
