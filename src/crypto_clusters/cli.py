"""Command-line entry point: `crypto-clusters report --out dist`."""

from __future__ import annotations

from pathlib import Path

import typer

from .report import write_report

app = typer.Typer(add_completion=False, help="Cryptocurrency clustering report.")
DEFAULT_DATA = Path("data") / "crypto_market_data.csv"


@app.callback()
def main() -> None:
    """Cryptocurrency clustering report."""


@app.command()
def report(
    out: Path = typer.Option(Path("dist"), help="Output directory for the static report."),
    data: Path = typer.Option(
        DEFAULT_DATA, help="CSV with coin_id and the seven price-change columns."
    ),
    seed: int = typer.Option(42, help="Seed for K-means initialisation and PCA."),
) -> None:
    """Run the K-means / PCA / HDBSCAN analysis and write the report."""
    path = write_report(out, data, seed=seed)
    print(f"wrote {path}")


if __name__ == "__main__":
    app()
