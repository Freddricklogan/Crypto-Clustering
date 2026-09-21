# AUDIT — Crypto-Clustering (pre-refactor)

Audit of the previous build: `crypto_clustering.py` (190 lines, a
script that printed to a terminal and saved PNGs), a truncated
`Resources/crypto_market_data.csv` of 16 coins, and three images. No
tests, no package, no CI.

---

## A. Data

### A1 — Sixteen coins where the dataset has forty-one
The committed CSV held 16 rows; the exercise's dataset, and the README's
description, cover 41 coins. **Fix:** the full 41-coin table is vendored
in `data/crypto_market_data.csv` with its provenance stated, and the
loader validates the seven columns, missing values, duplicates and a
minimum size.

### A2 — A fallback that invented data
`generate_sample_data()` (line 15) produced 40 uniformly random
"Crypto_N" rows if the CSV was absent, and the script would have
clustered them without saying so. **Fix:** removed; a missing or malformed
file is an error with a message.

## B. Method

### B1 — k chosen by eye
The script plotted an elbow curve and then set `k = 4` as a constant
(line 100). **Fix:** two stated rules — the knee of the inertia curve and
the maximum silhouette — both computed and both reported with their
memberships; on this data they disagree (4 and 3), and the page says so.

### B2 — No cluster quality measure
Inertia only. **Fix:** silhouette and Davies–Bouldin for every k in the
sweep, in both feature spaces.

### B3 — PCA result asserted, not compared
The script clustered PCA scores and printed labels; nothing compared
them with the full-feature clustering. **Fix:** the adjusted Rand index
between the two labelings (1.000 on this run) and a loadings table so
the components can be read.

### B4 — Singletons presented as clusters
At k = 4, two "clusters" hold one coin each (`celsius-degree-token`,
`ethlend`); the old output listed them like any other group. **Fix:**
the report counts singletons and calls them outliers the method has
isolated; HDBSCAN is run alongside because it may leave points
unassigned, and it leaves 24 of 41.

### B5 — `KMeans` without `n_init`
Default initialisation varied between scikit-learn versions. **Fix:**
`n_init=10` with an explicit seed; a test runs the analysis twice and
asserts identical labels.

## C. Engineering

### C1 — Script, prints, PNGs
**Fix:** an installable package with a `crypto-clusters report` command
that writes a static report (SVG charts, no image files) and
`report.json`; CI builds and publishes it.

### C2 — No tests, no types, no CI
**Fix:** 9 pytest tests at 100 % statement coverage (validation
rejections, monotone inertia, the knee and silhouette rules on synthetic
curves, determinism, PCA variance and unit loadings, HDBSCAN noise);
ruff, mypy strict, bandit, pip-audit, Trivy.
