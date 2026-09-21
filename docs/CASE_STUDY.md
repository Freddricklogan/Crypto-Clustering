# Case Study — Crypto-Clustering

**Repository:** [Crypto-Clustering](https://github.com/Freddricklogan/Crypto-Clustering) · **Live report:** [freddricklogan.github.io/Crypto-Clustering](https://freddricklogan.github.io/Crypto-Clustering/) · **Author:** Freddrick Logan

---

## 1. Who has this problem

Anyone presenting an unsupervised-learning result to someone who will ask "why that many clusters?": analysts reporting segmentation to a manager, students defending a k, and reviewers of a data-science portfolio who know that clustering is where unexamined choices hide. The coins are incidental; the discipline of choosing and reporting is the subject.

## 2. The problem, as a scenario

A reviewer opens the repository. The README says forty-one coins; the CSV has sixteen. The script plots an elbow, then sets `k = 4` on the next line. It clusters PCA scores too, prints both sets of labels, and never compares them. If the CSV were missing it would generate forty random coins and cluster those. There is no measure of cluster quality beyond inertia, no test, and no way to reproduce the figures except by reading terminal output that no longer exists. That was the earlier version of this repository.

## 3. What it costs to leave it alone

A k chosen by eye and hard-coded is a conclusion without an argument; the next person cannot tell whether three would have been better, and on this data the two standard rules do disagree. Presenting a one-coin cluster as a group misleads anyone who reads the table. A truncated dataset silently changes every result. And a fallback that invents data is the kind of defect that, once found, discredits the rest of the work.

## 4. The approach, and the alternative I rejected

I rejected simply fixing the CSV and keeping the script. The value of a clustering study is in the choices it makes visible, so the code became a package whose every choice is a function with a stated rule. `data.py` vendors the full table and validates it. `analysis.py` sweeps k from 1 to 11 with ten seeded initialisations, computes inertia, silhouette and Davies-Bouldin for each, picks k by two rules — the knee of the normalised inertia curve and the maximum silhouette — clusters at both, projects to three principal components with explained variance and loadings, clusters the scores at the same k and reports the adjusted Rand index against the full-feature partition, and runs HDBSCAN so that a method which may decline to assign a point sits beside one that must. `report.py` renders the sweep, the memberships, the scatter, the loadings and the HDBSCAN table as a static page that CI publishes.

## 5. What the code does today

`crypto-clusters report --out dist --seed 42` writes a report and a JSON file. The report states the data and the seed; shows the inertia curves for the scaled features and the PCA scores; tabulates inertia, silhouette and Davies-Bouldin for every k with the knee and the best silhouette marked; lists memberships at both chosen values with cluster sizes, naming singletons as outliers; plots the coins on the first two components coloured by cluster with each coin named on hover; gives the loadings per horizon and the explained variance; and reports HDBSCAN's clusters and noise with the silhouette over assigned points. The Executive Shell carries the headline numbers and a four-step tour.

## 6. Evidence

Nine tests at 100 % statement coverage cover loading and scaling of the 41-coin table, four validation rejections, monotone inertia across the sweep with silhouette bounds, the knee rule on a synthetic curve that bends at four and a flat curve that returns the first k, the silhouette rule, deterministic clustering with the expected label set and the adjusted Rand index, PCA variance ordering and unit-length loadings, HDBSCAN finding two blobs and a noise point, and the report writer's outputs and reproducibility. On seed 42 the knee picks k = 4 with silhouette 0.314 and sizes 13, 1, 26 and 1; the best silhouette is 0.703 at k = 3; three components explain 89.5 % and the PCA-space clustering matches the original exactly; HDBSCAN finds 3 clusters and leaves 24 coins as noise. The report rendered with zero console errors and no horizontal scroll at 1280 or 400 pixels. `AUDIT.md` records nine findings.

## 7. What it would take to run this in production

Nothing here should drive an investment decision, and the page says so. As an analysis pattern it would take a refreshed table from the source API on a schedule, the same validation, a stability check across bootstrap resamples so a cluster's persistence is measured rather than assumed, and a short written interpretation of each cluster reviewed by someone who knows the assets.

## 8. Limits and next steps

One snapshot of forty-one coins is a small table; results are sensitive to a handful of outliers, which is exactly what the singletons and the HDBSCAN noise count reveal. The horizons overlap, so the components are partly a matter of arithmetic. Next, in order: bootstrap stability of the partition, a per-cluster profile table of median changes by horizon, and a second snapshot to see whether membership persists.

## 9. Who should look at this

**Hiring manager:** evidence that I make unsupervised-learning choices explicit, compare methods rather than pick one, and correct my own earlier shortcuts.
**Consulting client:** a template for reporting any segmentation: rules for k, quality measures, memberships, and what the method could not place.
**Engineer:** read `elbow_k` and `silhouette_k` in `src/crypto_clusters/analysis.py` with their synthetic-curve tests.
