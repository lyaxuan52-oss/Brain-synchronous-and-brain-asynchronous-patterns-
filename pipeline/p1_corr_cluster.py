# -*- coding: utf-8 -*-
"""Part 1: characterise plasma-proteomic organ age gaps.
Computes inter-organ correlation matrix (Winsorized) + hierarchical clustering,
prints mean pairwise r, cluster membership, strongest pairs. Grounds report text."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

BASE = Path(os.environ.get("ORGAN_AGE_DATA_DIR",
                           Path(__file__).resolve().parent.parent / "data"))
df = pd.read_csv(BASE / "merged_age_gap.csv")

organ_cols = [c for c in df.columns if c.endswith("_age_gap")
              and not c.startswith(("Female", "Male"))]
organs = [c.replace("_age_gap", "") for c in organ_cols]
X = df[organ_cols].copy()
X.columns = organs

# Winsorize +/- 3 SD per organ (consistent with M0)
for c in X.columns:
    m, s = X[c].mean(), X[c].std()
    X[c] = X[c].clip(m - 3 * s, m + 3 * s)

R = X.corr(method="pearson")
n = len(X)

# mean pairwise r (off-diagonal)
iu = np.triu_indices_from(R.values, k=1)
pairwise = R.values[iu]
print(f"N = {n}")
print(f"n organs = {len(organs)}")
print(f"mean pairwise r = {pairwise.mean():.3f}")
print(f"median pairwise r = {np.median(pairwise):.3f}")
print(f"range r = [{pairwise.min():.3f}, {pairwise.max():.3f}]")
print(f"frac positive = {(pairwise>0).mean():.3f}")

# strongest positive pairs
pairs = []
for i in range(len(organs)):
    for j in range(i + 1, len(organs)):
        pairs.append((organs[i], organs[j], R.values[i, j]))
pairs.sort(key=lambda t: -t[2])
print("\nTop 12 strongest positive pairs:")
for a, b, r in pairs[:12]:
    print(f"  {a:10s} - {b:10s} r={r:+.3f}")
print("\nMost negative pairs:")
for a, b, r in pairs[-6:]:
    print(f"  {a:10s} - {b:10s} r={r:+.3f}")

# hierarchical clustering on 1 - r distance
D = 1 - R.values
np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")
for k in [3, 4, 5]:
    lab = fcluster(Z, k, criterion="maxclust")
    print(f"\n=== {k} clusters ===")
    for cl in sorted(set(lab)):
        members = [organs[i] for i in range(len(organs)) if lab[i] == cl]
        # mean within-cluster r
        idx = [i for i in range(len(organs)) if lab[i] == cl]
        if len(idx) > 1:
            sub = R.values[np.ix_(idx, idx)]
            wr = sub[np.triu_indices_from(sub, 1)].mean()
        else:
            wr = float("nan")
        print(f"  C{cl} (within-r={wr:+.3f}): {members}")

R.to_csv(BASE + r"\code\_p1_corr_matrix.csv")
print("\nsaved _p1_corr_matrix.csv")
