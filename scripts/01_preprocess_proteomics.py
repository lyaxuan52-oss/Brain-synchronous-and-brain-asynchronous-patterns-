# -*- coding: utf-8 -*-
"""Step 1 — Proteomics QC, standardisation and KNN imputation.

Reads the raw Olink NPX matrix (one row per participant, one column per
protein), removes low-quality samples/proteins, z-scores each protein and
imputes residual missing values with a k-nearest-neighbour imputer.

The imputer and the z-score statistics are fitted on a random subset of
``K**2`` participants (matching the K=152 neighbourhood size used in the
reference implementation) and then applied to the full cohort.

Inputs (place under ``data/raw/`` or override with environment variables):
    olink_instance0.csv        raw UKB-PPP Olink matrix (~2,900 proteins)
Outputs (under ``outputs/preprocessing/``):
    olink_cleaned.csv          z-scored, KNN-imputed matrix
    olink_cleaning_report.csv  QC summary
    dropped_features.csv       proteins removed (>10% missing)
    dropped_samples.csv        participants removed (>1,000 missing proteins)
    imputation_report.csv      3%-masking imputation MAE check
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# --------------------------------------------------------------------------- #
# Configuration (override via environment variables)
# --------------------------------------------------------------------------- #
MATRIX_CSV = Path(os.environ.get(
    "OLINK_MATRIX_CSV", "data/raw/olink_instance0.csv"))
OUT_DIR = Path(os.environ.get(
    "PREPROCESS_OUT_DIR", "outputs/preprocessing"))

SAMPLE_MISSING_MAX = 1000        # drop participants with >1000 missing proteins
PROTEIN_MISSING_MAX = 0.10       # drop proteins with >10% missing values
RANDOM_SEED = 42
TARGET_K = 152                   # KNN neighbours; subset size = K**2 = 23,104


def qc_and_impute(matrix_csv: Path):
    print("[1/6] Reading protein matrix ...")
    df = pd.read_csv(matrix_csv, low_memory=False)
    print(f"    shape={df.shape}")
    uid = df.iloc[:, 0]
    feats = df.columns[1:].tolist()

    print("[2/6] Converting to numeric (float32) ...")
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    print("[3/6] Sample filter: >", SAMPLE_MISSING_MAX, "missing proteins")
    miss_count = np.isnan(X.values).sum(axis=1)
    keep_rows = miss_count <= SAMPLE_MISSING_MAX
    dropped_uids = uid.loc[~keep_rows].astype(str).tolist()
    X = X.loc[keep_rows].reset_index(drop=True)
    uid = uid.loc[keep_rows].reset_index(drop=True)
    print(f"    kept {len(uid)} samples (dropped {len(dropped_uids)})")

    print("[4/6] Protein filter: >", PROTEIN_MISSING_MAX, "missing rate")
    miss_rate = np.isnan(X.values).mean(axis=0)
    keep_cols = miss_rate <= PROTEIN_MISSING_MAX
    dropped_feats = [f for f, k in zip(feats, keep_cols) if not k]
    X = X.loc[:, keep_cols]
    print(f"    kept {X.shape[1]} proteins (dropped {len(dropped_feats)})")

    print("[5/6] Z-score + KNN imputation (subset-fitted) ...")
    n_total = len(X)
    rng = np.random.RandomState(RANDOM_SEED)
    n_train = min(TARGET_K ** 2, n_total)
    train_idx = rng.choice(n_total, size=n_train, replace=False)
    is_train = np.zeros(n_total, dtype=bool)
    is_train[train_idx] = True

    means = np.nanmean(X.values[is_train], axis=0)
    stds = np.nanstd(X.values[is_train], axis=0)
    stds[stds == 0] = 1.0

    Z_all = (X.values - means) / stds
    imputer = KNNImputer(n_neighbors=TARGET_K)
    imputer.fit(Z_all[is_train])

    blocks = []
    for start in range(0, n_total, 1000):
        end = min(start + 1000, n_total)
        blocks.append(imputer.transform(Z_all[start:end]))
        print(f"    imputed rows {end}/{n_total}")
    Z_imp = pd.DataFrame(np.vstack(blocks), columns=X.columns)

    cleaned = pd.concat([uid.reset_index(drop=True),
                         Z_imp.reset_index(drop=True)], axis=1)
    cleaned.columns = [df.columns[0]] + list(Z_imp.columns)

    print("[6/6] Writing outputs ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_DIR / "olink_cleaned.csv", index=False)

    info = {
        "n_samples_before": int(len(df)),
        "n_samples_after": int(len(cleaned)),
        "n_features_before": int(len(feats)),
        "n_features_after": int(Z_imp.shape[1]),
        "knn_k": TARGET_K,
        "knn_train_n": int(n_train),
        "random_seed": RANDOM_SEED,
    }
    pd.DataFrame([info]).to_csv(OUT_DIR / "olink_cleaning_report.csv",
                                index=False)
    if dropped_feats:
        (pd.Series(miss_rate, index=feats, name="missing_rate")
           .loc[dropped_feats]
           .to_csv(OUT_DIR / "dropped_features.csv", header=True))
    if dropped_uids:
        (pd.DataFrame({"sample_id": dropped_uids,
                       "reason": "missing_count_gt_threshold"})
           .to_csv(OUT_DIR / "dropped_samples.csv", index=False))

    # Imputation sanity check: mask 3% of observed values, compare MAE in z space
    try:
        non_missing_rows = ~np.isnan(X.values).any(axis=1)
        idx_all = np.where(non_missing_rows)[0]
        rng2 = np.random.default_rng(RANDOM_SEED)
        if len(idx_all):
            G = X.iloc[idx_all].values.astype(float)
            mask = rng2.random(G.shape) < 0.03
            G_mask = G.copy()
            G_mask[mask] = np.nan
            Zm = (G_mask - means) / stds
            Zm_imp = KNNImputer(n_neighbors=TARGET_K).fit_transform(Zm)
            Z_true = (G - means) / stds
            mae = float(np.nanmean(np.abs(Zm_imp[mask] - Z_true[mask])))
        else:
            mae = np.nan
    except Exception:
        mae = np.nan
    pd.DataFrame([{"mae_all": mae}]).to_csv(
        OUT_DIR / "imputation_report.csv", index=False)

    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    qc_and_impute(MATRIX_CSV)
