# -*- coding: utf-8 -*-
"""Step 3 — Train one LASSO organ aging clock per organ panel.

Following Oh et al. (2023, Nature; 2025, Nat. Med.):

1. Chronological age is predicted from the organ-enriched protein panel with
   LassoCV (5-fold CV over 100 alpha values) fitted on the training set.
2. The penalty is selected with a 95%-of-best rule on cross-validated R²:
   among alphas reaching >= 95% of the best score, the *smallest* alpha is
   used (matching the reference implementation's sparsity choice).
3. The final LASSO is refit on the full training set and evaluated on a
   held-out test set (48%, seed 42).
4. Predicted age is generated for all participants; the age-bias-corrected
   age gap is the residual of predicted age regressed on chronological age;
   it is additionally z-scored for convenience.

Inputs:
    outputs/organs/<Organ>.csv    per-organ protein submatrices (step 2)
    data/raw/ReceptionAge.csv     baseline chronological age (field 21003-0.0)
Outputs (under ``outputs/models/``):
    <Organ>_model.json            coefficients, alpha, train/test metrics
    <Organ>_predictions.csv       eid, actual age, predicted age, age gaps
    model_training_summary.csv    performance table for all clocks
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ORGAN_DIR = Path(os.environ.get("ORGAN_DIR", "outputs/organs"))
AGE_CSV = Path(os.environ.get("AGE_CSV", "data/raw/ReceptionAge.csv"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "outputs/models"))

# 17 organs analysed in the paper plus the extra panels produced by step 2
# (Female/Male/Pituitary clocks were built but are not part of the main analysis;
#  'organismal' uses every non-organ-enriched protein).
ORGANS = ["Adipose", "Adrenal", "Artery", "Brain", "Esophagus", "Female",
          "Heart", "Immune", "Intestine", "Kidney", "Liver", "Lung",
          "Male", "Muscle", "Pancreas", "Pituitary", "Salivary", "Skin",
          "Stomach", "Thyroid", "organismal"]

TEST_SIZE = 0.48
RANDOM_SEED = 42


def train_organ_model(X, y, eid, organ_name):
    X_train, X_test, y_train, y_test, eid_train, eid_test = train_test_split(
        X, y, eid, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    lasso_cv = LassoCV(cv=5, max_iter=10_000, n_alphas=100,
                       random_state=RANDOM_SEED, n_jobs=-1)
    lasso_cv.fit(X_train, y_train)

    # 95%-of-best rule on cross-validated R² (smallest qualifying alpha)
    mse_path = lasso_cv.mse_path_.mean(axis=1)
    r2_path = 1 - mse_path / np.var(y_train)
    target = r2_path.max() * 0.95
    valid = np.where(r2_path >= target)[0]
    alpha = lasso_cv.alphas_[valid[-1]] if len(valid) else lasso_cv.alpha_

    lasso = Lasso(alpha=alpha, max_iter=10_000, random_state=RANDOM_SEED)
    lasso.fit(X_train, y_train)

    train_mae = mean_absolute_error(y_train, lasso.predict(X_train))
    test_mae = mean_absolute_error(y_test, lasso.predict(X_test))
    train_r2 = r2_score(y_train, lasso.predict(X_train))
    test_r2 = r2_score(y_test, lasso.predict(X_test))

    # Predict for everyone; age-bias-corrected gap = residual of pred ~ age
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    eid_all = np.concatenate([eid_train, eid_test])
    y_pred = lasso.predict(X_all)

    lr = LinearRegression().fit(y_all.reshape(-1, 1), y_pred)
    age_gap = y_pred - lr.predict(y_all.reshape(-1, 1))
    age_gap_z = (age_gap - age_gap.mean()) / age_gap.std()

    predictions = pd.DataFrame({
        "eid": eid_all, "age_actual": y_all, "age_predicted": y_pred,
        "age_gap": age_gap, "age_gap_z": age_gap_z,
    })
    return {
        "model": lasso, "alpha": alpha,
        "train_r2": train_r2, "test_r2": test_r2,
        "train_mae": train_mae, "test_mae": test_mae,
        "n_features": X.shape[1],
        "n_selected": int(np.sum(lasso.coef_ != 0)),
        "predictions": predictions,
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    age_df = pd.read_csv(AGE_CSV)
    age_df = age_df[["eid", "21003-0.0"]].rename(columns={"21003-0.0": "age"})
    age_df = age_df.dropna(subset=["age"])

    results = []
    for organ in ORGANS:
        path = ORGAN_DIR / f"{organ}.csv"
        if not path.exists():
            print(f"[skip] {organ}: {path} not found")
            continue
        df = pd.read_csv(path)
        uid_col = df.columns[0]
        df["eid"] = pd.to_numeric(
            df[uid_col].astype(str).str.replace("olink_instance_0.", ""),
            errors="coerce")
        protein_cols = [c for c in df.columns
                        if c.startswith("olink_instance_0.") and c != uid_col]
        merged = df[["eid"] + protein_cols].merge(age_df, on="eid", how="inner")
        if len(merged) < 100:
            print(f"[skip] {organ}: only {len(merged)} samples")
            continue

        X = merged[protein_cols].values
        y = merged["age"].values
        eid = merged["eid"].values
        ok = ~np.isnan(X).any(axis=1)
        X, y, eid = X[ok], y[ok], eid[ok]

        res = train_organ_model(X, y, eid, organ)
        weights = {
            "organ": organ,
            "intercept": float(res["model"].intercept_),
            "coefficients": {
                protein_cols[i]: float(res["model"].coef_[i])
                for i in range(len(protein_cols))
                if res["model"].coef_[i] != 0
            },
            "alpha": float(res["alpha"]),
            "n_features": res["n_features"],
            "n_features_selected": res["n_selected"],
            "train_r2": res["train_r2"], "test_r2": res["test_r2"],
            "train_mae": res["train_mae"], "test_mae": res["test_mae"],
        }
        with open(MODEL_DIR / f"{organ}_model.json", "w",
                  encoding="utf-8") as f:
            json.dump(weights, f, indent=2)
        res["predictions"].to_csv(
            MODEL_DIR / f"{organ}_predictions.csv", index=False)

        results.append({
            "organ": organ, "n_proteins": res["n_features"],
            "n_selected": res["n_selected"], "n_samples": len(eid),
            "alpha": res["alpha"], "train_r2": res["train_r2"],
            "test_r2": res["test_r2"], "train_mae": res["train_mae"],
            "test_mae": res["test_mae"],
        })
        print(f"[done] {organ}: test MAE={res['test_mae']:.2f} "
              f"R2={res['test_r2']:.3f} ({res['n_selected']}/"
              f"{res['n_features']} proteins)")

    pd.DataFrame(results).to_csv(
        MODEL_DIR / "model_training_summary.csv", index=False)
    print(f"Done -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
