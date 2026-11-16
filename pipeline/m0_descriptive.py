# -*- coding: utf-8 -*-
"""Module 0: descriptive statistics, sample composition and QC."""
import numpy as np
import pandas as pd
from scipy import stats
import config as C
from config import gap_col
import data_io as io
import viz


def run(master, reg, wlog, dcols):
    out = C.MODULE_DIRS["M0"]
    T, F = out / "tables", out / "figures"
    organ_cols = [gap_col(o) for o in C.ORGANS]
    has_gap = master[organ_cols].notna().all(axis=1)
    G = master[has_gap].copy()

    # ---- sample composition ----
    sub0 = [c for c in master.columns if "Thalamus-Proper (left" in c][0]
    comp = {
        "N_total_age_gap": int(has_gap.sum()),
        "N_with_demographics": int((has_gap & master["age"].notna() & master["gender"].notna()).sum()),
        "N_brain_global(eTIV)": int((has_gap & master["eTIV"].notna()).sum()),
        "N_brain_global(BrainSegVol)": int((has_gap & master["BrainSegVol"].notna()).sum()),
        "N_brain_subcortical": int((has_gap & master[sub0].notna()).sum()),
        "N_disease": int((has_gap & master[dcols[0]].notna()).sum()),
    }
    for name in C.COGNITION:
        comp[f"N_cog_{name}"] = int((has_gap & master[name].notna()).sum())
    pd.DataFrame([{"metric": k, "N": v} for k, v in comp.items()]).to_csv(
        T / "T0.1_sample_composition.csv", index=False)

    # ---- age / sex ----
    demo = G[G["age"].notna()]
    age = pd.to_numeric(demo["age"], errors="coerce").dropna()
    sex = pd.to_numeric(demo["gender"], errors="coerce").dropna()
    rows = [
        {"var": "age", "mean": age.mean(), "sd": age.std(), "min": age.min(),
         "max": age.max(), "median": age.median(),
         "IQR": age.quantile(.75) - age.quantile(.25), "n": len(age)},
        {"var": "sex_male(1)", "n_male": int((sex == 1).sum()),
         "n_female": int((sex == 0).sum()),
         "pct_male": 100 * (sex == 1).mean(), "n": len(sex)},
    ]
    pd.DataFrame(rows).to_csv(T / "T0.2_age_sex.csv", index=False)

    # ---- organ age gap descriptives ----
    rows = []
    for oc in organ_cols:
        s = pd.to_numeric(G[oc], errors="coerce").dropna()
        rows.append({"organ": oc, "mean": s.mean(), "sd": s.std(), "min": s.min(),
                     "max": s.max(), "median": s.median(),
                     "IQR": s.quantile(.75) - s.quantile(.25),
                     "skew": stats.skew(s), "kurtosis": stats.kurtosis(s), "n": len(s)})
    pd.DataFrame(rows).to_csv(T / "T0.3_organ_gap_descriptives.csv", index=False)

    # ---- brain metric descriptives by class ----
    rows = []
    for _, m in reg.iterrows():
        col = m["metric"]
        if col not in G.columns:
            continue
        s = pd.to_numeric(G[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({"metric": col, "class": m["klass"], "type": m["type"],
                     "mean": s.mean(), "sd": s.std(), "min": s.min(), "max": s.max(),
                     "n": len(s)})
    pd.DataFrame(rows).to_csv(T / "T0.4_brain_descriptives.csv", index=False)

    # ---- cognition descriptives ----
    rows = []
    for name, spec in C.COGNITION.items():
        s = pd.to_numeric(G[name], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({"cognition": name, "field": spec["col"], "log": spec["log"],
                     "mean": s.mean(), "sd": s.std(), "min": s.min(), "max": s.max(),
                     "skew": stats.skew(s), "n": len(s)})
    pd.DataFrame(rows).to_csv(T / "T0.5_cognition_descriptives.csv", index=False)

    # ---- disease counts ----
    dc = io.disease_counts(G, dcols)
    dc["included_primary(>=%d)" % C.DISEASE_MIN_CASES] = dc["n_case"] >= C.DISEASE_MIN_CASES
    dc["cohens_d_only(>=%d)" % C.DISEASE_D_MIN_CASES] = (
        (dc["n_case"] >= C.DISEASE_D_MIN_CASES) & (dc["n_case"] < C.DISEASE_MIN_CASES))
    dc.to_csv(T / "T0.6_disease_counts.csv", index=False)

    # ---- missingness ----
    miss_cols = organ_cols + ["age", "gender", "BrainSegVol", "eTIV"] + list(C.COGNITION)
    miss = pd.DataFrame({"variable": miss_cols,
                         "missing_frac": [G[c].isna().mean() if c in G else np.nan
                                          for c in miss_cols]})
    miss.to_csv(T / "T0.7_missingness.csv", index=False)

    # ---- winsorization log ----
    wlog.to_csv(T / "T0.8_winsorization_log.csv", index=False)

    # ---- figures ----
    viz.violin(G, organ_cols, "Organ age-gap distributions (winsorized ±3SD)",
               F / "F0.1_organ_gap_distributions.png")
    for klass, cols in reg.groupby("klass"):
        mets = [c for c in cols["metric"].tolist() if c in G.columns][:24]
        if mets:
            viz.violin(G, mets, f"Brain metric distributions (class {klass}, first {len(mets)})",
                       F / f"F0.2_brain_dist_class{klass}.png")
    viz.missingness(G, miss_cols, "Missingness by variable (organ-gap cohort)",
                    F / "F0.3_missingness.png")
    return comp
