# -*- coding: utf-8 -*-
"""Module 3: organ age gap -> cognition (partial correlation + dose-response).
Covariates: age + gender only (no eTIV / brain volume)."""
import numpy as np
import pandas as pd
import config as C
from config import gap_col, ORGANS
import stats_utils as S
import data_io as io
import viz


def run(master):
    out = C.MODULE_DIRS["M3"]
    T, F = out / "tables", out / "figures"
    organ_cols = [gap_col(o) for o in ORGANS]
    G = master[master[organ_cols].notna().all(axis=1)
               & master["age"].notna() & master["gender"].notna()].copy()
    cov = ["age", "gender"]

    # ------------------------------------------------------------------ M3-A
    rows = []
    for cog in C.COGNITION:
        if cog not in G.columns:
            continue
        for organ in ORGANS:
            res = S.ols_partial_corr(G, cog, gap_col(organ), cov)
            if res is None:
                continue
            rows.append({"organ": organ, "cognition": cog,
                         **{k: res[k] for k in ("b", "beta", "r", "t", "p", "n")}})
    m3a = pd.DataFrame(rows)
    m3a = S.add_fdr(m3a, "p", group="organ")
    m3a["sig"] = m3a.apply(S.sig_marks, axis=1)
    m3a.to_csv(T / "T3.1_M3A_partial_corr.csv", index=False)

    rmat = m3a.pivot(index="organ", columns="cognition", values="r")
    order = [o for o in C.ORGAN_SYSTEM_ORDER if o in rmat.index]
    rmat = rmat.reindex(order)
    marks = m3a.pivot(index="organ", columns="cognition", values="sig").reindex(order)
    rmat.to_csv(T / "T3.2_M3A_rmatrix.csv")
    viz.heatmap(rmat.fillna(0), marks.fillna(""),
                "M3-A partial r: organ × cognition", F / "F3.1_M3A_heatmap.png",
                cbar_label="partial r", xlabel="cognition", figsize=(9, 7))

    m3a[m3a["p_BH"] < C.FDR_ALPHA].sort_values(
        "r", key=lambda s: s.abs(), ascending=False).to_csv(
        T / "T3.3_M3A_significant.csv", index=False)

    # ------------------------------------------------------------------ M3-B
    dose_rows = []
    for method in ["top20", "z1"]:
        Gm, bcol = io.add_burden(G, method=method)
        grp = f"{bcol}_grp"
        Gm[grp] = io.merge_small_groups(Gm, grp, min_n=100)
        for cog in C.COGNITION:
            if cog not in Gm.columns:
                continue
            res = S.dose_response(Gm, cog, grp, cov)
            if res is None:
                continue
            dose_rows.append({"threshold": method, "cognition": cog, "F": res["F"],
                              "p_anova": res["p_anova"], "cohens_f": res["cohens_f"],
                              "p_linear": res["p_linear"], "b_linear": res["b_linear"],
                              "p_quadratic": res["p_quadratic"], "n": res["n"]})
            if method == "top20":
                viz.dose_curve(res["group_means"], f"M3-B dose-response: {cog}",
                               F / f"F3.2_M3B_dose_{cog}.png", ylabel=cog, n=res["n"])
    pd.DataFrame(dose_rows).to_csv(T / "T3.4_M3B_dose_response.csv", index=False)
    return {"m3a": m3a}
