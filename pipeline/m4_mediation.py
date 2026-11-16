# -*- coding: utf-8 -*-
"""Module 4: mediation  organ age gap -> brain structure -> (cognition | disease).

Prescreen (all three legs) then 200-iteration BCa bootstrap of the indirect
effect a*b.  Prescreen thresholds: X->M FDR-significant (M1-A), M->Y p<0.05,
X->Y total-effect p<0.10.
"""
import numpy as np
import pandas as pd
import config as C
from config import gap_col
import stats_utils as S
import viz


def _eTIV_needed(metric, reg):
    row = reg[reg["metric"] == metric]
    return (not row.empty) and bool(row.iloc[0]["control_eTIV"])


def run(master, reg, m1a, m2a, m3a, dcols):
    out = C.MODULE_DIRS["M4"]
    T, F = out / "tables", out / "figures"
    organ_cols = [gap_col(o) for o in C.ORGANS]
    base = master[master[organ_cols].notna().all(axis=1)
                  & master["age"].notna() & master["gender"].notna()].copy()

    xm = m1a[m1a["p_BH"] < C.FDR_ALPHA][["organ", "metric", "beta", "p_BH"]].copy()

    prescreen_rows, results = [], []

    # ---------------- cognition chains ----------------
    xy_cog = m3a[m3a["p"] < C.XY_TREND_P][["organ", "cognition", "p"]]
    my_cache = {}
    cand = []
    for _, xrow in xm.iterrows():
        organ, metric = xrow["organ"], xrow["metric"]
        for _, yrow in xy_cog[xy_cog["organ"] == organ].iterrows():
            cog = yrow["cognition"]
            key = (metric, cog)
            if key not in my_cache:
                cov = ["age", "gender"] + (["eTIV"] if _eTIV_needed(metric, reg) else [])
                r = S.ols_partial_corr(base, cog, metric, cov)
                my_cache[key] = r
            r = my_cache[key]
            if r is None or r["p"] >= C.MY_P:
                continue
            score = abs(xrow["beta"]) * abs(r["beta"])
            cand.append((score, organ, metric, cog))
            prescreen_rows.append({"chain": "cognition", "organ": organ,
                                   "mediator": metric, "outcome": cog,
                                   "xm_beta": xrow["beta"], "my_beta": r["beta"],
                                   "my_p": r["p"]})
    cand.sort(reverse=True)
    for score, organ, metric, cog in cand[:C.MED_MAX_COG]:
        cov = ["age", "gender"] + (["eTIV"] if _eTIV_needed(metric, reg) else [])
        res = S.bootstrap_mediation(base, gap_col(organ), metric, cog, cov,
                                    y_binary=False, n_boot=C.BOOTSTRAP_N,
                                    seed=C.RANDOM_SEED)
        if res:
            results.append({"chain": "cognition", "organ": organ, "mediator": metric,
                            "outcome": cog, **res})

    # ---------------- disease chains ----------------
    xy_dis = m2a[m2a["primary"] & (m2a["p"] < C.XY_TREND_P)][["organ", "disease", "p"]]
    my_cache = {}
    cand = []
    for _, xrow in xm.iterrows():
        organ, metric = xrow["organ"], xrow["metric"]
        for _, yrow in xy_dis[xy_dis["organ"] == organ].iterrows():
            dis = yrow["disease"]
            key = (metric, dis)
            if key not in my_cache:
                cov = ["age", "gender"] + (["eTIV"] if _eTIV_needed(metric, reg) else [])
                orr = S.logistic_or(base, dis, metric, covars=tuple(cov),
                                    standardize_x=True)
                my_cache[key] = orr
            orr = my_cache[key]
            if orr is None or orr["p"] >= C.MY_P:
                continue
            score = abs(xrow["beta"]) * abs(orr["logOR"])
            cand.append((score, organ, metric, dis))
            prescreen_rows.append({"chain": "disease", "organ": organ,
                                   "mediator": metric, "outcome": dis,
                                   "xm_beta": xrow["beta"], "my_beta": orr["logOR"],
                                   "my_p": orr["p"]})
    cand.sort(reverse=True)
    for score, organ, metric, dis in cand[:C.MED_MAX_DIS]:
        cov = ["age", "gender"] + (["eTIV"] if _eTIV_needed(metric, reg) else [])
        res = S.bootstrap_mediation(base, gap_col(organ), metric, dis, cov,
                                    y_binary=True, n_boot=C.BOOTSTRAP_N,
                                    seed=C.RANDOM_SEED)
        if res:
            results.append({"chain": "disease", "organ": organ, "mediator": metric,
                            "outcome": dis, **res})

    pd.DataFrame(prescreen_rows).to_csv(T / "T4.1_M4_prescreen.csv", index=False)
    resdf = pd.DataFrame(results)
    if not resdf.empty:
        resdf["classification"] = resdf.apply(_classify, axis=1)
        resdf = resdf.sort_values("ab", key=lambda s: s.abs(), ascending=False)
    resdf.to_csv(T / "T4.2_M4_mediation_results.csv", index=False)

    sig = resdf[resdf["sig"]] if not resdf.empty else resdf
    sig.to_csv(T / "T4.3_M4_significant_mediations.csv", index=False)

    # forest of indirect effects (significant)
    if not sig.empty:
        top = sig.head(25).assign(
            label=lambda d: d["organ"] + "→" + d["mediator"].str.slice(0, 18)
            + "→" + d["outcome"].astype(str).str.slice(0, 16))
        viz.forest(top, "ab", "ci_lo", "ci_hi", "label",
                   "M4 significant indirect effects (a*b, 95% BCa CI)",
                   F / "F4.1_M4_indirect_forest.png", xline=0.0,
                   xlabel="indirect effect a*b", color_col="sig")
    return {"mediation": resdf, "n_sig": int(sig.shape[0]) if not sig.empty else 0}


def _classify(r):
    if not r["sig"]:
        return "no mediation"
    total = r["c_total"]
    if abs(r["cprime"]) < 1e-9:
        return "full mediation"
    if np.sign(r["ab"]) == np.sign(r["cprime"]) or (total != 0 and abs(r["ab"]) < abs(total)):
        return "partial mediation"
    return "inconsistent/suppression"
