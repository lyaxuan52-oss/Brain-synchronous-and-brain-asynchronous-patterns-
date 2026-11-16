# -*- coding: utf-8 -*-
"""Module 5: sex stratification and interaction.

M5-A brain partial corr by sex, M5-B burden dose-response by sex,
M5-C sex x organ interaction (brain / cognition / disease),
M5-D disease OR by sex, M5-E cognition partial corr by sex.
"""
import numpy as np
import pandas as pd
import config as C
from config import gap_col, ORGANS
import stats_utils as S
import data_io as io
import viz

MALE, FEMALE = 1, 0


def _covars_brain(mrow):
    return ["age"] + (["eTIV"] if mrow["control_eTIV"] else [])


def run(master, reg, dcols):
    out = C.MODULE_DIRS["M5"]
    T, F = out / "tables", out / "figures"
    organ_cols = [gap_col(o) for o in ORGANS]
    G = master[master[organ_cols].notna().all(axis=1)
               & master["age"].notna() & master["gender"].notna()].copy()
    Gm = G[G["gender"] == MALE]; Gf = G[G["gender"] == FEMALE]

    # ---------------------------------------------------------------- M5-A
    rows = []
    for _, m in reg.iterrows():
        metric = m["metric"]
        if metric not in G.columns:
            continue
        cov = _covars_brain(m)
        for organ in ORGANS:
            x = gap_col(organ)
            rm = S.ols_partial_corr(Gm, metric, x, cov)
            rf = S.ols_partial_corr(Gf, metric, x, cov)
            rows.append({"organ": organ, "metric": metric, "class": m["klass"],
                         "r_male": rm["r"] if rm else np.nan,
                         "p_male": rm["p"] if rm else np.nan,
                         "n_male": rm["n"] if rm else 0,
                         "r_female": rf["r"] if rf else np.nan,
                         "p_female": rf["p"] if rf else np.nan,
                         "n_female": rf["n"] if rf else 0})
    m5a = pd.DataFrame(rows)
    m5a["r_diff"] = m5a["r_male"] - m5a["r_female"]
    m5a.to_csv(T / "T5.1_M5A_brain_by_sex.csv", index=False)

    # ---------------------------------------------------------------- M5-B
    dose_rows = []
    targets = ["TotalGrayVol", "CortexVol", "GlobalMeanThickness", "BrainSegVol"]
    for label, sub in [("male", Gm), ("female", Gf)]:
        subb, bcol = io.add_burden(sub, method="top20")
        grp = f"{bcol}_grp"
        subb[grp] = io.merge_small_groups(subb, grp, min_n=80)
        for metric in targets:
            if metric not in subb.columns:
                continue
            mrow = reg[reg["metric"] == metric]
            cov = _covars_brain(mrow.iloc[0]) if not mrow.empty else ["age"]
            res = S.dose_response(subb, metric, grp, cov)
            if res:
                dose_rows.append({"sex": label, "metric": metric, "F": res["F"],
                                  "p_anova": res["p_anova"], "cohens_f": res["cohens_f"],
                                  "p_linear": res["p_linear"], "n": res["n"]})
    pd.DataFrame(dose_rows).to_csv(T / "T5.2_M5B_dose_by_sex.csv", index=False)

    # ---------------------------------------------------------------- M5-C
    inter_rows = []
    # brain interactions
    for _, m in reg.iterrows():
        metric = m["metric"]
        if metric not in G.columns:
            continue
        cov = ["age"] + (["eTIV"] if m["control_eTIV"] else [])
        for organ in ORGANS:
            r = S.ols_interaction(G, metric, gap_col(organ), "gender", cov)
            if r:
                inter_rows.append({"domain": "brain", "organ": organ, "outcome": metric,
                                   "class": m["klass"], "beta_int": r["beta_int"],
                                   "p_int": r["p_int"], "n": r["n"]})
    # cognition interactions
    for cog in C.COGNITION:
        if cog not in G.columns:
            continue
        for organ in ORGANS:
            r = S.ols_interaction(G, cog, gap_col(organ), "gender", ["age"])
            if r:
                inter_rows.append({"domain": "cognition", "organ": organ, "outcome": cog,
                                   "class": "-", "beta_int": r["beta_int"],
                                   "p_int": r["p_int"], "n": r["n"]})
    # disease interactions
    inc = io.included_diseases(G, dcols, C.DISEASE_MIN_CASES)
    for dis in inc:
        for organ in ORGANS:
            r = S.logit_interaction(G, dis, gap_col(organ), "gender", ["age"])
            if r:
                inter_rows.append({"domain": "disease", "organ": organ, "outcome": dis,
                                   "class": "-", "beta_int": r["beta_int"],
                                   "p_int": r["p_int"], "n": r["n"]})
    m5c = pd.DataFrame(inter_rows)
    m5c = S.add_fdr(m5c, "p_int", group="domain", bh_col="p_int_BH", by_col="p_int_BY")
    m5c["sig"] = m5c.apply(lambda r: S.sig_marks(r, p="p_int", bh="p_int_BH", by="p_int_BY"), axis=1)
    m5c.to_csv(T / "T5.3_M5C_interactions.csv", index=False)
    m5c[m5c["p_int_BH"] < C.FDR_ALPHA].sort_values("p_int").to_csv(
        T / "T5.4_M5C_significant_interactions.csv", index=False)

    # ---------------------------------------------------------------- M5-D
    drows = []
    for dis in inc:
        for label, sub in [("male", Gm), ("female", Gf)]:
            for organ in ORGANS:
                orr = S.logistic_or(sub, dis, gap_col(organ), covars=("age",))
                if orr:
                    drows.append({"disease": dis, "sex": label, "organ": organ,
                                  "OR": orr["OR"], "ci_lo": orr["ci_lo"],
                                  "ci_hi": orr["ci_hi"], "p": orr["p"],
                                  "n_case": orr["n_case"], "n": orr["n"]})
    m5d = pd.DataFrame(drows)
    if not m5d.empty:
        m5d = S.add_fdr(m5d, "p", group="sex")
        m5d["sig"] = m5d.apply(S.sig_marks, axis=1)
    m5d.to_csv(T / "T5.5_M5D_disease_OR_by_sex.csv", index=False)

    # ---------------------------------------------------------------- M5-E
    crows = []
    for cog in C.COGNITION:
        if cog not in G.columns:
            continue
        for label, sub in [("male", Gm), ("female", Gf)]:
            for organ in ORGANS:
                r = S.ols_partial_corr(sub, cog, gap_col(organ), ["age"])
                if r:
                    crows.append({"cognition": cog, "sex": label, "organ": organ,
                                  "r": r["r"], "p": r["p"], "n": r["n"]})
    m5e = pd.DataFrame(crows)
    if not m5e.empty:
        m5e = S.add_fdr(m5e, "p", group="sex")
        m5e["sig"] = m5e.apply(S.sig_marks, axis=1)
    m5e.to_csv(T / "T5.6_M5E_cognition_by_sex.csv", index=False)

    # ---------------------------------------------------------------- figures
    sig_int = m5c[m5c["p_int"] < 0.05]
    if not sig_int.empty:
        topc = (sig_int.assign(absb=sig_int["beta_int"].abs())
                .sort_values("absb", ascending=False).head(20))
        viz.bar(topc.assign(lab=topc["organ"] + "→" + topc["outcome"].str.slice(0, 16)),
                "lab", "beta_int", "M5-C top sex×organ interactions (p<0.05)",
                F / "F5.1_M5C_interactions.png", ylabel="interaction β")
    # M5-A male vs female mean |r| per organ
    agg = (m5a.groupby("organ")[["r_male", "r_female"]]
           .apply(lambda d: pd.Series({"male": d["r_male"].abs().mean(),
                                       "female": d["r_female"].abs().mean()}))
           .reset_index())
    viz.grouped_compare(agg, "organ", "male", "female",
                        "M5-A mean |partial r| on brain by sex",
                        F / "F5.2_M5A_meanabs_r_by_sex.png", "mean |r|")
    return {"m5c": m5c, "n_sig_interactions": int((m5c["p_int_BH"] < C.FDR_ALPHA).sum())}
