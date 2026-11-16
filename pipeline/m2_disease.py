# -*- coding: utf-8 -*-
"""Module 2: organ age gap -> disease outcomes (Cohen's d + Logistic OR),
Neuro-Preference Index, pathway differentiation."""
import numpy as np
import pandas as pd
import config as C
from config import gap_col, ORGANS
import stats_utils as S
import data_io as io
import viz


def run(master, dcols):
    out = C.MODULE_DIRS["M2"]
    T, F = out / "tables", out / "figures"
    organ_cols = [gap_col(o) for o in ORGANS]
    G = master[master[organ_cols].notna().all(axis=1)
               & master["age"].notna() & master["gender"].notna()].copy()

    inc = io.included_diseases(G, dcols, C.DISEASE_MIN_CASES)
    dcounts = io.disease_counts(G, dcols).set_index("disease")

    # ------------------------------------------------------------------ M2-A
    rows = []
    for disease in dcols:
        ncase = int(dcounts.loc[disease, "n_case"])
        neuro = C.is_neuro_disease(disease)
        # keep all systemic diseases >=30 cases; keep ALL neuro diseases >=5 cases
        # (so parkinsonism/dementia enter the neuro-preference index)
        min_c = C.NEURO_NPI_MIN_CASES if neuro else C.DISEASE_D_MIN_CASES
        if ncase < min_c:
            continue
        is_primary = ncase >= C.DISEASE_MIN_CASES
        dv = pd.to_numeric(G[disease], errors="coerce")
        for organ in ORGANS:
            x = gap_col(organ)
            xv = pd.to_numeric(G[x], errors="coerce")
            d = S.cohens_d(xv[dv == 1], xv[dv == 0])
            rec = {"organ": organ, "disease": disease, "is_neuro": neuro,
                   "n_case": ncase, "primary": is_primary, "cohens_d": d,
                   "small_sample": ncase < C.DISEASE_SMALL_FLAG}
            if is_primary:
                orr = S.logistic_or(G, disease, x)
                if orr:
                    rec.update({"OR": orr["OR"], "OR_ci_lo": orr["ci_lo"],
                                "OR_ci_hi": orr["ci_hi"], "logOR": orr["logOR"],
                                "p": orr["p"], "n": orr["n"]})
            rows.append(rec)
    m2a = pd.DataFrame(rows)
    # FDR on logistic p (primary diseases only)
    prim = m2a[m2a["primary"] & m2a["p"].notna()].copy()
    prim = S.add_fdr(prim, "p", group="organ")
    prim["sig"] = prim.apply(S.sig_marks, axis=1)
    m2a = m2a.merge(prim[["organ", "disease", "p_BH", "p_BY", "sig"]],
                    on=["organ", "disease"], how="left")
    m2a.to_csv(T / "T2.1_M2A_effect_matrix.csv", index=False)

    # d heatmap (primary diseases)
    dprim = m2a[m2a["primary"]]
    dmat = dprim.pivot(index="organ", columns="disease", values="cohens_d")
    organ_order = [o for o in C.ORGAN_SYSTEM_ORDER if o in dmat.index]
    dmat = dmat.reindex(organ_order)
    smat = dprim.assign(mark=dprim.apply(
        lambda r: S.sig_marks(r), axis=1)).pivot(index="organ", columns="disease",
                                                 values="mark").reindex(organ_order)
    dmat.to_csv(T / "T2.2_M2A_cohensd_matrix.csv")
    viz.heatmap(dmat.fillna(0), smat.fillna(""),
                "M2-A Cohen's d: organ × disease", F / "F2.1_M2A_cohensd_heatmap.png",
                cbar_label="Cohen's d", xlabel="disease", n=len(G))

    # OR forest for top pairs
    orf = m2a[m2a["OR"].notna()].assign(absl=lambda x: x["logOR"].abs()) \
        .sort_values("absl", ascending=False).head(25)
    if not orf.empty:
        orf = orf.assign(label=orf["organ"] + " → " + orf["disease"])
        viz.forest(orf, "OR", "OR_ci_lo", "OR_ci_hi", "label",
                   "M2-A top organ→disease odds ratios (per 1 SD age gap)",
                   F / "F2.2_M2A_OR_forest.png", logx=True)

    # ------------------------------------------------------------------ M2-B  NPI
    # NPI = specificity, not raw strength: (|d|_neuro - |d|_sys)/(|d|_neuro + |d|_sys).
    # The ranking is NOT robust to tiny neuro diseases (dementia n=8), so the PRIMARY
    # NPI uses only well-powered neuro diseases (>=50); two sensitivity versions add
    # the small ones. We also report neuro strength and system strength SEPARATELY so
    # a "non-specific" organ (high on both -> NPI~0) is distinguishable from a weak one.
    def _npi_for(neuro_min):
        recs = []
        for organ in ORGANS:
            sub = m2a[(m2a["organ"] == organ) & m2a["cohens_d"].notna()]
            nd = sub[sub["is_neuro"] & (sub["n_case"] >= neuro_min)]["cohens_d"].abs()
            sd = sub[~sub["is_neuro"]]["cohens_d"].abs()
            recs.append({"organ": organ, "neuro_strength_d": nd.mean(),
                         "system_strength_d": sd.mean(), "NPI": _npi(nd.mean(), sd.mean()),
                         "n_neuro_dis": int(len(nd)), "n_system_dis": int(len(sd))})
        return pd.DataFrame(recs)

    npi_sets = {name: _npi_for(mn) for name, mn in C.NPI_NEURO_SETS.items()}
    npidf = npi_sets["primary_ge50"].copy()
    # OR-based NPI (primary diseases only, all have >=50)
    oro = []
    for organ in ORGANS:
        subor = m2a[(m2a["organ"] == organ) & m2a["logOR"].notna()]
        ndo = subor[subor["is_neuro"]]["logOR"].abs()
        sdo = subor[~subor["is_neuro"]]["logOR"].abs()
        oro.append({"organ": organ, "neuro_strength_logOR": ndo.mean(),
                    "system_strength_logOR": sdo.mean(), "NPI_OR": _npi(ndo.mean(), sdo.mean())})
    npidf = npidf.merge(pd.DataFrame(oro), on="organ").sort_values("NPI", ascending=False)
    npidf.to_csv(T / "T2.3_M2B_NPI.csv", index=False)

    # sensitivity table: NPI under each neuro set + which neuro diseases each set holds
    sens = npi_sets["primary_ge50"][["organ"]].copy()
    for name, df in npi_sets.items():
        sens = sens.merge(df[["organ", "NPI"]].rename(columns={"NPI": f"NPI_{name}"}), on="organ")
    sens = sens.sort_values("NPI_primary_ge50", ascending=False)
    sens.to_csv(T / "T2.3b_M2B_NPI_sensitivity.csv", index=False)
    # rank stability across sets
    rc = sens[[c for c in sens.columns if c.startswith("NPI_")]].corr(method="spearman")
    rc.to_csv(T / "T2.3c_M2B_NPI_rank_correlation.csv")

    bar_df = npidf.dropna(subset=["NPI"]).copy()
    viz.bar(bar_df, "organ", "NPI", "M2-B Neuro-Preference Index (primary: neuro dis. ≥50)",
            F / "F2.3_M2B_NPI_bar.png", ylabel="NPI  (+neuro / −systemic)")
    # two-dimension view: neuro strength vs system strength
    viz.scatter(npidf["system_strength_d"], npidf["neuro_strength_d"], npidf["organ"],
                "M2-B neuro vs systemic strength (diagonal = non-specific)",
                F / "F2.5_M2B_strength_scatter.png",
                "systemic strength (mean |d|)", "neuro strength (mean |d|)")

    # ------------------------------------------------------------------ M2-C
    # Classify on specificity (NPI) AND absolute involvement. An organ with high
    # strength on both axes but NPI~0 is "non-specific (broadly involved)", which is
    # different from a weak organ near 0.
    med_neuro = npidf["neuro_strength_d"].median()
    med_sys = npidf["system_strength_d"].median()
    def classify(r):
        v = r["NPI"]
        if pd.isna(v):
            return "NA"
        hi = (r["neuro_strength_d"] >= med_neuro) or (r["system_strength_d"] >= med_sys)
        if v > 0.2:
            return "neuro-leaning"
        if v < -0.2:
            return "system-leaning"
        return "non-specific (broadly involved)" if hi else "mixed / weak"
    npidf["preference"] = npidf.apply(classify, axis=1)
    npidf[["organ", "NPI", "NPI_OR", "neuro_strength_d", "system_strength_d",
           "preference"]].to_csv(T / "T2.4_M2C_pathway_classification.csv", index=False)
    # per-disease top organ (by |d|)
    top_org = (m2a[m2a["primary"] & m2a["cohens_d"].notna()]
               .assign(absd=lambda x: x["cohens_d"].abs())
               .sort_values("absd", ascending=False)
               .groupby("disease").head(3))
    top_org.to_csv(T / "T2.5_M2C_top_organ_per_disease.csv", index=False)

    # ------------------------------------------------------------------ M2-D
    # Organ x disease-CATEGORY profile: the core question — how does each organ's
    # aging map onto DIFFERENT disease types? For every organ we summarise, per
    # physiological system, the mean |Cohen's d|, mean OR, and number of BH-sig
    # associations. Restricted to diseases with >=30 cases so each category cell is
    # reasonably powered (tiny neuro diseases dementia n=8 / parkinsonism n=23 excluded).
    catbase = m2a[m2a["cohens_d"].notna() & (m2a["n_case"] >= C.DISEASE_D_MIN_CASES)].copy()
    catbase["category"] = catbase["disease"].map(C.disease_category)
    prof = (catbase.assign(absd=catbase["cohens_d"].abs())
            .groupby(["organ", "category"])
            .agg(mean_abs_d=("absd", "mean"), max_abs_d=("absd", "max"),
                 n_disease=("disease", "nunique"),
                 mean_OR=("OR", "mean"),
                 n_sig_BH=("p_BH", lambda s: int((s < C.FDR_ALPHA).sum())))
            .reset_index())
    prof.to_csv(T / "T2.6_M2D_organ_disease_category_profile.csv", index=False)
    # organ x category matrix of mean |d| (the profile heatmap)
    cat_order = [c for c, _ in C.DISEASE_CATEGORY_RULES] + ["Other/Unclassified"]
    cat_order = [c for c in cat_order if c in prof["category"].unique()]
    organ_order = [o for o in C.ORGAN_SYSTEM_ORDER if o in prof["organ"].unique()]
    dmat = (prof.pivot(index="organ", columns="category", values="mean_abs_d")
            .reindex(index=organ_order, columns=cat_order))
    dmat.to_csv(T / "T2.7_M2D_category_meanabs_d_matrix.csv")
    viz.heatmap(dmat.fillna(0), dmat.notna().replace({True: "", False: ""}),
                "M2-D organ × disease-type profile (mean |Cohen's d|)",
                F / "F2.4_M2D_organ_category_heatmap.png",
                cbar_label="mean |d|", xlabel="disease category", n=len(G))
    # dominant disease type per organ (highest mean |d|)
    dom = (prof.sort_values("mean_abs_d", ascending=False)
           .groupby("organ").head(1)
           .rename(columns={"category": "dominant_category",
                            "mean_abs_d": "dominant_mean_abs_d"})
           [["organ", "dominant_category", "dominant_mean_abs_d"]])
    dom = dom.set_index("organ").reindex(organ_order).reset_index()
    dom.to_csv(T / "T2.8_M2D_organ_dominant_disease_type.csv", index=False)

    # ------------------------------------------------------------------ M2-E
    # Cumulative extreme-aging burden -> disease. Mirrors the cognition dose-response
    # (M3-B): does carrying MORE extremely-aged organs raise disease risk? For each
    # primary disease we fit Logit(disease) ~ burden + age + gender (OR per +1 aged
    # organ); we also tabulate prevalence across burden groups (0/1/2/3/4+) and test
    # the total primary-disease COUNT against burden (ANOVA + linear/quadratic trend).
    Gb, bcol = io.add_burden(G, method="top20")
    grp = f"{bcol}_grp"
    Gb[grp] = io.merge_small_groups(Gb, grp, min_n=100)
    prim_dis = [d for d in inc]
    brows, prev_rows = [], []
    for disease in prim_dis:
        orr = S.logistic_or(Gb, disease, bcol, standardize_x=False)
        if orr:
            brows.append({"disease": disease, "category": C.disease_category(disease),
                          "OR_per_organ": orr["OR"], "OR_ci_lo": orr["ci_lo"],
                          "OR_ci_hi": orr["ci_hi"], "logOR": orr["logOR"],
                          "p": orr["p"], "n": orr["n"], "n_case": orr["n_case"]})
        dv = pd.to_numeric(Gb[disease], errors="coerce")
        for g, sub in Gb.assign(_d=dv).dropna(subset=["_d", grp]).groupby(grp):
            prev_rows.append({"disease": disease, "burden_group": int(g),
                              "n": len(sub), "n_case": int((sub["_d"] == 1).sum()),
                              "prevalence": float((sub["_d"] == 1).mean())})
    bdf = pd.DataFrame(brows)
    if not bdf.empty:
        bdf = S.add_fdr(bdf, "p")
        bdf["sig"] = bdf.apply(S.sig_marks, axis=1)
        bdf = bdf.sort_values("logOR", ascending=False)
    bdf.to_csv(T / "T2.9_M2E_burden_to_disease_OR.csv", index=False)
    pd.DataFrame(prev_rows).to_csv(T / "T2.10_M2E_prevalence_by_burden.csv", index=False)
    # total primary-disease count vs burden (cumulative multimorbidity)
    dcount = pd.to_numeric(Gb[prim_dis].apply(pd.to_numeric, errors="coerce")
                           .eq(1).sum(axis=1), errors="coerce")
    Gb["_disease_count"] = dcount
    res = S.dose_response(Gb, "_disease_count", grp, ["age", "gender"])
    if res is not None:
        pd.DataFrame([{"outcome": "primary_disease_count", "F": res["F"],
                       "p_anova": res["p_anova"], "cohens_f": res["cohens_f"],
                       "p_linear": res["p_linear"], "b_linear": res["b_linear"],
                       "p_quadratic": res["p_quadratic"], "n": res["n"]}]
                     ).to_csv(T / "T2.11_M2E_disease_count_dose_response.csv", index=False)
        viz.dose_curve(res["group_means"],
                       "M2-E cumulative aging burden → primary-disease count",
                       F / "F2.6_M2E_disease_count_dose.png",
                       ylabel="mean # primary diseases", n=res["n"])
    if not bdf.empty:
        bf = bdf.assign(absl=bdf["logOR"].abs()).sort_values("absl", ascending=False).head(20)
        bf = bf.assign(label=bf["disease"].str[:30])
        viz.forest(bf, "OR_per_organ", "OR_ci_lo", "OR_ci_hi", "label",
                   "M2-E burden→disease odds ratio (per +1 extremely-aged organ)",
                   F / "F2.5_M2E_burden_OR_forest.png", logx=True)

    return {"m2a": m2a, "npi": npidf, "npi_sens": sens, "npi_rankcorr": rc,
            "category_profile": prof, "dominant": dom, "burden_disease": bdf,
            "included": inc}


def _npi(n, s):
    if pd.isna(n) or pd.isna(s) or (n + s) == 0:
        return np.nan
    return (n - s) / (n + s)
