# -*- coding: utf-8 -*-
"""Module 1: organ age gap -> brain structure.

M1-A single-organ partial correlation (+ global-control specific effects),
M1-B multi-organ simultaneous regression + VIF,
M1-C stratified summary, M1-D cumulative-burden dose-response.
"""
import numpy as np
import pandas as pd
import config as C
from config import gap_col, ORGANS
import stats_utils as S
import data_io as io
import viz


def _covars(metric_row):
    cov = ["age", "gender"]
    if metric_row["control_eTIV"]:
        cov = cov + ["eTIV"]
    return cov


def _global_control_var(metric_row):
    if metric_row["klass"] == "B":
        return "TotalGrayVol"
    if metric_row["klass"] == "C":
        return "CortexVol"
    return None


def run(master, reg):
    out = C.MODULE_DIRS["M1"]
    T, F = out / "tables", out / "figures"
    organ_cols = [gap_col(o) for o in ORGANS]
    G = master[master[organ_cols].notna().all(axis=1)].copy()

    # ------------------------------------------------------------------ M1-A
    rows, gc_rows = [], []
    for _, m in reg.iterrows():
        metric = m["metric"]
        if metric not in G.columns:
            continue
        cov = _covars(m)
        gcv = _global_control_var(m)
        for organ in ORGANS:
            x = gap_col(organ)
            res = S.ols_partial_corr(G, metric, x, cov)
            if res is None:
                continue
            rows.append({"organ": organ, "metric": metric, "class": m["klass"],
                         "type": m["type"], **{k: res[k] for k in
                         ("b", "beta", "r", "t", "p", "n")}})
            # global-control specific-effect model (B/C only)
            if gcv and gcv in G.columns and gcv != metric:
                res2 = S.ols_partial_corr(G, metric, x, cov + [gcv])
                if res2 is not None:
                    atten = (abs(res["beta"]) - abs(res2["beta"])) / abs(res["beta"]) \
                        if res["beta"] != 0 else np.nan
                    vif = S.vif_table(G, [x] + cov + [gcv])
                    maxvif = vif.loc[vif["predictor"] == x, "VIF"].max()
                    gc_rows.append({"organ": organ, "metric": metric,
                                    "class": m["klass"], "beta_base": res["beta"],
                                    "beta_globalctrl": res2["beta"],
                                    "p_globalctrl": res2["p"],
                                    "attenuation": atten, "max_VIF": maxvif,
                                    "collinearity_warn": maxvif > C.VIF_SEVERE,
                                    "n": res2["n"]})
    m1a = pd.DataFrame(rows)
    m1a = S.add_fdr(m1a, "p", group="organ")
    m1a["sig"] = m1a.apply(S.sig_marks, axis=1)
    # role: Brain's own age gap vs brain structure is a MODEL-VALIDITY check (the brain
    # clock is trained on brain data, so a strong link is expected and not biologically
    # 'special'); the PERIPHERAL organs' age gap vs brain structure is the genuinely
    # informative cross-organ (body->brain) signal. Keep the two separate throughout.
    m1a["role"] = np.where(m1a["organ"] == C.CENTRAL_ORGAN,
                           "brain_self (model validity)", "peripheral (body->brain)")
    m1a.to_csv(T / "T1.1_M1A_partial_corr_long.csv", index=False)
    pd.DataFrame(gc_rows).to_csv(T / "T1.2_M1A_global_control_specific_effects.csv",
                                 index=False)

    # significant summary tables — split by role (do NOT mix the two interpretations)
    sig = m1a[m1a["p_BH"] < C.FDR_ALPHA].copy()
    sig.sort_values("r", key=lambda s: s.abs(), ascending=False).to_csv(
        T / "T1.3_M1A_significant_pairs.csv", index=False)
    sig[sig["organ"] == C.CENTRAL_ORGAN].sort_values(
        "r", key=lambda s: s.abs(), ascending=False).to_csv(
        T / "T1.3a_M1A_brain_self_coupling.csv", index=False)
    sig[sig["organ"] != C.CENTRAL_ORGAN].sort_values(
        "r", key=lambda s: s.abs(), ascending=False).to_csv(
        T / "T1.3b_M1A_peripheral_to_brain.csv", index=False)

    # wide r matrices + heatmaps per class
    for klass in ["A", "B", "C"]:
        sub = m1a[m1a["class"] == klass]
        if sub.empty:
            continue
        rmat = sub.pivot(index="organ", columns="metric", values="r")
        organ_order = [o for o in C.ORGAN_SYSTEM_ORDER if o in rmat.index]
        rmat = rmat.reindex(organ_order)
        marks = sub.pivot(index="organ", columns="metric", values="sig").reindex(organ_order)
        rmat.to_csv(T / f"T1.4_M1A_rmatrix_class{klass}.csv")
        viz.heatmap(rmat.fillna(0), marks.fillna(""),
                    f"M1-A partial r: organ × brain metric (class {klass})",
                    F / f"F1.1_M1A_heatmap_class{klass}.png",
                    n=int(sub["n"].median()))

    # ------------------------------------------------------------------ M1-B
    mb_rows, vif_rows = [], []
    for _, m in reg.iterrows():
        metric = m["metric"]
        if metric not in G.columns:
            continue
        cov = _covars(m)
        preds = organ_cols + cov
        d = G[[metric] + preds].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < len(preds) + 10:
            continue
        # standardized regression
        dz = d.copy()
        for c in d.columns:
            if dz[c].nunique() > 2 and dz[c].std() > 0:
                dz[c] = (dz[c] - dz[c].mean()) / dz[c].std()
        import statsmodels.api as sm
        X = sm.add_constant(dz[preds])
        fit = sm.OLS(dz[metric].values, X.values).fit()
        vt = S.vif_table(d, organ_cols + cov)
        vt = vt[vt["predictor"].isin(organ_cols)]  # VIF reported among the age gaps only
        maxvif = vt["VIF"].max()
        worst = vt.loc[vt["VIF"].idxmax(), "predictor"] if not vt.empty else None
        vif_rows.append({"metric": metric, "class": m["klass"], "max_VIF": maxvif,
                         "worst_predictor": worst,
                         "warn": ("severe" if maxvif > C.VIF_SEVERE else
                                  "moderate" if maxvif > C.VIF_WARN else "ok"),
                         "n": len(d)})
        for i, organ in enumerate(ORGANS):
            mb_rows.append({"organ": organ, "metric": metric, "class": m["klass"],
                            "beta": fit.params[i + 1], "p": fit.pvalues[i + 1],
                            "is_central": organ == C.CENTRAL_ORGAN,
                            "max_VIF": maxvif, "n": len(d)})
    m1b = pd.DataFrame(mb_rows)
    m1b = S.add_fdr(m1b, "p", group="organ")
    m1b["sig"] = m1b.apply(S.sig_marks, axis=1)
    m1b.to_csv(T / "T1.5_M1B_multiorgan_betas.csv", index=False)
    vifdf = pd.DataFrame(vif_rows)
    vifdf.to_csv(T / "T1.6_M1B_VIF_diagnostics.csv", index=False)

    # brain vs peripheral |beta| comparison
    bcmp = (m1b.assign(absbeta=m1b["beta"].abs())
            .groupby("is_central")["absbeta"].agg(["mean", "median", "std", "count"]))
    bcmp.to_csv(T / "T1.7_M1B_brain_vs_peripheral.csv")

    # ------------------------------------------------------------------ M1-C
    layer_rows = []
    for (organ, klass), sub in m1a.groupby(["organ", "class"]):
        nsig = int((sub["p_BH"] < C.FDR_ALPHA).sum())
        pos = int((sub["r"] > 0).sum()); neg = int((sub["r"] < 0).sum())
        layer_rows.append({"organ": organ, "class": klass,
                           "mean_abs_r": sub["r"].abs().mean(),
                           "n_sig_BH": nsig, "n_pos": pos, "n_neg": neg,
                           "n_tests": len(sub)})
    m1c = pd.DataFrame(layer_rows)
    m1c.to_csv(T / "T1.8_M1C_stratified_summary.csv", index=False)
    layer_summary = m1c.groupby("class").agg(
        mean_abs_r=("mean_abs_r", "mean"), total_sig=("n_sig_BH", "sum")).reset_index()
    layer_summary.to_csv(T / "T1.9_M1C_layer_comparison.csv", index=False)
    viz.bar(layer_summary, "class", "mean_abs_r",
            "M1-C mean |r| by brain layer", F / "F1.2_M1C_layer_meanabs_r.png",
            xlabel="brain class", ylabel="mean |r|", rotate=0)

    # ------------------------------------------------------------------ M1-D
    dose_rows = []
    global_targets = ["TotalGrayVol", "CortexVol", "GlobalMeanThickness",
                      "BrainSegVol", "SubCortGrayVol", "CerebralWhiteMatterVol"]
    for method in ["top20", "z1", "top25"]:
        Gm, bcol = io.add_burden(G, method=method)
        grp_col = f"{bcol}_grp"
        Gm[grp_col] = io.merge_small_groups(Gm, grp_col, min_n=100)
        for metric in global_targets:
            if metric not in Gm.columns:
                continue
            mrow = reg[reg["metric"] == metric]
            cov = _covars(mrow.iloc[0]) if not mrow.empty else ["age", "gender"]
            res = S.dose_response(Gm, metric, grp_col, cov)
            if res is None:
                continue
            dose_rows.append({"threshold": method, "metric": metric, "F": res["F"],
                              "p_anova": res["p_anova"], "partial_eta2": res["partial_eta2"],
                              "cohens_f": res["cohens_f"], "p_linear": res["p_linear"],
                              "b_linear": res["b_linear"], "p_quadratic": res["p_quadratic"],
                              "n": res["n"]})
            if method == "top20":
                viz.dose_curve(res["group_means"],
                               f"M1-D dose-response: {metric}",
                               F / f"F1.3_M1D_dose_{metric}.png",
                               ylabel=metric, n=res["n"])
    pd.DataFrame(dose_rows).to_csv(T / "T1.10_M1D_dose_response.csv", index=False)

    return {"m1a": m1a, "m1b": m1b,
            "n_sig_m1a": int((m1a["p_BH"] < C.FDR_ALPHA).sum())}
