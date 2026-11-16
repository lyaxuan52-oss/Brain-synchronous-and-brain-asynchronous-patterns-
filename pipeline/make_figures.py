# -*- coding: utf-8 -*-
"""Generate publication-quality figures for the results report.
Figures 1-6. Flagship integrated network is in make_network.py.
All figures use English labels (organ/metric names); DPI 300."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.font_manager as fm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

ORGAN_ORDER = None  # set after clustering

# integrated-group colour scheme (used across figures for organ consistency)
GROUP = {
    # (1) brain-routed core (volume pathway)
    "Kidney": 1, "Adrenal": 1, "Muscle": 1, "Salivary": 1,
    # (2) cortical-surface -> cognition
    "Lung": 2, "Skin": 2, "Thyroid": 2,
    # (3) non-brain systemic
    "Intestine": 3, "Stomach": 3, "Pancreas": 3, "Adipose": 3,
    "Liver": 3, "Heart": 3, "Artery": 3, "Esophagus": 3, "Immune": 3,
    "Brain": 0,
}
GROUP_COLOR = {0: "#7f7f7f", 1: "#d62728", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd"}
GROUP_LABEL = {
    1: "(1) Brain-routed core (volume pathway)",
    2: "(2) Cortical-surface \u2192 cognition",
    3: "(3) Non-brain systemic",
    0: "Brain (self / reference)",
}


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


# =====================================================================
# FIG 1a  organ age-gap distributions (z-scored)
# =====================================================================
def fig1a():
    z = pd.read_csv(BASE + r"\data1\merged_age_gap_z.csv")
    cols = [c for c in z.columns if c.endswith("_age_gap_z")
            and not c.startswith(("Female", "Male"))]
    organs = [c.replace("_age_gap_z", "") for c in cols]
    data = [z[c].clip(-4, 4).dropna().values for c in cols]
    # order by dispersion of raw gap (use SD of z is 1; order alphabetically w/ group)
    order = sorted(range(len(organs)), key=lambda i: (GROUP.get(organs[i], 9), organs[i]))
    organs = [organs[i] for i in order]
    data = [data[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    parts = ax.violinplot(data, vert=False, showmeans=False, showmedians=True,
                          widths=0.9)
    for i, pc in enumerate(parts["bodies"]):
        g = GROUP.get(organs[i], 0)
        pc.set_facecolor(GROUP_COLOR[g]); pc.set_alpha(0.65); pc.set_edgecolor("white")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#333333"); parts[key].set_linewidth(0.9)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(range(1, len(organs) + 1)); ax.set_yticklabels(organs)
    ax.set_xlabel("Organ age gap (z-scored)")
    ax.set_title("A  Distribution of plasma-proteomic organ age gaps\n(N = 44,810, 17 organs)",
                 loc="left", fontsize=11, fontweight="bold")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=GROUP_COLOR[g],
               label=GROUP_LABEL[g], markersize=9) for g in [1, 2, 3]]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=True)
    savefig(fig, "Fig1a_agegap_distributions.png")


# =====================================================================
# FIG 1b  inter-organ age-gap correlation clustermap
# =====================================================================
def fig1b():
    R = pd.read_csv(BASE + r"\code\_p1_corr_matrix.csv", index_col=0)
    organs = list(R.columns)
    D = 1 - R.values
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    dn = dendrogram(Z, no_plot=True, labels=organs)
    order = dn["leaves"]
    Ro = R.iloc[order, order]
    labels = [organs[i] for i in order]

    fig = plt.figure(figsize=(9.2, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.22, 1], height_ratios=[0.16, 1],
                          wspace=0.02, hspace=0.02)
    # top dendrogram
    axd = fig.add_subplot(gs[0, 1])
    dendrogram(Z, ax=axd, color_threshold=0, above_threshold_color="#555555",
               no_labels=True)
    axd.set_xticks([]); axd.set_yticks([])
    for s in axd.spines.values():
        s.set_visible(False)
    # heatmap
    axh = fig.add_subplot(gs[1, 1])
    vmax = 0.35
    im = axh.imshow(Ro.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axh.set_xticks(range(len(labels))); axh.set_xticklabels(labels, rotation=90, fontsize=8)
    axh.set_yticks(range(len(labels))); axh.set_yticklabels(labels, fontsize=8)
    # colour organ tick labels by group
    for t in axh.get_xticklabels():
        t.set_color(GROUP_COLOR[GROUP.get(t.get_text(), 0)])
    for t in axh.get_yticklabels():
        t.set_color(GROUP_COLOR[GROUP.get(t.get_text(), 0)])
    # annotate strong |r|>=0.2
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = Ro.values[i, j]
            if i != j and abs(v) >= 0.20:
                axh.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=6, color="black")
    cbar = fig.colorbar(im, ax=axh, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson r (age gap, Winsorized \u00b13SD)")
    iu = np.triu_indices(len(organs), 1)
    meanr = R.values[iu].mean()
    fig.suptitle(f"Inter-organ age-gap correlation & hierarchical clustering\n"
                 f"mean pairwise r = {meanr:.2f} (93% of pairs positive; cf. Oh 2025 r\u22480.21)",
                 x=0.13, ha="left", fontsize=12, fontweight="bold", y=1.04)
    savefig(fig, "Fig1b_interorgan_corr_cluster.png")


# =====================================================================
# FIG 2  cumulative burden dose-response
# =====================================================================
def fig2():
    orr = pd.read_csv(DELIV + r"\M2_disease\tables\T2.9_M2E_burden_to_disease_OR.csv")
    prev = pd.read_csv(DELIV + r"\M2_disease\tables\T2.10_M2E_prevalence_by_burden.csv")
    cog = pd.read_csv(DELIV + r"\M3_cognition\tables\T3.4_M3B_dose_response.csv")
    cog = cog[cog["threshold"] == "top20"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.subplots_adjust(hspace=0.42, wspace=0.28)

    # (A) OR per +1 aged organ, forest
    ax = axes[0, 0]
    d = orr.sort_values("OR_per_organ", ascending=True)
    d = d[d["p"] < 0.05]
    y = np.arange(len(d))
    ax.errorbar(d["OR_per_organ"], y,
                xerr=[d["OR_per_organ"] - d["OR_ci_lo"], d["OR_ci_hi"] - d["OR_per_organ"]],
                fmt="o", color="#c0392b", ecolor="#c0392b", capsize=3, ms=6, lw=1.3)
    ax.axvline(1, color="grey", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(d["disease"], fontsize=8.5)
    ax.set_xlabel("Odds ratio per +1 extremely-aged organ (95% CI)")
    ax.set_title("A  Cumulative aging burden \u2192 disease risk\n(N = 7,383; 13/17 BH-FDR sig.)",
                 loc="left", fontsize=11, fontweight="bold")

    # (B) prevalence dose-response curves for key diseases
    ax = axes[0, 1]
    key = ["type 2 diabetes", "hypertension", "coronary heart disease",
           "renal failure", "arthritis", "anaemias"]
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, dis in enumerate(key):
        sub = prev[prev["disease"] == dis].sort_values("burden_group")
        ax.plot(sub["burden_group"], sub["prevalence"] * 100, marker="o",
                label=dis, color=palette[i], lw=1.8)
    ax.set_xlabel("Cumulative extreme-aging burden (n organs)")
    ax.set_ylabel("Disease prevalence (%)")
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xticklabels(["0", "1", "2", "3", "4+"])
    ax.legend(fontsize=7.5, frameon=True)
    ax.set_title("B  Monotonic prevalence gradient by aging burden",
                 loc="left", fontsize=11, fontweight="bold")

    # (C) multimorbidity: expected #primary diseases (sum of prevalences)
    ax = axes[1, 0]
    mm = prev.groupby("burden_group")["prevalence"].sum()
    ncnt = prev.groupby("burden_group")["n"].first()
    ax.bar(mm.index, mm.values, color="#e67e22", alpha=0.85, width=0.65)
    for x, v, nn in zip(mm.index, mm.values, ncnt.values):
        ax.text(x, v + 0.02, f"{v:.2f}\n(n={nn})", ha="center", fontsize=7.5)
    ax.set_xlabel("Cumulative extreme-aging burden (n organs)")
    ax.set_ylabel("Expected no. of primary diseases\n(\u03a3 prevalence over 17 diseases)")
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xticklabels(["0", "1", "2", "3", "4+"])
    ax.set_title("C  Multimorbidity rises with aging burden\n(ANOVA F=29.4, linear p=8.5e-25)",
                 loc="left", fontsize=11, fontweight="bold")

    # (D) cognition dose-response linear trend effect sizes
    ax = axes[1, 1]
    cog = cog.copy()
    cog["signed"] = cog["cohens_f"] * np.where(cog["p_anova"] < 0.05, 1, 0.35)
    cog = cog.sort_values("cohens_f", ascending=True)
    colors = ["#2c7fb8" if p < 0.05 else "#b8c6d1" for p in cog["p_anova"]]
    yy = np.arange(len(cog))
    ax.barh(yy, cog["cohens_f"], color=colors)
    for i, (f, p) in enumerate(zip(cog["cohens_f"], cog["p_anova"])):
        star = "*" if p < 0.05 else ""
        ax.text(f + 0.001, i, star, va="center", fontsize=11)
    ax.set_yticks(yy); ax.set_yticklabels(cog["cognition"], fontsize=8.5)
    ax.set_xlabel("Dose-response effect size (Cohen's f)")
    ax.set_title("D  Cumulative burden \u2192 worse cognition\n(6/7 tests ANOVA sig.; * p<.05)",
                 loc="left", fontsize=11, fontweight="bold")

    fig.suptitle("Cumulative multi-organ aging burden \u2192 systemic disease, multimorbidity and cognition",
                 fontsize=13, fontweight="bold", y=0.97)
    savefig(fig, "Fig2_dose_response.png")


# =====================================================================
# FIG 3  Module 1: organ -> brain structure profile heatmap
# =====================================================================
BRAIN_GROUPS = [
    ("Cortical surface area", lambda m, t: t == "area"),
    ("Cortical/global volume", lambda m, t: t == "volume" and any(
        k in m for k in ["Cortex", "BrainSeg", "SupraTentorial", "TotalGray",
                          "CerebralWhiteMatter", "Cerebellum", "CortexVol", "GrayVol"])),
    ("Cortical thickness", lambda m, t: t == "thickness"),
    ("Basal ganglia", lambda m, t: any(k in m for k in
        ["Pallidum", "Putamen", "Caudate", "Accumbens", "Thalamus", "VentralDC"])),
    ("Limbic-memory", lambda m, t: any(k in m for k in
        ["Hippocampus", "Amygdala", "entorhinal", "parahippocampal", "Brain-Stem"])),
    ("Ventricle/CSF", lambda m, t: any(k in m for k in
        ["Ventricle", "CSF", "Choroid"])),
]


def brain_group(metric, mtype):
    for name, fn in BRAIN_GROUPS:
        try:
            if fn(metric, mtype):
                return name
        except Exception:
            pass
    return None


def fig3():
    d = pd.read_csv(DELIV + r"\M1_brain\tables\T1.1_M1A_partial_corr_long.csv")
    d = d[d["role"].str.startswith("peripheral")].copy()
    d["bg"] = [brain_group(m, t) for m, t in zip(d["metric"], d["type"])]
    d = d.dropna(subset=["bg"])
    d["sigflag"] = d["sig"].fillna("").str.contains("†")
    grp_names = [g[0] for g in BRAIN_GROUPS]
    organs = [o for o in d["organ"].unique()]
    # keep organs with >=1 BH-sig association
    sig_org = d[d["sigflag"]].groupby("organ").size()
    organs = [o for o in organs if o in sig_org.index]
    organs = sorted(organs, key=lambda o: (GROUP.get(o, 9), o))
    mat = pd.DataFrame(index=organs, columns=grp_names, dtype=float)
    cnt = pd.DataFrame(0, index=organs, columns=grp_names, dtype=int)
    for o in organs:
        for g in grp_names:
            sub = d[(d["organ"] == o) & (d["bg"] == g)]
            if len(sub):
                mat.loc[o, g] = sub["r"].mean()
                cnt.loc[o, g] = int(sub["sigflag"].sum())
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    vmax = np.nanmax(np.abs(mat.values))
    im = ax.imshow(mat.values.astype(float), cmap="RdBu_r",
                   norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(len(grp_names))); ax.set_xticklabels(grp_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(organs))); ax.set_yticklabels(organs, fontsize=9)
    for t in ax.get_yticklabels():
        t.set_color(GROUP_COLOR[GROUP.get(t.get_text(), 0)])
    for i, o in enumerate(organs):
        for j, g in enumerate(grp_names):
            c = cnt.loc[o, g]
            if c > 0:
                ax.text(j, i, f"{int(c)}", ha="center", va="center", fontsize=8,
                        fontweight="bold", color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean partial r (organ \u2192 brain metric)")
    ax.set_title("Module 1  Organ aging \u2192 brain structure\n"
                 "cell = mean partial r; number = # BH-FDR sig. metrics in that system (N\u22483,960)",
                 loc="left", fontsize=11, fontweight="bold")
    savefig(fig, "Fig3_module1_brain.png")


# =====================================================================
# FIG 4  Module 2: organ x disease-category profile
# =====================================================================
def fig4():
    m = pd.read_csv(DELIV + r"\M2_disease\tables\T2.7_M2D_category_meanabs_d_matrix.csv")
    m = m.set_index("organ")
    order = sorted(m.index, key=lambda o: (GROUP.get(o, 9), o))
    m = m.loc[order]
    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    im = ax.imshow(m.values, cmap="OrRd", aspect="auto",
                   vmin=0, vmax=np.percentile(m.values, 98))
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index, fontsize=9)
    for t in ax.get_yticklabels():
        t.set_color(GROUP_COLOR[GROUP.get(t.get_text(), 0)])
    # mark each organ's dominant category
    for i, o in enumerate(m.index):
        j = int(np.argmax(m.loc[o].values))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                     edgecolor="#111111", lw=2))
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean |Cohen's d| (case vs control)")
    ax.set_title("Module 2  Organ aging \u2192 disease-type profile\n"
                 "black box = each organ's dominant disease system (N\u22487,383)",
                 loc="left", fontsize=11, fontweight="bold")
    savefig(fig, "Fig4_module2_disease.png")


# =====================================================================
# FIG 5  Module 3: organ x cognitive domain
# =====================================================================
def fig5():
    m = pd.read_csv(DELIV + r"\M3_cognition\tables\T3.2_M3A_rmatrix.csv").set_index("organ")
    sig = pd.read_csv(DELIV + r"\M3_cognition\tables\T3.3_M3A_significant.csv")
    # orient so that "worse cognition" is positive: flip accuracy tests sign
    # timed tasks (ReactionTime, TrailMaking) higher = worse; memory/IQ higher = better
    worse = m.copy()
    for c in ["FluidIntelligence", "NumericMemory", "PairedAssociateLearning", "TowerRearranging"]:
        worse[c] = -worse[c]
    order = sorted(m.index, key=lambda o: (GROUP.get(o, 9), o))
    worse = worse.loc[order]
    fig, ax = plt.subplots(figsize=(8.5, 7.8))
    vmax = 0.06
    im = ax.imshow(worse.values, cmap="PuOr_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cols = list(worse.columns)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(worse.index))); ax.set_yticklabels(worse.index, fontsize=9)
    for t in ax.get_yticklabels():
        t.set_color(GROUP_COLOR[GROUP.get(t.get_text(), 0)])
    # mark BH-significant
    sigset = set(zip(sig["organ"], sig["cognition"])) if {"organ", "cognition"}.issubset(sig.columns) else set()
    for i, o in enumerate(worse.index):
        for j, c in enumerate(cols):
            if (o, c) in sigset or abs(worse.loc[o, c]) >= 0.045:
                ax.text(j, i, "*", ha="center", va="center", fontsize=11, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Partial r, oriented so + = worse cognition")
    ax.set_title("Module 3  Organ aging \u2192 cognition\n"
                 "* |r|\u22650.045 / BH-FDR sig. (N\u22483,650\u20137,350 per test)",
                 loc="left", fontsize=11, fontweight="bold")
    savefig(fig, "Fig5_module3_cognition.png")


# =====================================================================
# FIG 6  Module 4: mediation profile (organ x outcome-type + region hubs)
# =====================================================================
def fig6():
    d = pd.read_csv(DELIV + r"\M4_mediation\tables\T4.3_M4_significant_mediations.csv")
    # drop QC/topology mediators
    drop = ["SurfaceHoles", "EulerNumber", "Holes"]
    d = d[~d["mediator"].astype(str).str.contains("|".join(drop))]
    d = d[d["organ"] != "Brain"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2),
                             gridspec_kw={"width_ratios": [1, 1]})
    # (A) stacked bar per organ: cognition vs disease significant paths
    ax = axes[0]
    d["ytype"] = np.where(d["chain"] == "cognition", "cognition", "disease")
    ct = d.groupby(["organ", "ytype"]).size().unstack(fill_value=0)
    ct = ct.reindex(sorted(ct.index, key=lambda o: (GROUP.get(o, 9), o)))
    y = np.arange(len(ct))
    cogv = ct.get("cognition", pd.Series(0, index=ct.index))
    disv = ct.get("disease", pd.Series(0, index=ct.index))
    ax.barh(y, cogv, color="#2c7fb8", label="\u2192 cognition")
    ax.barh(y, disv, left=cogv, color="#c0392b", label="\u2192 disease")
    ax.set_yticks(y); ax.set_yticklabels(ct.index, fontsize=9)
    for t in ax.get_yticklabels():
        t.set_color(GROUP_COLOR[GROUP.get(t.get_text(), 0)])
    ax.set_xlabel("No. of significant mediation paths (BCa 95% CI)")
    ax.legend(fontsize=9)
    ax.set_title("A  Brain-mediated paths per organ\n(488 significant; split by outcome type)",
                 loc="left", fontsize=11, fontweight="bold")

    # (B) top mediator brain regions (hubs) — regional only
    ax = axes[1]
    globalkw = ["BrainSegVol", "SupraTentorial", "TotalGray", "TotalSurfArea",
                "CortexVol", "CerebralWhiteMatter", "MeanThickness", "WhiteSurfArea",
                "eTIV", "MaskVol", "SubCortGray", "GlobalMean", "BrainSegVolNotVent",
                "lhCortex", "rhCortex", "lhCerebral", "rhCerebral", "lhWhite", "rhWhite",
                "Ventricle", "Choroid", "CSF"]
    reg = d[~d["mediator"].astype(str).str.contains("|".join(globalkw))]
    # shorten mediator names
    def short(m):
        m = m.replace("Volume of ", "").replace(" (right hemisphere)", " (R)")
        m = m.replace(" (left hemisphere)", " (L)").replace(" (whole brain)", "")
        return m[:34]
    reg = reg.copy(); reg["mshort"] = reg["mediator"].map(short)
    top = reg.groupby("mshort").size().sort_values(ascending=True).tail(14)
    ax.barh(np.arange(len(top)), top.values, color="#6a51a3")
    ax.set_yticks(np.arange(len(top))); ax.set_yticklabels(top.index, fontsize=8.5)
    ax.set_xlabel("No. of significant regional mediation paths")
    ax.set_title("B  Regional brain hubs carrying multi-organ paths\n(shared convergence points)",
                 loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("Module 4  Brain structure mediates organ aging \u2192 cognition / disease",
                 fontsize=13, fontweight="bold", y=1.01)
    savefig(fig, "Fig6_module4_mediation.png")


if __name__ == "__main__":
    fig1a()
    fig1b()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("ALL DONE")
