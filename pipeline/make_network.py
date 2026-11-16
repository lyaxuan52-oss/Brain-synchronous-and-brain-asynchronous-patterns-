# -*- coding: utf-8 -*-
"""Flagship figures:
Fig7 - data-driven organ co-pattern network (edges = shared patterns across the
        four modules: brain-structure system, disease category, cognitive domain,
        mediator region).  Node colour = integrated group; size = total signal.
Fig8 - tripartite organ -> brain pathway -> outcome schematic (the synthesis)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
import networkx as nx

plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.dpi"] = 300

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIV = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

GROUP = {
    "Kidney": 1, "Adrenal": 1, "Muscle": 1, "Salivary": 1,
    "Lung": 2, "Skin": 2, "Thyroid": 2,
    "Intestine": 3, "Stomach": 3, "Pancreas": 3, "Adipose": 3,
    "Liver": 3, "Heart": 3, "Artery": 3, "Esophagus": 3, "Immune": 3,
    "Brain": 0,
}
GC = {0: "#7f7f7f", 1: "#d62728", 2: "#1f77b4", 3: "#2ca02c", 4: "#9467bd"}
GLAB = {1: "(1) Brain-routed core (volume/subcortical pathway)",
        2: "(2) Cortical-surface \u2192 cognition group",
        3: "(3) Non-brain systemic (direct) group"}

# ---- brain metric -> system mapping (same as make_figures) ----
def brain_sys(metric, mtype):
    if mtype == "area":
        return "SurfaceArea"
    if mtype == "thickness":
        return "Thickness"
    if any(k in metric for k in ["Pallidum", "Putamen", "Caudate", "Accumbens",
                                 "Thalamus", "VentralDC"]):
        return "BasalGanglia"
    if any(k in metric for k in ["Hippocampus", "Amygdala", "entorhinal",
                                 "parahippocampal", "Brain-Stem"]):
        return "LimbicMemory"
    if mtype == "volume" and any(k in metric for k in
        ["Cortex", "BrainSeg", "SupraTentorial", "TotalGray", "CerebralWhiteMatter"]):
        return "GlobalVolume"
    return None


def build_features():
    # 1) brain systems (BH-sig peripheral -> brain)
    b = pd.read_csv(DELIV + r"\M1_brain\tables\T1.1_M1A_partial_corr_long.csv")
    b = b[b["role"].str.startswith("peripheral")].copy()
    b["sys"] = [brain_sys(m, t) for m, t in zip(b["metric"], b["type"])]
    b["sigflag"] = b["sig"].fillna("").str.contains("†")
    brain_feat = {}
    for o, sub in b[b["sigflag"]].groupby("organ"):
        brain_feat[o] = set("BRN:" + s for s in sub["sys"].dropna().unique())

    # 2) disease categories (mean|d| >= 0.15 in T2.7)
    dc = pd.read_csv(DELIV + r"\M2_disease\tables\T2.7_M2D_category_meanabs_d_matrix.csv").set_index("organ")
    dis_feat = {}
    for o in dc.index:
        cats = dc.columns[dc.loc[o].values >= 0.15]
        dis_feat[o] = set("DIS:" + c for c in cats)

    # 3) cognitive domains (|r| >= 0.045 in T3.2)
    cg = pd.read_csv(DELIV + r"\M3_cognition\tables\T3.2_M3A_rmatrix.csv").set_index("organ")
    cog_feat = {}
    for o in cg.index:
        hits = cg.columns[np.abs(cg.loc[o].values) >= 0.045]
        cog_feat[o] = set("COG:" + c for c in hits)

    # 4) mediator regions (regional only) in T4.3
    md = pd.read_csv(DELIV + r"\M4_mediation\tables\T4.3_M4_significant_mediations.csv")
    md = md[~md["mediator"].astype(str).str.contains("Holes|EulerNumber")]
    globalkw = ["BrainSegVol", "SupraTentorial", "TotalGray", "TotalSurfArea", "CortexVol",
                "CerebralWhiteMatter", "MeanThickness", "WhiteSurfArea", "eTIV", "MaskVol",
                "SubCortGray", "GlobalMean", "lhCortex", "rhCortex", "lhCerebral",
                "rhCerebral", "lhWhite", "rhWhite", "Ventricle", "Choroid"]
    reg = md[~md["mediator"].astype(str).str.contains("|".join(globalkw))]
    med_feat = {}
    for o, sub in reg.groupby("organ"):
        med_feat[o] = set("MED:" + str(m) for m in sub["mediator"].unique())

    organs = [o for o in GROUP if o != "Brain"]
    feats, sig_total = {}, {}
    for o in organs:
        f = set()
        f |= brain_feat.get(o, set())
        f |= dis_feat.get(o, set())
        f |= cog_feat.get(o, set())
        f |= med_feat.get(o, set())
        feats[o] = f
        # signal size = count of significant associations across modules
        sig_total[o] = (len(brain_feat.get(o, set())) * 1
                        + len(dis_feat.get(o, set()))
                        + len(cog_feat.get(o, set()))
                        + len(med_feat.get(o, set())))
    return organs, feats, sig_total


def fig7():
    organs, feats, sig = build_features()
    G = nx.Graph()
    for o in organs:
        G.add_node(o, group=GROUP[o], size=sig[o])
    for i in range(len(organs)):
        for j in range(i + 1, len(organs)):
            a, b = organs[i], organs[j]
            shared = feats[a] & feats[b]
            w = len(shared)
            if w >= 2:
                G.add_edge(a, b, weight=w)

    EDGE_MIN = 3  # only draw the stronger co-pattern links for clarity

    # ---- grouped layout: each integrated group on a small circle around an anchor ----
    anchors = {1: (-2.6, 1.5), 2: (2.6, 1.5), 3: (0.0, -1.9)}
    members = {g: [o for o in organs if GROUP[o] == g] for g in [1, 2, 3]}
    pos = {}
    for g, cen in anchors.items():
        ms = members[g]
        m = len(ms)
        if m == 1:
            pos[ms[0]] = cen
        else:
            rad = 0.55 + 0.12 * m
            for idx, o in enumerate(ms):
                ang = 2 * np.pi * idx / m + (0.3 if g == 3 else 0)
                pos[o] = (cen[0] + rad * np.cos(ang), cen[1] + rad * np.sin(ang))

    fig, ax = plt.subplots(figsize=(12, 10.5))
    # edges: within-group grey, cross-group coloured to highlight bridges
    for u, v in G.edges():
        w = G[u][v]["weight"]
        if w < EDGE_MIN:
            continue
        cross = GROUP[u] != GROUP[v]
        col = "#e08214" if cross else "#7f7f7f"
        alpha = 0.6 if cross else 0.7
        x1, y1 = pos[u]; x2, y2 = pos[v]
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                     connectionstyle="arc3,rad=0.1",
                     lw=0.6 + 0.8 * (w - EDGE_MIN + 1), color=col, alpha=alpha, zorder=1))
    # nodes
    for o in G.nodes():
        x, y = pos[o]
        s = 230 + 130 * G.nodes[o]["size"]
        ax.scatter([x], [y], s=s, color=GC[G.nodes[o]["group"]],
                   edgecolors="white", linewidths=1.6, alpha=0.95, zorder=3)
        ax.text(x, y, o, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=4)
    ax.set_xlim(-4.4, 4.8); ax.set_ylim(-3.6, 3.0)
    ax.set_axis_off()
    handles = [Patch(facecolor=GC[g], label=GLAB[g]) for g in [1, 2, 3]]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True,
              title="Integrated groups (cross-module synthesis)")
    ax.set_title("Multi-organ synergistic aging network\n"
                 "nodes = organs (size = total cross-module signal); "
                 "edges = shared patterns across brain / disease / cognition / mediation "
                 "(\u22653 shared features; orange = cross-group bridge)",
                 fontsize=12, fontweight="bold", loc="left")
    fig.savefig(os.path.join(FIG, "Fig7_multiorgan_network.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved Fig7_multiorgan_network.png")


def fig8():
    fig, ax = plt.subplots(figsize=(13.5, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_axis_off()

    def box(x, y, w, h, text, fc, fs=9, tc="black", lw=1.2, ec="white"):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                     edgecolor=ec, lw=lw, zorder=3, alpha=0.95,
                     joinstyle="round"))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
                zorder=4, fontweight="bold")

    def arrow(x1, y1, x2, y2, color, lw=1.8, style="-", rad=0.0, alpha=0.9):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                     mutation_scale=14, lw=lw, color=color, ls=style,
                     alpha=alpha, zorder=2))

    # column x positions
    xo, xb, xy = 1.4, 5.0, 8.7
    ax.text(xo, 9.7, "ORGAN AGING\n(plasma age gap)", ha="center", fontsize=11, fontweight="bold")
    ax.text(xb, 9.7, "BRAIN STRUCTURE\n(common intermediate)", ha="center", fontsize=11, fontweight="bold")
    ax.text(xy, 9.7, "DOWNSTREAM OUTCOME", ha="center", fontsize=11, fontweight="bold")

    # ----- brain pathway nodes -----
    bp = {
        "GlobalVol": (xb, 7.3, "Global volume /\nwhite matter"),
        "BasalG": (xb, 5.8, "Basal ganglia\n(pallidum, putamen)"),
        "Limbic": (xb, 4.4, "Limbic-memory\n(hippocampus)"),
        "Surface": (xb, 2.6, "Cortical\nsurface area"),
    }
    for k, (x, y, t) in bp.items():
        box(x, y, 1.9, 0.95, t, "#f0e6f5" if k != "Surface" else "#e3eef7",
            fs=8.5, ec="#888")

    # ----- outcome nodes -----
    outc = {
        "T2D": (xy, 7.5, "Type 2 diabetes"),
        "HTN": (xy, 6.2, "Hypertension"),
        "CHD": (xy, 4.9, "Coronary heart dis. /\nanaemia"),
        "COG": (xy, 3.0, "Cognition\n(memory, fluid IQ)"),
    }
    for k, (x, y, t) in outc.items():
        c = "#ffd9b3" if k != "COG" else "#cfe8cf"
        box(x, y, 2.0, 0.9, t, c, fs=8.5, ec="#888")

    # ----- organ nodes by group -----
    g1 = ["Kidney", "Adrenal", "Muscle", "Salivary"]
    g2 = ["Lung", "Skin", "Thyroid"]
    g3 = ["Intestine", "Stomach", "Pancreas", "Adipose", "Liver", "Heart", "Artery", "Immune"]
    g4 = []
    ypos = {}
    y = 8.9
    for o in g1:
        box(xo, y, 1.5, 0.5, o, GC[1], fs=8.5, tc="white"); ypos[o] = y; y -= 0.62
    y -= 0.2
    for o in g2:
        box(xo, y, 1.5, 0.5, o, GC[2], fs=8.5, tc="white"); ypos[o] = y; y -= 0.62
    y -= 0.2
    for o in g4:
        box(xo, y, 1.5, 0.5, o, GC[4], fs=8.5, tc="white"); ypos[o] = y; y -= 0.62
    y -= 0.25
    for o in g3:
        box(xo, y, 1.5, 0.44, o, GC[3], fs=8, tc="white"); ypos[o] = y; y -= 0.5

    # ----- edges: group1 -> global volume + basal ganglia -> T2D/HTN/COG -----
    for o in g1:
        arrow(xo + 0.75, ypos[o], xb - 0.95, bp["GlobalVol"][1], GC[1], lw=1.6, rad=0.05)
    arrow(xo + 0.75, ypos["Muscle"], xb - 0.95, bp["BasalG"][1], GC[1], lw=1.4, rad=-0.1)
    arrow(xo + 0.75, ypos["Kidney"], xb - 0.95, bp["BasalG"][1], GC[1], lw=1.4, rad=-0.15)
    arrow(xo + 0.75, ypos["Salivary"], xb - 0.95, bp["Limbic"][1], GC[1], lw=1.4, rad=-0.2)
    # global volume -> outcomes
    arrow(xb + 0.95, bp["GlobalVol"][1], xy - 1.0, outc["T2D"][1], "#d62728", lw=1.8, rad=0.08)
    arrow(xb + 0.95, bp["GlobalVol"][1], xy - 1.0, outc["HTN"][1], "#d62728", lw=1.6, rad=0.05)
    arrow(xb + 0.95, bp["BasalG"][1], xy - 1.0, outc["COG"][1], "#d62728", lw=1.5, rad=-0.15)
    arrow(xb + 0.95, bp["Limbic"][1], xy - 1.0, outc["COG"][1], "#d62728", lw=1.5, rad=-0.1)

    # ----- edges: group2 -> surface area -> cognition (+some disease) -----
    for o in g2:
        arrow(xo + 0.75, ypos[o], xb - 0.95, bp["Surface"][1], GC[2], lw=1.6, rad=-0.05)
    arrow(xo + 0.75, ypos["Salivary"], xb - 0.95, bp["Surface"][1], GC[1], lw=1.2, rad=0.15)
    arrow(xb + 0.95, bp["Surface"][1], xy - 1.0, outc["COG"][1], "#1f77b4", lw=2.0, rad=0.05)
    arrow(xb + 0.95, bp["Surface"][1], xy - 1.0, outc["T2D"][1], "#1f77b4", lw=1.2, rad=0.2)

    # ----- group3 direct organ -> outcome (bypass brain, dashed) -----
    for o in g3:
        yo = ypos[o]
        tgt = outc["T2D"][1] if o in ["Intestine", "Pancreas", "Adipose", "Liver"] else (
              outc["HTN"][1] if o in ["Artery"] else outc["CHD"][1])
        arrow(xo + 0.75, yo, xy - 1.05, tgt, GC[3], lw=1.0, style=(0, (4, 3)),
              rad=0.32, alpha=0.55)

    ax.text(5.0, 0.6,
            "Solid = organ\u2192brain\u2192outcome (brain-mediated).  "
            "Dashed green = non-brain direct route (systemic disease, bypasses brain).",
            ha="center", fontsize=8.5, style="italic", color="#333")
    ax.set_title("Synthesis: two brain-mediated convergence routes + one non-brain direct route",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(os.path.join(FIG, "Fig8_tripartite_synthesis.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved Fig8_tripartite_synthesis.png")


if __name__ == "__main__":
    fig7()
    fig8()
    print("NETWORK DONE")
