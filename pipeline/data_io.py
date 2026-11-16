# -*- coding: utf-8 -*-
"""Data loading, brain-metric registry, winsorization and per-module samples.

The organ-aging cohort barely overlaps the imaging cohort, so every module gets
its own sample via listwise deletion on exactly the variables that module needs.
"""
import re
import numpy as np
import pandas as pd
import config as C
from config import gap_col


# --------------------------------------------------------------------------- #
# Brain-metric registry
# --------------------------------------------------------------------------- #
_GLOBAL_VOLUME = [
    "BrainSegVol", "BrainSegVolNotVent", "VentricleChoroidVol", "lhCortexVol",
    "rhCortexVol", "CortexVol", "lhCerebralWhiteMatterVol", "rhCerebralWhiteMatterVol",
    "CerebralWhiteMatterVol", "SubCortGrayVol", "TotalGrayVol", "SupraTentorialVol",
    "SupraTentorialVolNotVent", "MaskVol",
]
_GLOBAL_RATIO = ["BrainSegVol-to-eTIV", "MaskVol-to-eTIV"]
_GLOBAL_AREA = ["lhWhiteSurfArea", "rhWhiteSurfArea", "TotalSurfArea"]
_GLOBAL_THICK = ["lhMeanThickness", "rhMeanThickness", "GlobalMeanThickness"]
_GLOBAL_TOPO = ["lhSurfaceHoles", "rhSurfaceHoles", "SurfaceHoles",
                "lhEulerNumber", "rhEulerNumber", "TotalEulerNumber",
                "lhMeanFoldInd", "rhMeanFoldInd", "GlobalFoldInd",
                "lhMeanCurvInd", "rhMeanCurvInd", "GlobalCurvInd"]
_GLOBAL_EXCLUDE = ["eTIV", "lhNumVert", "rhNumVert"]  # covariate / QC only


def _clean(col):
    """Normalise messy brain-structure column names (spaces, newlines)."""
    return re.sub(r"\s+", " ", str(col).replace("\n", " ")).strip()


def load_all():
    """Load every source, clean columns, extract eid, return dict of frames."""
    gap = pd.read_csv(C.FILES["age_gap"])
    bg = pd.read_csv(C.FILES["brain_global"])
    bg["eid"] = bg["ID"].astype(str).str.split("_").str[0].astype(int)
    bs = pd.read_csv(C.FILES["brain_subcort"])
    bs.columns = [_clean(c) if c != "eid" else "eid" for c in bs.columns]
    bc = pd.read_csv(C.FILES["brain_cortex"])
    bc.columns = [_clean(c) if c != "eid" else "eid" for c in bc.columns]
    cv = pd.read_csv(C.FILES["clinical"], low_memory=False)
    cog = pd.read_csv(C.FILES["cognition"], low_memory=False)
    dis = pd.read_csv(C.FILES["disease"], low_memory=False)
    return {"gap": gap, "bg": bg, "bs": bs, "bc": bc, "cv": cv, "cog": cog, "dis": dis}


def build_brain_registry(frames):
    """Return DataFrame describing every brain outcome metric."""
    rows = []
    bg = frames["bg"]
    for col in bg.columns:
        if col in ("ID", "eid") or col in _GLOBAL_EXCLUDE:
            continue
        if col in _GLOBAL_VOLUME:
            typ, eTIV = "volume", True
        elif col in _GLOBAL_RATIO:
            typ, eTIV = "ratio", False
        elif col in _GLOBAL_AREA:
            typ, eTIV = "area", False
        elif col in _GLOBAL_THICK:
            typ, eTIV = "thickness", False
        elif col in _GLOBAL_TOPO:
            typ, eTIV = "topology", False
        else:
            typ, eTIV = "other", False
        rows.append({"metric": col, "clean": col, "source": "global",
                     "klass": "A", "type": typ, "control_eTIV": eTIV})
    for col in frames["bs"].columns:
        if col == "eid":
            continue
        rows.append({"metric": col, "clean": col, "source": "subcort",
                     "klass": "B", "type": "volume", "control_eTIV": True})
    for col in frames["bc"].columns:
        if col == "eid":
            continue
        rows.append({"metric": col, "clean": col, "source": "cortex",
                     "klass": "C", "type": "volume", "control_eTIV": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Master merge
# --------------------------------------------------------------------------- #
def build_master(frames):
    """Merge everything on eid (outer), coalesce age/sex, winsorize organ gaps.

    Returns (master_df, winsor_log_df).  Brain-metric columns are prefixed with
    their class so cortical/subcortical names never collide with global ones.
    """
    gap = frames["gap"].copy()
    organ_cols = [gap_col(o) for o in C.ORGANS]
    master = gap[["eid"] + organ_cols].copy()

    # demographics: coalesce clinical (31-0.0 sex, 21003-2.0 age) with disease table
    cv = frames["cv"][["eid", "31-0.0", "21003-2.0"]].rename(
        columns={"31-0.0": "sex_cv", "21003-2.0": "age_cv"})
    dis = frames["dis"]
    demo = cv.merge(dis[["eid", "gender", "age"]].rename(
        columns={"gender": "sex_dis", "age": "age_dis"}), on="eid", how="outer")
    demo["gender"] = demo["sex_cv"].combine_first(demo["sex_dis"])
    demo["age"] = demo["age_cv"].combine_first(demo["age_dis"])
    master = master.merge(demo[["eid", "age", "gender"]], on="eid", how="left")

    # brain metrics (prefix to avoid collisions)
    bg = frames["bg"].copy()
    keep_bg = [c for c in bg.columns if c not in ("ID",)]
    bg = bg[keep_bg]
    master = master.merge(bg, on="eid", how="left")
    master = master.merge(frames["bs"], on="eid", how="left")
    master = master.merge(frames["bc"], on="eid", how="left")

    # cognition (instance 2)
    cog = frames["cog"]
    cog_cols = ["eid"] + [spec["col"] for spec in C.COGNITION.values() if spec["col"] in cog.columns]
    master = master.merge(cog[cog_cols], on="eid", how="left")
    for name, spec in C.COGNITION.items():
        col = spec["col"]
        if col not in master.columns:
            master[name] = np.nan
            continue
        v = pd.to_numeric(master[col], errors="coerce")
        if spec["log"]:
            v = np.log(v.where(v > 0))
        master[name] = v

    # disease columns
    dcols = disease_columns(frames)
    master = master.merge(dis[["eid"] + dcols], on="eid", how="left")

    # winsorize organ age gaps (+-3SD), record truncation counts
    log = []
    for oc in organ_cols:
        s = pd.to_numeric(master[oc], errors="coerce")
        mu, sd = s.mean(), s.std()
        lo, hi = mu - C.WINSOR_SD * sd, mu + C.WINSOR_SD * sd
        n_lo = int((s < lo).sum()); n_hi = int((s > hi).sum())
        master[oc] = s.clip(lo, hi)
        log.append({"organ": oc, "mean": mu, "sd": sd, "low_cut": lo, "high_cut": hi,
                    "n_winsor_low": n_lo, "n_winsor_high": n_hi,
                    "n_winsor_total": n_lo + n_hi})
    return master, pd.DataFrame(log)


def disease_columns(frames):
    """All disease indicator columns present in the disease table."""
    dis = frames["dis"]
    start = dis.columns.get_loc("coronary heart disease")
    end = dis.columns.get_loc("lymphoma and blood cancer")
    return list(dis.columns[start:end + 1])


def disease_counts(master, dcols):
    rows = []
    for d in dcols:
        v = pd.to_numeric(master[d], errors="coerce")
        # only within organ-gap cohort (master already keyed on gap eids)
        valid = v.notna()
        ncase = int((v == 1).sum()); nctrl = int((v == 0).sum())
        rows.append({"disease": d, "n_case": ncase, "n_control": nctrl,
                     "n_valid": int(valid.sum()),
                     "prevalence_pct": 100 * ncase / valid.sum() if valid.sum() else np.nan,
                     "is_neuro": C.is_neuro_disease(d)})
    return pd.DataFrame(rows).sort_values("n_case", ascending=False)


def included_diseases(master, dcols, min_cases=None):
    min_cases = C.DISEASE_MIN_CASES if min_cases is None else min_cases
    dc = disease_counts(master, dcols)
    return dc[dc["n_case"] >= min_cases]["disease"].tolist()


# --------------------------------------------------------------------------- #
# Extreme-aging burden score
# --------------------------------------------------------------------------- #
def add_burden(master, method="top20"):
    """Add cumulative extreme-aging burden score (0..18) and grouped version."""
    organ_cols = [gap_col(o) for o in C.ORGANS]
    flags = pd.DataFrame(index=master.index)
    for oc in organ_cols:
        s = pd.to_numeric(master[oc], errors="coerce")
        if method == "top20":
            thr = s.quantile(1 - C.EXTREME_TOP_PCT)
        elif method == "top25":
            thr = s.quantile(1 - C.EXTREME_TOP_PCT_SENS)
        elif method == "z1":
            thr = s.mean() + C.EXTREME_ALT_Z * s.std()
        else:
            raise ValueError(method)
        flags[oc] = (s > thr).astype(float)
        flags.loc[s.isna(), oc] = np.nan
    burden = flags.sum(axis=1, min_count=1)
    col = f"burden_{method}"
    master[col] = burden
    grp = burden.clip(upper=4)  # 0,1,2,3,4+
    master[f"{col}_grp"] = grp
    return master, col


def merge_small_groups(master, grp_col, min_n=100):
    """Merge adjacent burden groups with < min_n into the neighbour below."""
    counts = master[grp_col].value_counts().sort_index()
    mapping = {}
    ordered = sorted(counts.index)
    for g in ordered:
        mapping[g] = g
    # merge upward-sparse groups downward
    for g in reversed(ordered):
        if counts.get(g, 0) < min_n and g > ordered[0]:
            mapping[g] = mapping[g - 1] if (g - 1) in mapping else g - 1
    # resolve chains
    def resolve(g):
        seen = set()
        while mapping[g] != g and g not in seen:
            seen.add(g); g = mapping[g]
        return g
    new = master[grp_col].map(lambda x: resolve(x) if pd.notna(x) else x)
    return new
