# -*- coding: utf-8 -*-
"""Robustness of NPI ranking to which neuro diseases are included."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import config as C, data_io as io
from config import gap_col, ORGANS
import stats_utils as S

frames = io.load_all()
master, _ = io.build_master(frames)
dcols = io.disease_columns(frames)
organ_cols = [gap_col(o) for o in ORGANS]
G = master[master[organ_cols].notna().all(axis=1)
           & master["age"].notna() & master["gender"].notna()].copy()
dc = io.disease_counts(G, dcols).set_index("disease")

neuro_all = [d for d in dcols if C.is_neuro_disease(d)]
neuro_n = {d: int(dc.loc[d, "n_case"]) for d in neuro_all}
print("neuro diseases + n:", neuro_n)
systemic = [d for d in dcols if not C.is_neuro_disease(d)
            and int(dc.loc[d, "n_case"]) >= C.DISEASE_D_MIN_CASES]

def dmat(diseases):
    out = {}
    for dis in diseases:
        dv = pd.to_numeric(G[dis], errors="coerce")
        out[dis] = {o: abs(S.cohens_d(pd.to_numeric(G[gap_col(o)], errors="coerce")[dv == 1],
                                      pd.to_numeric(G[gap_col(o)], errors="coerce")[dv == 0]))
                    for o in ORGANS}
    return pd.DataFrame(out)  # organ x disease

sys_d = dmat(systemic)
sys_strength = sys_d.mean(axis=1)

sets = {
    "5 (incl dementia8+park23)": neuro_all,
    "4 (drop dementia8)":        [d for d in neuro_all if neuro_n[d] >= 20],
    "3 (>=50 only)":             [d for d in neuro_all if neuro_n[d] >= 50],
}
tab = pd.DataFrame({"system_strength": sys_strength})
for label, ns in sets.items():
    nd = dmat(ns)
    ns_strength = nd.mean(axis=1)
    npi = (ns_strength - sys_strength) / (ns_strength + sys_strength)
    tab[f"neuroStr[{label}]"] = ns_strength
    tab[f"NPI[{label}]"] = npi

pd.set_option("display.width", 200, "display.max_rows", 30, "display.float_format", lambda v: f"{v:+.3f}")
cols_npi = [c for c in tab.columns if c.startswith("NPI")]
print("\n=== NPI under different neuro sets (sorted by 3-disease) ===")
print(tab[cols_npi].sort_values(cols_npi[-1], ascending=False).to_string())
print("\n=== rank correlations between neuro-set NPIs ===")
print(tab[cols_npi].corr(method="spearman").to_string())
print("\n=== neuro strength vs system strength (3-disease set) ===")
show = tab[["neuroStr[3 (>=50 only)]", "system_strength", cols_npi[-1]]].sort_values(cols_npi[-1], ascending=False)
print(show.to_string())
