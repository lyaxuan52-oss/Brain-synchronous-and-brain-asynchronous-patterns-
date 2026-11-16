# -*- coding: utf-8 -*-
"""Step 2 — Assign proteins to organ panels and split the cleaned matrix.

Proteins are assigned to organs using the organ-enrichment annotations of
Oh et al. (2025, Nat. Med.; supplementary table ST2): a protein belongs to an
organ's panel when that organ's ``enriched`` flag is TRUE. Proteins not
enriched in any organ form the "organismal" panel.

Inputs:
    data/raw/NatMed25_supp.xlsx   ST2 sheet of Oh et al. supplementary material
    outputs/preprocessing/olink_cleaned.csv   from step 1
Outputs (under ``outputs/organs/``):
    <Organ>.csv            per-organ protein submatrix (uid + protein columns)
    organismal.csv         proteins not enriched in any organ
    organ_to_proteins.json        organ -> protein list mapping
    organ_protein_counts_st2.csv  panel sizes from ST2
    organ_enrichment_summary.csv  matched panel sizes in the cleaned matrix
"""
import json
import os
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ST2_XLSX = Path(os.environ.get(
    "ST2_XLSX", "data/raw/NatMed25_supp.xlsx"))
CLEANED_CSV = Path(os.environ.get(
    "OLINK_CLEANED_CSV", "outputs/preprocessing/olink_cleaned.csv"))
OUT_DIR = Path(os.environ.get("ORGAN_OUT_DIR", "outputs/organs"))

CHUNK = 10_000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Organ -> enriched protein mapping from ST2
    st2 = pd.read_excel(ST2_XLSX, sheet_name="ST2", dtype=str)
    st2 = st2[st2["enriched"].str.upper() == "TRUE"].copy()
    st2["Assay"] = st2["Assay"].str.strip().str.upper()
    st2["organ"] = st2["organ"].str.strip()

    organ_to_proteins = {
        organ: sorted(group["Assay"].unique())
        for organ, group in st2.groupby("organ")
    }
    organs = sorted(organ_to_proteins)
    print(f"{len(organs)} organs in ST2")

    with open(OUT_DIR / "organ_to_proteins.json", "w",
              encoding="utf-8") as f:
        json.dump(organ_to_proteins, f, indent=2, ensure_ascii=False)

    pd.DataFrame(
        [{"organ": o, "n_proteins_st2": len(p)}
         for o, p in sorted(organ_to_proteins.items())]
    ).to_csv(OUT_DIR / "organ_protein_counts_st2.csv", index=False)

    # 2. Match ST2 assay names to matrix columns ("olink_instance_0.<ASSAY>")
    header = pd.read_csv(CLEANED_CSV, nrows=0)
    uid_col = header.columns[0]
    protein_cols = header.columns[1:].tolist()
    col_to_protein = {
        c: c.replace("olink_instance_0.", "").upper() for c in protein_cols
    }

    organ_to_cols = {}
    all_organ_cols = set()
    for organ, proteins in organ_to_proteins.items():
        matched = [c for c, p in col_to_protein.items() if p in proteins]
        organ_to_cols[organ] = matched
        all_organ_cols.update(matched)
        print(f"  {organ:<15s} {len(matched):3d}/{len(proteins):3d} matched")
    organismal_cols = [c for c in protein_cols if c not in all_organ_cols]

    # 3. Stream the cleaned matrix and write one submatrix per organ
    buckets = {o: [] for o in organs}
    buckets["organismal"] = []
    for chunk in pd.read_csv(CLEANED_CSV, chunksize=CHUNK):
        for organ in organs:
            cols = [uid_col] + organ_to_cols[organ]
            if len(cols) > 1:
                buckets[organ].append(chunk[cols])
        buckets["organismal"].append(chunk[[uid_col] + organismal_cols])

    matched_counts = {}
    for organ, parts in buckets.items():
        if not parts:
            matched_counts[organ] = 0
            continue
        sub = pd.concat(parts, ignore_index=True)
        sub.to_csv(OUT_DIR / f"{organ}.csv", index=False)
        matched_counts[organ] = len(sub.columns) - 1

    stats = []
    for organ in organs + ["organismal"]:
        row = {"category": organ, "n_proteins": matched_counts.get(organ, 0)}
        if organ in organ_to_proteins:
            n_st2 = len(organ_to_proteins[organ])
            row["n_proteins_st2"] = n_st2
            row["match_rate_pct"] = round(
                100 * row["n_proteins"] / n_st2, 2) if n_st2 else 0
        stats.append(row)
    pd.DataFrame(stats).to_csv(
        OUT_DIR / "organ_enrichment_summary.csv", index=False)
    print(f"Done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
