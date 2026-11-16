# -*- coding: utf-8 -*-
"""End-to-end driver for the organ-aging analysis pipeline."""
import sys, os, json, time, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import config as C
import data_io as io
import m0_descriptive, m1_brain, m2_disease, m3_cognition, m4_mediation, m5_sex
import report
import final_summary


def main():
    t0 = time.time()
    C.ensure_dirs()
    print("[load] reading sources ...")
    frames = io.load_all()
    reg = io.build_brain_registry(frames)
    reg.to_csv(C.RESULTS_DIR / "brain_metric_registry.csv", index=False)
    master, wlog = io.build_master(frames)
    dcols = io.disease_columns(frames)
    print(f"[load] master={master.shape}, brain metrics={len(reg)}, diseases={len(dcols)}")

    print("[M0] descriptive / QC ...")
    r0 = m0_descriptive.run(master, reg, wlog, dcols)
    print("[M1] brain ...")
    r1 = m1_brain.run(master, reg)
    print(f"      M1-A significant (BH): {r1['n_sig_m1a']}")
    print("[M2] disease ...")
    r2 = m2_disease.run(master, dcols)
    print("[M3] cognition ...")
    r3 = m3_cognition.run(master)
    print("[M4] mediation ...")
    r4 = m4_mediation.run(master, reg, r1["m1a"], r2["m2a"], r3["m3a"], dcols)
    print(f"      significant mediations: {r4['n_sig']}")
    print("[M5] sex ...")
    r5 = m5_sex.run(master, reg, dcols)
    print(f"      significant interactions (BH): {r5['n_sig_interactions']}")

    elapsed = time.time() - t0
    print("[report] writing report / guide / manifest ...")
    manifest = build_manifest(master, reg, dcols, r0, r1, r2, r3, r4, r5, elapsed)
    with open(C.RESULTS_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    report.write_report(master, reg, dcols, r0, r1, r2, r3, r4, r5, manifest)
    report.write_methods_guide(master, reg, dcols, manifest)
    final_summary.build()
    print(f"[done] {elapsed:.1f}s  ->  {C.RESULTS_DIR}")


def build_manifest(master, reg, dcols, r0, r1, r2, r3, r4, r5, elapsed):
    import scipy, statsmodels, matplotlib, seaborn
    return {
        "pipeline": "organ-aging multi-module analysis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "random_seed": C.RANDOM_SEED,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
        },
        "parameters": {
            "FDR_alpha": C.FDR_ALPHA, "fdr_methods": ["BH (per-organ family)", "BY (global)"],
            "bootstrap_n": C.BOOTSTRAP_N, "bootstrap_ci": C.BOOTSTRAP_CI,
            "winsor_sd": C.WINSOR_SD, "extreme_top_pct": C.EXTREME_TOP_PCT,
            "extreme_alt_z": C.EXTREME_ALT_Z,
            "disease_min_cases_primary": C.DISEASE_MIN_CASES,
            "disease_min_cases_cohensd": C.DISEASE_D_MIN_CASES,
            "partial_corr_method": "direct multiple OLS regression (no residualisation)",
            "mediation_prescreen": {"MY_p": C.MY_P, "XY_trend_p": C.XY_TREND_P,
                                    "max_cog": C.MED_MAX_COG, "max_dis": C.MED_MAX_DIS},
        },
        "organs": C.ORGANS, "central_organ": C.CENTRAL_ORGAN,
        "n_brain_metrics": int(len(reg)),
        "brain_metric_classes": reg["klass"].value_counts().to_dict(),
        "n_disease_columns": len(dcols),
        "diseases_included_primary": r2["included"],
        "sample_composition": r0,
        "results_summary": {
            "M1A_significant_BH": r1["n_sig_m1a"],
            "M4_significant_mediations": r4["n_sig"],
            "M5_significant_interactions_BH": r5["n_sig_interactions"],
        },
        "inputs": {k: str(v) for k, v in C.FILES.items()},
        "outputs": {k: str(v) for k, v in C.MODULE_DIRS.items()},
    }


if __name__ == "__main__":
    main()
