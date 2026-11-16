# -*- coding: utf-8 -*-
"""Central configuration for the organ-aging analysis pipeline.

All tunable parameters, file locations, variable definitions and the brain-metric
registry live here so the analysis modules stay declarative.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repository root = parent of this pipeline/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Input tables (see data/README.md for how to obtain them).
# Override with the ORGAN_AGE_DATA_DIR environment variable if your data
# directory lives elsewhere.
DATA_DIR = Path(os.environ.get("ORGAN_AGE_DATA_DIR", REPO_ROOT / "data"))

# Output directory. Override with ORGAN_AGE_RESULTS_DIR if desired.
RESULTS_DIR = Path(os.environ.get("ORGAN_AGE_RESULTS_DIR", REPO_ROOT / "results"))

FILES = {
    "age_gap":      DATA_DIR / "merged_age_gap.csv",      # raw organ age gaps (NOT z)
    "brain_global": DATA_DIR / "全局脑指标.csv",
    "brain_subcort": DATA_DIR / "脑结构指标.csv",
    "brain_cortex": DATA_DIR / "脑结构指标2.csv",
    "clinical":     DATA_DIR / "Clinical_vars.csv",
    "cognition":    DATA_DIR / "Cognition9.csv",
    "disease":      DATA_DIR / "final_analysis_table.csv",
}

# --------------------------------------------------------------------------- #
# Global analysis parameters
# --------------------------------------------------------------------------- #
RANDOM_SEED = 12345
FDR_ALPHA = 0.05
BOOTSTRAP_N = 200
BOOTSTRAP_CI = "BCa"
WINSOR_SD = 3.0
EXTREME_TOP_PCT = 0.20          # top 20% per organ = extreme (accelerated) aging
EXTREME_ALT_Z = 1.0             # alternative: age gap z > 1
EXTREME_TOP_PCT_SENS = 0.25     # sensitivity: top 25%
VIF_WARN, VIF_SEVERE = 5.0, 10.0

# Mediation prescreen thresholds and caps (keeps bootstrap tractable).
MY_P = 0.05                     # mediator -> outcome leg significance
XY_TREND_P = 0.10               # exposure -> outcome total-effect trend
MED_MAX_COG = 60                # cap on bootstrapped cognition chains
MED_MAX_DIS = 40                # cap on bootstrapped disease chains

# Disease inclusion (decided from data: see code review).
DISEASE_MIN_CASES = 50          # primary threshold for OR / dose / mediation / sex
DISEASE_D_MIN_CASES = 30        # compute Cohen's d down to 30 (flag small sample)
NEURO_NPI_MIN_CASES = 5         # smallest neuro set (sensitivity only; dementia n=8)
# NPI ranking is NOT robust to including tiny neuro diseases (dementia n=8 flips the
# ranking, Spearman 5-set vs 3-set = 0.44). Primary NPI uses well-powered neuro
# diseases (>=50), sensitivity versions add the small ones with an explicit warning.
NPI_NEURO_SETS = {"primary_ge50": 50, "sens_ge20": 20, "sens_ge5": 5}
DISEASE_SMALL_FLAG = 100        # < this => "small sample, interpret with caution"

# --------------------------------------------------------------------------- #
# Reporting thresholds. The narrative reports EVERY effect whose size clears a
# meaningful threshold (not an arbitrary top-N). Only if fewer than REPORT_MIN_N
# clear the bar do we fall back to ranking (top REPORT_MIN_N). A generous safety
# cap (REPORT_MAX_N) keeps the prose finite; the full lists always live in tables/.
REPORT_R_MIN = 0.10             # |partial r| considered a meaningful association
REPORT_COG_R_MIN = 0.05         # cognition effects are smaller -> lower bar
REPORT_OR_LOG_MIN = 0.140       # |ln OR| ~ OR<=0.87 or >=1.15 (per 1 SD)
REPORT_D_MIN = 0.20             # |Cohen's d| small-effect floor
REPORT_BETA_INT_MIN = 0.05      # |interaction beta| floor
REPORT_AB_MIN = 0.0             # mediation: report all bootstrap-significant paths
REPORT_MIN_N = 8                # if fewer clear the bar, rank the top this-many
REPORT_MAX_N = 40               # safety cap on how many lines the prose lists

# --------------------------------------------------------------------------- #
# Organs (17: 16 peripheral + Brain central)
# --------------------------------------------------------------------------- #
ORGANS = ["Adipose", "Adrenal", "Artery", "Brain", "Esophagus", "Heart", "Immune",
          "Intestine", "Kidney", "Liver", "Lung", "Muscle", "Pancreas", "Salivary",
          "Skin", "Stomach", "Thyroid"]
CENTRAL_ORGAN = "Brain"
PERIPHERAL_ORGANS = [o for o in ORGANS if o != CENTRAL_ORGAN]

def gap_col(organ):
    return f"{organ}_age_gap"

# Physiological grouping used for heatmap axis ordering (not by significance).
ORGAN_SYSTEM_ORDER = [
    "Heart", "Artery",                                   # cardiovascular
    "Liver", "Pancreas", "Kidney", "Adipose", "Muscle",  # metabolic
    "Esophagus", "Stomach", "Intestine",                 # digestive
    "Brain", "Adrenal", "Thyroid",                       # neuro-endocrine
    "Immune", "Lung", "Skin", "Salivary",                # immune / other
]

# --------------------------------------------------------------------------- #
# Disease classification for NPI (neuro vs systemic)
# --------------------------------------------------------------------------- #
NEURO_DISEASES = ["stroke", "parkinsonism", "dementia", "depressive episode",
                  "anxiety disorders"]

def is_neuro_disease(name):
    n = name.lower()
    keys = ["stroke", "parkinson", "dementia", "depress", "anxiety", "alzheim"]
    return any(k in n for k in keys)

# Disease-system taxonomy: maps each disease to its physiological system so we can
# profile how EACH organ's aging maps onto DIFFERENT disease types (the core question),
# not merely a binary neuro-vs-systemic contrast. Matching is by keyword substring;
# the first matching category wins (order matters -> neuro checked first).
DISEASE_CATEGORY_RULES = [
    ("Neuro/Psychiatric", ["stroke", "parkinson", "dementia", "depress", "anxiety",
                            "alzheim", "epilep", "multiple sclerosis", "migraine"]),
    ("Cardiovascular",    ["hypertension", "angina", "coronary", "conduction",
                            "pulmonary heart", "heart failure", "arrhythm",
                            "atrial", "valve", "cardiomyopath", "aneurysm"]),
    ("Cerebro/Vascular",  ["venous thrombosis", "thrombo", "embolism", "varicose",
                            "peripheral vascular", "atheroscler"]),
    ("Metabolic/Endocrine", ["diabetes", "thyroid", "obesity", "lipid",
                             "metabolic", "gout", "adrenal", "pituitary gland"]),
    ("Respiratory",       ["asthma", "copd", "bronch", "emphysema", "respiratory",
                           "pneumon"]),
    ("Musculoskeletal",   ["arthritis", "osteo", "rheumat", "spondyl", "muscle",
                           "joint", "fracture"]),
    ("Digestive/Hepatobiliary", ["gallbladder", "colitis", "pancreas", "hepat",
                                 "liver", "gastr", "bowel", "crohn", "ulcer",
                                 "biliary", "cirrhos", "intestin"]),
    ("Renal/Urinary",     ["renal", "kidney", "urinary", "nephr", "bladder"]),
    ("Hematologic",       ["anaem", "anemia", "blood disease", "leuk", "lymph",
                           "coagul", "platelet"]),
    ("Dermatologic",      ["psoriasis", "dermat", "eczema", "skin"]),
    ("Neoplasm",          ["cancer", "neoplasm", "tumour", "tumor", "carcinoma",
                           "malignan"]),
]

def disease_category(name):
    n = name.lower()
    for cat, keys in DISEASE_CATEGORY_RULES:
        if any(k in n for k in keys):
            return cat
    return "Other/Unclassified"

# --------------------------------------------------------------------------- #
# Cognition specification (UKB instance 2 = imaging visit)
# direction: +1 => higher is better; -1 => higher is worse
# --------------------------------------------------------------------------- #
COGNITION = {
    "FluidIntelligence":        {"col": "20016-2.0", "log": False, "direction": +1},
    "ReactionTime":             {"col": "20023-2.0", "log": True,  "direction": -1},
    "NumericMemory":            {"col": "4282-2.0",  "log": False, "direction": +1},
    "TowerRearranging":         {"col": "21004-2.0", "log": False, "direction": +1},
    "PairedAssociateLearning":  {"col": "20197-2.0", "log": False, "direction": +1},
    "TrailMaking_Numeric":      {"col": "6348-2.0",  "log": True,  "direction": -1},
    "TrailMaking_Alphanumeric": {"col": "6350-2.0",  "log": True,  "direction": -1},
}
# Requirement fields 23324 (Symbol digit) and 6373 (Matrix pattern) are NOT present
# in Cognition9.csv, so they cannot be analysed. Documented in the report.

# --------------------------------------------------------------------------- #
# Output subfolders
# --------------------------------------------------------------------------- #
MODULE_DIRS = {
    "M0": RESULTS_DIR / "M0_descriptive",
    "M1": RESULTS_DIR / "M1_brain",
    "M2": RESULTS_DIR / "M2_disease",
    "M3": RESULTS_DIR / "M3_cognition",
    "M4": RESULTS_DIR / "M4_mediation",
    "M5": RESULTS_DIR / "M5_sex",
}

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for d in MODULE_DIRS.values():
        (d / "tables").mkdir(parents=True, exist_ok=True)
        (d / "figures").mkdir(parents=True, exist_ok=True)
