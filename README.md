# Multi-organ proteomic aging and brain structure

Analysis code for **"Brain-synchronous and brain-asynchronous patterns of
multi-organ ageing: insights from peripheral organ proteomic age gaps in UK
Biobank"**.

Using baseline plasma proteomes from 44,810 UK Biobank participants, we built
LASSO aging clocks for 17 organ systems from organ-enriched protein panels,
computed organ age gaps, and related them to 111 structural brain MRI measures,
16 chronic diseases and 7 cognitive tests obtained ~11 years later, including
mediation analyses through brain structure.

## Repository layout

```
scripts/                  upstream pipeline (proteomics -> organ clocks)
  01_preprocess_proteomics.py   QC, z-score, KNN imputation (k = 152)
  02_organ_enrichment.py        ST2 organ-enriched panels -> per-organ matrices
  03_train_organ_clocks.py      LassoCV organ clocks, age-bias-corrected gaps
pipeline/                 downstream analysis (modules M0-M5)
  config.py                   all paths and tunable parameters
  data_io.py                  loading, brain-metric registry, winsorization
  stats_utils.py              regressions, partial correlations, BH/BY FDR,
                              BCa bootstrap mediation
  m0_descriptive.py           cohort descriptives and QC tables
  m1_brain.py                 organ age gaps x 111 MRI measures
  m2_disease.py               organ age gaps x 16 diseases (+ burden OR)
  m3_cognition.py             organ age gaps x 7 cognitive tests (+ dose)
  m4_mediation.py             indirect effects through brain structure
  m5_sex.py                   sex-stratified sensitivity analyses
  run_pipeline.py             end-to-end driver
  make_figures.py             main-text figures
  make_network.py             integration/network figures
  p1_corr_cluster.py          inter-organ correlation clustering
data/README.md            how to obtain the required UK Biobank files
reference/                organ -> protein mappings (from Oh et al. ST2)
models/                   fitted LASSO coefficients (JSON, aggregate only)
results/                  all module output tables, figures and run manifest
figures/                  manuscript figures (PNG + SVG)
```

## Requirements

Python 3.10+ with `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`,
`matplotlib`, `seaborn`, `networkx`, `openpyxl` (see `requirements.txt`).

## Reproducing the analysis

UK Biobank data are available to approved researchers only and **cannot be
redistributed**; this repository therefore ships code, fitted model weights,
reference mappings and aggregate results, but no participant-level data.

1. Obtain the input files listed in `data/README.md` through your UK Biobank
   project and place them under `data/` (or set `ORGAN_AGE_DATA_DIR`).
2. Build the organ clocks:

   ```bash
   python scripts/01_preprocess_proteomics.py
   python scripts/02_organ_enrichment.py
   python scripts/03_train_organ_clocks.py
   ```

3. Assemble the analysis inputs described in `data/README.md`
   (`merged_age_gap.csv`, brain MRI tables, clinical, cognition and disease
   tables) and run the module pipeline:

   ```bash
   python pipeline/run_pipeline.py        # M0-M5 + reports
   python pipeline/make_figures.py        # figures into figures/
   python pipeline/make_network.py
   ```

Key settings (all in `pipeline/config.py`): BH-FDR applied within organ
(`FDR_ALPHA = 0.05`), age gaps winsorized at ±3 SD, "extreme aging" = top 20%
per organ (top 25% and z > 1 as sensitivity definitions), mediation screened at
M–Y p < .05 and X–Y p < .10 with 200 BCa bootstrap resamples.

## Data availability

UK Biobank data are accessible via the UK Biobank Access Management System
(https://www.ukbiobank.ac.uk). No individual-level data are included in this
repository. `results/` contains aggregate statistics only.

## Citation

If you use this code, please cite the manuscript (in preparation).
