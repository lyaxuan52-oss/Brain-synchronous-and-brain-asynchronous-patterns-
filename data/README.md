# Input data

Participant-level UK Biobank data **cannot be redistributed**. Place the
following files here (or point `ORGAN_AGE_DATA_DIR` / the script-level
environment variables at your own directory) after extracting them under an
approved UK Biobank project.

## For `scripts/` (organ-clock construction)

| File | Content |
| --- | --- |
| `olink_instance0.csv` | UKB-PPP Olink Explore 3072 NPX matrix, baseline (instance 0); first column = participant ID, remaining columns named `olink_instance_0.<ASSAY>` |
| `ReceptionAge.csv` | chronological age at baseline blood sampling (field `21003-0.0`) keyed by `eid` |
| `NatMed25_supp.xlsx` | Oh et al. (2025, Nat. Med.) supplementary table ST2 (organ-enrichment flags); download from the journal page |

## For `pipeline/` (modules M0-M5)

`pipeline/config.py` (`FILES` dict) expects these tables in `data/`:

| File | Content |
| --- | --- |
| `merged_age_gap.csv` | one row per `eid`; columns `<Organ>_age_gap` (age-bias-corrected gap per clock, built by `scripts/03`) |
| `全局脑指标.csv` | 34 global FreeSurfer metrics incl. `eTIV` (ID column `ID` formatted `<eid>_...`) |
| `脑结构指标.csv` | 15 subcortical volumes, keyed by `eid` |
| `脑结构指标2.csv` | 62 cortical regional volumes, keyed by `eid` |
| `Clinical_vars.csv` | clinical extract incl. sex (`31-0.0`) and imaging-visit age (`21003-2.0`) |
| `Cognition9.csv` | cognitive battery, instance 2 (fields `20016-2.0`, `20023-2.0`, `4282-2.0`, `21004-2.0`, `20197-2.0`, `6348-2.0`, `6350-2.0`) |
| `final_analysis_table.csv` | disease indicator table (columns from `coronary heart disease` through `lymphoma and blood cancer`, plus `eid`, `gender`, `age`) |

Field numbers refer to the UK Biobank showcase. Column names may be renamed in
`pipeline/config.py` / `pipeline/data_io.py` if your extract differs.
