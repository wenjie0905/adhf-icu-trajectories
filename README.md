# ADHF ICU multitrajectory main analysis

Reproducible code for the completed main analysis of five-day decongestion
trajectories in the MIMIC-IV v2.2 ADHF ICU cohort. Patient-level data, model
objects, and derived results are intentionally excluded.

## Frozen main-analysis definition

- Strict source cohort: 1,458 first qualifying ICU stays.
- Primary trajectory set: GE3 (at least three complete marker-days; completed
  run: 1,282 stays).
- Complete-five-day sensitivity set: FULL5 (completed run: 695 stays).
- Four longitudinal markers: cumulative net fluid balance, urine output,
  creatinine, and IV loop-diuretic dose in furosemide equivalents.
- Primary pooled, outcome-blind model: GE3, K=4.
- Candidate grid: K=2 through K=6. K=6 here is only a candidate model; it is
  not the old K=6 primary pipeline. K=3 and K=5 are the adjacent sensitivity
  models used in downstream summaries.
- Primary outcome: 90-day mortality from ICU admission, analysed from the ICU
  day-5 landmark (0-85 days after the landmark).

## The 11 code files

| Order | File | Role |
|---:|---|---|
| 1 | `rebuild_adhf_cohort_v4.py` | Apply the strict HF/ICU, two-window IV loop initiation-or-escalation, exclusion, and first-stay rules. |
| 2 | `prepare_adhf_trajectory_v1.py` | Build terminal-aware ICU day 0-4 markers, pooled winsorisation/z-scaling, completeness, and landmark outcomes. |
| 3 | `prepare_adhf_clustering_inputs_v1.py` | Produce GE2, GE3, and FULL5 model-ready long, daily, and matrix inputs. |
| 4 | `fit_adhf_gbmt_grid_v1.R` | Fit the outcome-blind GBMT candidate grid and save assignments/posteriors. |
| 5 | `analyze_adhf_gbmt_outcomes_v1.R` | Freeze GE3-K4 as primary and report phenotype, trajectory, posterior, outcome, KM, Cox, and FULL5 crosswalk summaries. |
| 6 | `prepare_adhf_mortality_covariates_v2.py` | Extract demographics, ICD comorbidity, first-24-hour laboratory, and support covariates. |
| 7 | `prepare_adhf_plan_covariates_v2.py` | Join/audit EF, severity, admission physiology/rhythm, NT-proBNP/congestion, and chronic-HF-therapy supplements. |
| 8 | `prepare_adhf_plan_mortality_input_v2.py` | Build the GE3-K4 day-5 landmark analysis table directly from frozen assignments, outcomes, cohort, and plan covariates. |
| 9 | `adhf_survival_utils.py` | Shared Cox, IPCW AUC/Brier, calibration, and NRI/IDI functions. |
| 10 | `validate_adhf_plan_survival_v2.py` | Shared design/diagnostic functions required by file 11; it can also run frozen temporal validation when temporal assignments are supplied. |
| 11 | `fit_adhf_plan_primary_v2.py` | Fit the plan-complete base and base+phenotype Cox models and evaluate incremental value, PH, calibration, integrated Brier, DCA, KM, and phenotype-by-EF interaction. |

Files 9 and 10 must remain in the same directory as file 11 so that the local
Python imports resolve.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

R packages:

```r
install.packages(c("gbmt", "survival"))
```

## Main execution order

The following is a path-explicit template. Replace every `/path/...` value
with a local path; no script contains a hard-coded personal directory.

```bash
python code/rebuild_adhf_cohort_v4.py \
  --mimic-dir /path/mimiciv/2.2 \
  --candidate-cohort /path/hf_icu_candidates.csv \
  --output-dir /path/results/01_cohort

python code/prepare_adhf_trajectory_v1.py \
  --mimic-dir /path/mimiciv/2.2 \
  --cohort /path/results/01_cohort/adhf_icu_cohort_v4.csv \
  --output-dir /path/results/02_trajectory

python code/prepare_adhf_clustering_inputs_v1.py \
  --trajectory-dir /path/results/02_trajectory \
  --cohort /path/results/01_cohort/adhf_icu_cohort_v4.csv \
  --output-dir /path/results/03_gbmt_input \
  --primary-set GE3

Rscript code/fit_adhf_gbmt_grid_v1.R \
  --input_dir /path/results/03_gbmt_input \
  --output_dir /path/results/04_gbmt \
  --sets GE3,FULL5 \
  --k_min 2 \
  --k_max 6

Rscript code/analyze_adhf_gbmt_outcomes_v1.R \
  --model_input_dir /path/results/03_gbmt_input \
  --trajectory_dir /path/results/02_trajectory \
  --gbmt_dir /path/results/04_gbmt \
  --cohort /path/results/01_cohort/adhf_icu_cohort_v4.csv \
  --output_dir /path/results/05_phenotypes \
  --sets GE3,FULL5 \
  --ks 3,4,5 \
  --primary_set GE3 \
  --primary_k 4

python code/prepare_adhf_mortality_covariates_v2.py \
  --mimic-dir /path/mimiciv/2.2 \
  --cohort /path/results/01_cohort/adhf_icu_cohort_v4.csv \
  --output-dir /path/results/06_base_covariates

python code/prepare_adhf_plan_covariates_v2.py \
  --cohort /path/results/01_cohort/adhf_icu_cohort_v4.csv \
  --base-covariates /path/results/06_base_covariates/analysis_covariates_newcohort_v1.csv \
  --supplement sofa=/path/supplements/sofa_first24.csv \
  --supplement ef=/path/supplements/ef_category.csv \
  --supplement hemodynamics=/path/supplements/admission_vitals.csv \
  --supplement biomarker=/path/supplements/ntprobnp.csv \
  --supplement hf_therapy=/path/supplements/chronic_hf_therapy.csv \
  --output-dir /path/results/07_plan_covariates

python code/prepare_adhf_plan_mortality_input_v2.py \
  --cohort /path/results/01_cohort/adhf_icu_cohort_v4.csv \
  --trajectory-dir /path/results/02_trajectory \
  --gbmt-dir /path/results/04_gbmt \
  --plan-covariates /path/results/07_plan_covariates/analysis_covariates_plan_v2.csv \
  --model-set GE3 \
  --k 4 \
  --output-dir /path/results/08_landmark_input

python code/fit_adhf_plan_primary_v2.py \
  --input /path/results/08_landmark_input/mortality_input_plan_v2.csv \
  --output-dir /path/results/09_primary_survival \
  --severity-score sofa \
  --reference-class 1
```

The plan-covariate gate is deliberately strict. The supplement names are only
labels; the input CSV columns are resolved through the conservative aliases in
the script. Use `--allow-missing-plan-fields` only for an audit, not for the
final adjusted model.

## Repository hygiene

Do not commit MIMIC source tables, patient-level CSV/RDS files, cached model
objects, or generated result directories. The included `.gitignore` blocks
the usual local data and output paths. MIMIC-IV access remains subject to its
data-use agreement.

## Scope boundary

This package is the current strict-cohort, pooled GE3-K4 main analysis. It does
not contain the obsolete 2,508-stay/K=6 primary pipeline. The optional temporal
validation CLI in `validate_adhf_plan_survival_v2.py` expects separately frozen
K=4 temporal assignments and is not invoked by the main commands above.
