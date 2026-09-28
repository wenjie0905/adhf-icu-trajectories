#!/usr/bin/env python3
"""Rebuild baseline covariates for the final strict-plan cohort.

This is an optional first step when the old ``analysis_covariates_fullcohort``
file is unavailable.  It computes demographics, ICD-derived comorbidities,
first-24-hour baseline laboratory values, and first-24-hour treatment flags
from MIMIC-IV v2.2.  It does not calculate SOFA/SAPS-II or infer EF/BNP; those
should be joined from the official score/measurement exports before the
primary adjusted model if they are required by the analysis plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


COMORBIDITY_RULES = {
    "diabetes": r"^(250|E08|E09|E10|E11|E13)",
    "ckd": r"^(585|586|V451|V56|N18|N19|Z992)",
    "copd": r"^(490|491|492|4932|494|496|J40|J41|J42|J43|J44)",
    "atrial_fibrillation": r"^(42731|42732|I48)",
    "prior_mi": r"^(410|412|I21|I22|I252)",
    "hypertension": r"^(401|402|403|404|405|I10|I11|I12|I13|I15)",
    "valvular_disease": r"^(394|395|396|397|424|I05|I06|I07|I08|I34|I35|I36|I37|I38|I39)",
    "anemia": r"^(280|281|282|283|284|285|D50|D51|D52|D53|D55|D56|D57|D58|D59|D60|D61|D62|D63|D64)",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mimic-dir", required=True, type=Path)
    p.add_argument("--cohort", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    return p.parse_args()


def q(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("'", "''")


def read_all(con, path: Path, name: str) -> None:
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('{q(path)}', header=true, union_by_name=true, null_padding=true, all_varchar=true)")


def main() -> None:
    a = args()
    root, cohort_path, out = a.mimic_dir.expanduser().resolve(), a.cohort.expanduser().resolve(), a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    required = [cohort_path, root / "hosp" / "patients.csv.gz", root / "hosp" / "diagnoses_icd.csv.gz", root / "hosp" / "d_labitems.csv.gz", root / "hosp" / "labevents.csv.gz", root / "icu" / "d_items.csv.gz", root / "icu" / "inputevents.csv.gz"]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))
    cohort = pd.read_csv(cohort_path, low_memory=False)
    needed = ["subject_id", "hadm_id", "stay_id", "intime"]
    miss = [x for x in needed if x not in cohort.columns]
    if miss:
        raise ValueError(f"Cohort missing columns: {miss}")
    for c in ["subject_id", "hadm_id", "stay_id"]:
        cohort[c] = pd.to_numeric(cohort[c], errors="raise").astype("int64")
    cohort = cohort.drop_duplicates("stay_id").copy()
    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce")
    if cohort["intime"].isna().any():
        raise ValueError("Cohort contains missing/invalid intime")

    # ``age_at_admission`` is a field carried by the rebuilt cohort; it is
    # not a column in MIMIC-IV ``hosp.admissions``.  Normalize the field here
    # so the SQL below never tries to read it from the admissions table.  The
    # fallback aliases are only for older candidate files; the final strict
    # cohort already contains age_at_admission.
    if "age_at_admission" not in cohort.columns:
        for alias in ("age", "age_for_analysis"):
            if alias in cohort.columns:
                cohort["age_at_admission"] = cohort[alias]
                break
        else:
            cohort["age_at_admission"] = np.nan
    cohort["age_at_admission"] = pd.to_numeric(cohort["age_at_admission"], errors="coerce")

    con = duckdb.connect(database=":memory:")
    con.register("cohort_df", cohort[["subject_id", "hadm_id", "stay_id", "intime", "age_at_admission"]])
    con.execute("CREATE OR REPLACE TABLE c AS SELECT * FROM cohort_df")
    read_all(con, root / "hosp" / "patients.csv.gz", "patients")
    read_all(con, root / "hosp" / "diagnoses_icd.csv.gz", "dx")
    read_all(con, root / "hosp" / "d_labitems.csv.gz", "d_lab")
    read_all(con, root / "hosp" / "labevents.csv.gz", "labs")
    read_all(con, root / "icu" / "d_items.csv.gz", "d_items")
    read_all(con, root / "icu" / "inputevents.csv.gz", "inputs")

    demo = con.execute("""
      SELECT c.subject_id,c.hadm_id,c.stay_id,c.intime,
             coalesce(try_cast(c.age_at_admission AS DOUBLE),
                      try_cast(p.anchor_age AS DOUBLE)) AS age_for_analysis,
             upper(coalesce(p.gender,'')) AS sex,
             p.anchor_year_group
      FROM c LEFT JOIN patients p ON try_cast(p.subject_id AS BIGINT)=c.subject_id
    """).df()
    demo["sex_female"] = (demo["sex"].astype("string").str.upper() == "F").astype(float)
    demo["age_per10"] = demo["age_for_analysis"] / 10.0

    dx_cte = """
      WITH x AS (
        SELECT c.stay_id, upper(replace(cast(d.icd_code AS varchar),'.','')) AS icd
        FROM dx d JOIN c ON try_cast(d.subject_id AS BIGINT)=c.subject_id AND try_cast(d.hadm_id AS BIGINT)=c.hadm_id
      )
      SELECT stay_id,
    """
    expressions = [f"max(CASE WHEN regexp_matches(icd, '{rule}') THEN 1 ELSE 0 END) AS {name}" for name, rule in COMORBIDITY_RULES.items()]
    comorb = con.execute(dx_cte + ",\n".join(expressions) + " FROM x GROUP BY stay_id").df()

    # ``label`` can be parsed as a reserved identifier in some DuckDB
    # versions.  Use a quoted source column and a non-reserved alias.
    lab_map = con.execute("""
      WITH item_labels AS (
        SELECT try_cast(d.itemid AS BIGINT) AS itemid,
               lower(coalesce(cast(d."label" AS VARCHAR), '')) AS item_label
        FROM d_lab AS d
      )
      SELECT itemid, item_label,
        CASE
          WHEN item_label='creatinine' THEN 'creatinine'
          -- MIT-LCP chemistry.sql: blood BUN only, not urine/body-fluid urea.
          WHEN itemid=51006 THEN 'bun'
          WHEN item_label='sodium' THEN 'sodium'
          WHEN item_label='potassium' THEN 'potassium'
          WHEN item_label='bicarbonate' THEN 'bicarbonate'
          WHEN item_label='chloride' THEN 'chloride'
          WHEN item_label='hemoglobin' THEN 'hemoglobin'
          WHEN item_label='albumin' THEN 'albumin'
          WHEN item_label LIKE '%nt-pro%' OR item_label LIKE '%nt pro%' OR item_label LIKE '%bnp%' THEN 'ntprobnp'
          ELSE NULL END AS lab_name
      FROM item_labels
    """).df()
    con.register("lab_map_df", lab_map)
    con.execute("CREATE OR REPLACE TABLE lab_map_tmp AS SELECT * FROM lab_map_df")
    labs_long = con.execute("""
      SELECT c.stay_id,m.lab_name,
             try_cast(l.valuenum AS DOUBLE) AS lab_value,
             try_cast(l.charttime AS TIMESTAMP) AS lab_charttime,
             try_cast(l.labevent_id AS BIGINT) AS lab_event_id
      FROM labs l JOIN c ON try_cast(l.subject_id AS BIGINT)=c.subject_id AND try_cast(l.hadm_id AS BIGINT)=c.hadm_id
      JOIN lab_map_tmp m ON try_cast(l.itemid AS BIGINT)=m.itemid
      WHERE m.lab_name IS NOT NULL AND try_cast(l.valuenum AS DOUBLE) IS NOT NULL
        -- Official BUN cleaning: retain 0 < valuenum <= 300 mg/dL.
        -- Filter before selecting the first valid value in the baseline window.
        AND (m.lab_name <> 'bun' OR
             (try_cast(l.valuenum AS DOUBLE)>0 AND try_cast(l.valuenum AS DOUBLE)<=300))
        AND try_cast(l.charttime AS TIMESTAMP)>=c.intime
        AND try_cast(l.charttime AS TIMESTAMP)<c.intime+INTERVAL '24 hours'
    """).df()
    if not labs_long.empty:
        labs_long = labs_long.sort_values(["stay_id", "lab_name", "lab_charttime"])
        labs_first = labs_long.drop_duplicates(["stay_id", "lab_name"])[["stay_id", "lab_name", "lab_value"]].pivot(index="stay_id", columns="lab_name", values="lab_value").reset_index()
        labs_first.columns = ["stay_id" if x == "stay_id" else f"baseline_{x}" for x in labs_first.columns]
        # Make BUN ties reproducible, without changing other lab selections.
        bun_first = (labs_long[labs_long.lab_name.eq("bun")]
                     .sort_values(["stay_id", "lab_charttime", "lab_event_id"], kind="stable")
                     .drop_duplicates("stay_id").set_index("stay_id").lab_value)
        labs_first["baseline_bun"] = labs_first.stay_id.map(bun_first)
    else:
        labs_first = pd.DataFrame({"stay_id": cohort.stay_id})

    item_map = con.execute("""
      SELECT try_cast(d.itemid AS BIGINT) AS itemid,
             lower(coalesce(cast(d."label" AS VARCHAR), '')) AS item_label
      FROM d_items AS d
    """).df()
    con.register("item_map_df", item_map)
    con.execute("CREATE OR REPLACE TABLE item_map_tmp AS SELECT * FROM item_map_df")
    treatment = con.execute("""
      WITH ev AS (
        SELECT c.stay_id,
          CASE
            WHEN regexp_matches(m.item_label,'norepinephrine|levophed|epinephrine|phenylephrine|neosynephrine|vasopressin|dopamine') THEN 'vasopressor_first24h'
            WHEN regexp_matches(m.item_label,'dobutamine|milrinone') THEN 'inotrope_first24h'
            WHEN regexp_matches(m.item_label,'mechanical vent|ventilation|intubation|tracheostomy') THEN 'mechanical_ventilation_first24h'
            WHEN regexp_matches(m.item_label,'dialysis|crrt|cvvh|cvvhd|hemodialysis|ultrafiltration|renal replacement') THEN 'rrt_first24h'
            ELSE NULL END event_type
        FROM inputs i JOIN c ON try_cast(i.stay_id AS BIGINT)=c.stay_id
        JOIN item_map_tmp m ON try_cast(i.itemid AS BIGINT)=m.itemid
        WHERE try_cast(i.starttime AS TIMESTAMP)>=c.intime
          AND try_cast(i.starttime AS TIMESTAMP)<c.intime+INTERVAL '24 hours'
      )
      SELECT stay_id,
        max(CASE WHEN event_type='vasopressor_first24h' THEN 1 ELSE 0 END) vasopressor_first24h,
        max(CASE WHEN event_type='inotrope_first24h' THEN 1 ELSE 0 END) inotrope_first24h,
        max(CASE WHEN event_type='mechanical_ventilation_first24h' THEN 1 ELSE 0 END) mechanical_ventilation_first24h,
        max(CASE WHEN event_type='rrt_first24h' THEN 1 ELSE 0 END) rrt_first24h
      FROM ev GROUP BY stay_id
    """).df()

    cov = demo.merge(comorb, on="stay_id", how="left").merge(labs_first, on="stay_id", how="left").merge(treatment, on="stay_id", how="left")
    if "baseline_bun" not in cov.columns:
        cov["baseline_bun"] = np.nan
    for col in list(COMORBIDITY_RULES) + ["vasopressor_first24h", "inotrope_first24h", "mechanical_ventilation_first24h", "rrt_first24h"]:
        if col not in cov.columns:
            cov[col] = 0
        cov[col] = pd.to_numeric(cov[col], errors="coerce").fillna(0).astype(int)
    cov["comorbidity_count_v1"] = cov[list(COMORBIDITY_RULES)].sum(axis=1)
    cov.to_csv(out / "analysis_covariates_newcohort_v1.csv", index=False)
    pd.DataFrame({"variable": cov.columns, "n_missing": cov.isna().sum().values, "pct_missing": 100 * cov.isna().mean().values}).to_csv(out / "analysis_covariates_missingness_v1.csv", index=False)
    cov.select_dtypes(include=[np.number]).describe().T.to_csv(out / "analysis_covariates_summary_v1.csv")
    (out / "RUN_METADATA.json").write_text(json.dumps({
        "script": "prepare_adhf_mortality_covariates_v2.py",
        "bun_extraction_version": "official_blood_itemid_range_20260917",
        "bun_rule": "labevents itemid=51006; 0 < valuenum <= 300 mg/dL; first valid observation in ICU [0,24) hours; absent valid values remain missing",
        "bun_rule_source": "https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/chemistry.sql",
        "mimic_version": "2.2", "cohort": str(cohort_path), "cohort_n": int(len(cohort)),
        "baseline_window": "ICU intime through ICU intime + 24 hours",
        "score_sources": "SOFA/SAPS-II not calculated here; join official score exports separately",
        "outputs": ["analysis_covariates_newcohort_v1.csv", "analysis_covariates_missingness_v1.csv", "analysis_covariates_summary_v1.csv"]
    }, indent=2), encoding="utf-8")
    print(f"Saved: {out / 'analysis_covariates_newcohort_v1.csv'}")


if __name__ == "__main__":
    main()
