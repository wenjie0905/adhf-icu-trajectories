#!/usr/bin/env python3
"""Prepare auditable ICU-day 0--4 trajectory data for the rebuilt ADHF cohort.

This is the next step after cohort rebuilding.  It does not fit a trajectory
model.  It extracts and quality-checks the four prespecified daily markers:

* cumulative net fluid balance (running input minus output),
* daily urine output,
* daily serum creatinine, and
* daily IV loop-diuretic dose in furosemide equivalents.

The ICU-relative windows are [intime + 24*d, intime + 24*(d+1)) for d=0,...,4.
No itemid integers are hard-coded: fluid/urine/loop concepts are resolved from
the MIMIC-IV dictionary labels.  Urine and creatinine remain missing when no
measurement is available.  Fluid and loop-dose values retain explicit event
availability flags so that a zero is not confused with an unobserved marker.

The script writes raw daily data, winsorized/z-scored data, long and wide
trajectory matrices, landmark outcomes, and QC tables.  It intentionally stops
before GBMT/GMM fitting; the resulting files are the inputs for the next R
phenotype-analysis step.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


LOOP_RE = r"(furosemide|lasix|bumetanide|bumex|torsemide|demadex|ethacrynic|edecrin)"
URINE_RE = r"(urine|foley|void|condom[[:space:]]*cath|ileoconduit|suprapubic|nephrostomy|straight[[:space:]]*cath|ureteral[[:space:]]*stent|incontinent)"
STATUS_EXCLUDE_RE = r"not started|not given|refus|hold|off schedule|cancel|discontinu"
VOLUME_UNIT_RE = r"(^|[^a-z])(ml|milliliter|milliliters|l|liter|liters)([^a-z]|$)"
MASS_UNIT_RE = r"(^|[^a-z])(mg|milligram|milligrams|mcg|ug|microgram|micrograms|g|gram|grams)([^a-z]|$)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mimic-dir", required=True, type=Path,
                   help="MIMIC-IV v2.2 root containing hosp/ and icu/")
    p.add_argument("--cohort", required=True, type=Path,
                   help="one-row-per-patient rebuilt cohort CSV (for example adhf_icu_cohort_v4.csv)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--winsor-lower", type=float, default=0.01,
                   help="lower pooled percentile used before scaling")
    p.add_argument("--winsor-upper", type=float, default=0.99,
                   help="upper pooled percentile used before scaling")
    return p.parse_args()


def sql_path(path: Path) -> str:
    return str(path.expanduser().resolve()).replace("'", "''")


def require_columns(con: duckdb.DuckDBPyConnection, table: str, required: list[str]) -> None:
    present = {row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()}
    missing = [col for col in required if col not in present]
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def qpercentile(values: pd.Series, probability: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return float("nan")
    return float(np.quantile(numeric, probability))


def main() -> None:
    a = parse_args()
    root = a.mimic_dir.expanduser().resolve()
    cohort_path = a.cohort.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    hosp = root / "hosp"
    icu = root / "icu"

    if not (0 <= a.winsor_lower < a.winsor_upper <= 1):
        raise ValueError("winsor percentiles must satisfy 0 <= lower < upper <= 1")

    required = [
        cohort_path,
        hosp / "admissions.csv.gz",
        hosp / "patients.csv.gz",
        hosp / "labevents.csv.gz",
        hosp / "d_labitems.csv.gz",
        icu / "inputevents.csv.gz",
        icu / "outputevents.csv.gz",
        icu / "d_items.csv.gz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    out.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(out / "trajectory_prepare.duckdb"))
    cp = sql_path(cohort_path)
    hp = lambda name: sql_path(hosp / name)
    ip = lambda name: sql_path(icu / name)

    # ------------------------------------------------------------------
    # 1. Cohort and ICU-relative day skeleton
    # ------------------------------------------------------------------
    con.execute(f"""
      CREATE OR REPLACE TABLE candidate AS
      SELECT *
      FROM read_csv_auto('{cp}', header=true, union_by_name=true,
                         null_padding=true, all_varchar=true)
    """)
    candidate_columns = {row[0] for row in con.execute("DESCRIBE candidate").fetchall()}
    require_columns(con, "candidate", [
        "subject_id", "hadm_id", "stay_id", "admittime", "dischtime",
        "intime", "outtime", "age_at_admission", "icu_los_hours",
    ])
    # Rebuilt cohort files can already contain parsed audit columns named
    # `icu_in`, `icu_out`, `adm`, `dis`, `age`, or `los_h`. If those string
    # columns are retained alongside a second alias with the same name,
    # DuckDB resolves expressions such as `c.icu_in + INTERVAL ...` to the
    # VARCHAR column. Exclude any pre-existing parsed names before creating
    # one canonical, typed version of each field.
    parsed_names = [
        "subject_id_num", "hadm_id_num", "stay_id_num", "adm", "dis",
        "icu_in", "icu_out", "age", "los_h",
    ]
    exclude_names = [name for name in parsed_names if name in candidate_columns]
    star_sql = "*"
    if exclude_names:
        star_sql += " EXCLUDE (" + ", ".join(exclude_names) + ")"
    con.execute(f"""
      CREATE OR REPLACE TABLE c AS
      SELECT {star_sql},
        try_cast(subject_id AS BIGINT) AS subject_id_num,
        try_cast(hadm_id AS BIGINT) AS hadm_id_num,
        try_cast(stay_id AS BIGINT) AS stay_id_num,
        try_cast(admittime AS TIMESTAMP) AS adm,
        try_cast(dischtime AS TIMESTAMP) AS dis,
        try_cast(intime AS TIMESTAMP) AS icu_in,
        try_cast(outtime AS TIMESTAMP) AS icu_out,
        try_cast(age_at_admission AS DOUBLE) AS age,
        try_cast(icu_los_hours AS DOUBLE) AS los_h
      FROM candidate
    """)
    cohort_n = con.execute("SELECT count(*) FROM c").fetchone()[0]
    duplicate_stays = con.execute("""
      SELECT count(*) FROM (
        SELECT stay_id_num FROM c GROUP BY stay_id_num HAVING count(*) > 1
      )
    """).fetchone()[0]
    if duplicate_stays:
        raise ValueError(f"cohort must contain one row per stay_id; duplicated stay groups: {duplicate_stays}")
    invalid_time_n = con.execute("""
      SELECT count(*) FROM c
      WHERE subject_id_num IS NULL OR hadm_id_num IS NULL OR stay_id_num IS NULL
         OR icu_in IS NULL OR icu_out IS NULL OR icu_out <= icu_in
    """).fetchone()[0]
    if invalid_time_n:
        raise ValueError(f"cohort contains {invalid_time_n} rows with invalid identifiers or ICU times")

    con.execute("""
      CREATE OR REPLACE TABLE icu_days AS
      SELECT c.subject_id_num AS subject_id,
             c.hadm_id_num AS hadm_id,
             c.stay_id_num AS stay_id,
             r.day_num,
             c.icu_in + r.day_num * INTERVAL '24 hours' AS day_start,
             c.icu_in + (r.day_num + 1) * INTERVAL '24 hours' AS day_end,
             c.icu_in,
             c.icu_out,
             greatest(
               0,
               date_diff(
                 'second',
                 c.icu_in + r.day_num * INTERVAL '24 hours',
                 least(c.icu_in + (r.day_num + 1) * INTERVAL '24 hours', c.icu_out)
               )
             ) / 3600.0 AS icu_hours_in_day,
             (
               c.icu_in + r.day_num * INTERVAL '24 hours' < c.icu_out
             ) AS day_in_icu
      FROM c
      CROSS JOIN (SELECT range AS day_num FROM range(0, 5)) r
    """)

    # ------------------------------------------------------------------
    # 2. Dictionary and raw event extracts
    # ------------------------------------------------------------------
    con.execute(f"""
      CREATE OR REPLACE TABLE d_items_norm AS
      SELECT try_cast(itemid AS BIGINT) AS itemid,
             lower(trim(coalesce(CAST(label AS VARCHAR), ''))) AS item_label
      FROM read_csv_auto('{ip('d_items.csv.gz')}', header=true,
                         union_by_name=true, null_padding=true, all_varchar=true)
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE input_raw AS
      SELECT try_cast(i.subject_id AS BIGINT) AS subject_id,
             try_cast(i.hadm_id AS BIGINT) AS hadm_id,
             try_cast(i.stay_id AS BIGINT) AS stay_id,
             try_cast(i.starttime AS TIMESTAMP) AS starttime,
             try_cast(i.endtime AS TIMESTAMP) AS endtime,
             try_cast(i.itemid AS BIGINT) AS itemid,
             d.item_label,
             try_cast(i.amount AS DOUBLE) AS amount,
             lower(trim(coalesce(CAST(i.amountuom AS VARCHAR), ''))) AS amount_uom,
             try_cast(i.rate AS DOUBLE) AS rate,
             lower(trim(coalesce(CAST(i.rateuom AS VARCHAR), ''))) AS rate_uom,
             lower(trim(coalesce(CAST(i.statusdescription AS VARCHAR), ''))) AS statusdescription
      FROM read_csv_auto('{ip('inputevents.csv.gz')}', header=true,
                         union_by_name=true, null_padding=true, all_varchar=true) i
      INNER JOIN (SELECT DISTINCT subject_id_num AS subject_id,
                                  hadm_id_num AS hadm_id,
                                  stay_id_num AS stay_id FROM c) cc
        ON try_cast(i.subject_id AS BIGINT)=cc.subject_id
       AND try_cast(i.hadm_id AS BIGINT)=cc.hadm_id
       AND try_cast(i.stay_id AS BIGINT)=cc.stay_id
      LEFT JOIN d_items_norm d
        ON try_cast(i.itemid AS BIGINT)=d.itemid
      WHERE try_cast(i.starttime AS TIMESTAMP) IS NOT NULL
        AND NOT regexp_matches(lower(trim(coalesce(CAST(i.statusdescription AS VARCHAR), ''))), '{STATUS_EXCLUDE_RE}')
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE output_raw AS
      SELECT try_cast(o.subject_id AS BIGINT) AS subject_id,
             try_cast(o.hadm_id AS BIGINT) AS hadm_id,
             try_cast(o.stay_id AS BIGINT) AS stay_id,
             try_cast(o.charttime AS TIMESTAMP) AS charttime,
             try_cast(o.itemid AS BIGINT) AS itemid,
             d.item_label,
             try_cast(o.value AS DOUBLE) AS value,
             lower(trim(coalesce(CAST(o.valueuom AS VARCHAR), ''))) AS value_uom
      FROM read_csv_auto('{ip('outputevents.csv.gz')}', header=true,
                         union_by_name=true, null_padding=true, all_varchar=true) o
      INNER JOIN (SELECT DISTINCT subject_id_num AS subject_id,
                                  hadm_id_num AS hadm_id,
                                  stay_id_num AS stay_id FROM c) cc
        ON try_cast(o.subject_id AS BIGINT)=cc.subject_id
       AND try_cast(o.hadm_id AS BIGINT)=cc.hadm_id
       AND try_cast(o.stay_id AS BIGINT)=cc.stay_id
      LEFT JOIN d_items_norm d
        ON try_cast(o.itemid AS BIGINT)=d.itemid
      WHERE try_cast(o.charttime AS TIMESTAMP) IS NOT NULL
    """)

    # ------------------------------------------------------------------
    # 3. Daily input/output fluid volumes with interval splitting
    # ------------------------------------------------------------------
    con.execute(f"""
      CREATE OR REPLACE TABLE input_fluid_alloc AS
      WITH x AS (
        SELECT i.*,
          CASE
            WHEN regexp_matches(i.amount_uom, '{VOLUME_UNIT_RE}')
              AND regexp_matches(i.amount_uom, '(^|[^a-z])(ml|milliliter|milliliters)([^a-z]|$)')
              THEN i.amount
            WHEN regexp_matches(i.amount_uom, '(^|[^a-z])(l|liter|liters)([^a-z]|$)')
              THEN i.amount * 1000.0
            ELSE NULL
          END AS amount_ml,
          CASE
            WHEN regexp_matches(i.rate_uom, '(^|[^a-z])ml(/|\\s*)h|milliliter.*hour')
              THEN i.rate
            WHEN regexp_matches(i.rate_uom, '(^|[^a-z])l(/|\\s*)h|liter.*hour')
              THEN i.rate * 1000.0
            ELSE NULL
          END AS rate_ml_h
        FROM input_raw i
      ), overlap AS (
        SELECT x.*, d.day_num, d.day_start, d.day_end, d.icu_out,
          (
            regexp_matches(x.amount_uom, '{VOLUME_UNIT_RE}')
            OR regexp_matches(x.rate_uom, '(^|[^a-z])(ml|milliliter|milliliters|l|liter|liters)([^a-z]|$)')
          ) AS is_fluid_candidate,
          (x.endtime IS NOT NULL AND x.endtime > x.starttime) AS valid_interval,
          greatest(0, date_diff('second', greatest(x.starttime, d.day_start),
                                  least(x.endtime, d.day_end, d.icu_out))) / 3600.0 AS overlap_hours
        FROM x
        INNER JOIN icu_days d
          ON x.subject_id=d.subject_id AND x.hadm_id=d.hadm_id AND x.stay_id=d.stay_id
         AND d.day_start < d.icu_out
         AND (
              (x.endtime IS NOT NULL AND x.endtime > x.starttime
               AND x.starttime < least(d.day_end, d.icu_out) AND x.endtime > d.day_start)
              OR ((x.endtime IS NULL OR x.endtime <= x.starttime)
                  AND x.starttime >= d.day_start AND x.starttime < least(d.day_end, d.icu_out))
         )
      )
      SELECT subject_id,hadm_id,stay_id,day_num,itemid,item_label,starttime,endtime,
        amount_uom,rate_uom,amount_ml,rate_ml_h,is_fluid_candidate,
        CASE
          WHEN valid_interval AND amount_ml IS NOT NULL
            THEN amount_ml * overlap_hours / NULLIF(date_diff('second', starttime, endtime) / 3600.0, 0)
          WHEN valid_interval AND amount_ml IS NULL AND rate_ml_h IS NOT NULL
            THEN rate_ml_h * overlap_hours
          WHEN NOT valid_interval AND starttime >= day_start AND starttime < least(day_end, icu_out)
            THEN amount_ml
          ELSE NULL
        END AS allocated_input_ml
      FROM overlap
    """)
    con.execute("""
      CREATE OR REPLACE TABLE input_daily AS
      SELECT subject_id,hadm_id,stay_id,day_num,
             sum(allocated_input_ml) AS input_volume_ml,
             count(*) AS input_event_n,
             count(allocated_input_ml) AS input_quantified_event_n,
             count(*) FILTER (WHERE allocated_input_ml IS NULL) AS input_unquantified_event_n
      FROM input_fluid_alloc
      WHERE is_fluid_candidate
      GROUP BY subject_id,hadm_id,stay_id,day_num
    """)
    con.execute("""
      CREATE OR REPLACE TABLE input_presence_daily AS
      SELECT i.subject_id,i.hadm_id,i.stay_id,d.day_num,
             count(*) AS input_presence_event_n
      FROM input_raw i
      INNER JOIN icu_days d
        ON i.subject_id=d.subject_id AND i.hadm_id=d.hadm_id AND i.stay_id=d.stay_id
       AND d.day_start < d.icu_out
       AND (
            (i.endtime IS NOT NULL AND i.endtime > i.starttime
             AND i.starttime < least(d.day_end, d.icu_out) AND i.endtime > d.day_start)
            OR ((i.endtime IS NULL OR i.endtime <= i.starttime)
                AND i.starttime >= d.day_start AND i.starttime < least(d.day_end, d.icu_out))
       )
      GROUP BY i.subject_id,i.hadm_id,i.stay_id,d.day_num
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE output_daily AS
      WITH x AS (
        SELECT *,
          CASE
            WHEN regexp_matches(value_uom, '(^|[^a-z])(ml|milliliter|milliliters)([^a-z]|$)')
              THEN value
            WHEN regexp_matches(value_uom, '(^|[^a-z])(l|liter|liters)([^a-z]|$)')
              THEN value * 1000.0
            ELSE NULL
          END AS value_ml
        FROM output_raw
      )
      SELECT x.subject_id,x.hadm_id,x.stay_id,d.day_num,
             sum(x.value_ml) AS output_volume_ml,
             count(*) AS output_event_n,
             count(x.value_ml) AS output_quantified_event_n
      FROM x
      INNER JOIN icu_days d
        ON x.subject_id=d.subject_id AND x.hadm_id=d.hadm_id AND x.stay_id=d.stay_id
       AND d.day_start < d.icu_out
       AND x.charttime >= d.day_start AND x.charttime < least(d.day_end, d.icu_out)
      GROUP BY x.subject_id,x.hadm_id,x.stay_id,d.day_num
    """)
    con.execute(f"""
      CREATE OR REPLACE TABLE urine_daily AS
      WITH x AS (
        SELECT *,
          CASE
            WHEN regexp_matches(value_uom, '(^|[^a-z])(ml|milliliter|milliliters)([^a-z]|$)')
              THEN value
            WHEN regexp_matches(value_uom, '(^|[^a-z])(l|liter|liters)([^a-z]|$)')
              THEN value * 1000.0
            ELSE NULL
          END AS value_ml
        FROM output_raw
        WHERE regexp_matches(item_label, '{URINE_RE}')
      )
      SELECT x.subject_id,x.hadm_id,x.stay_id,d.day_num,
             sum(x.value_ml) AS urine_output_ml,
             count(*) AS urine_event_n,
             count(x.value_ml) AS urine_quantified_event_n
      FROM x
      INNER JOIN icu_days d
        ON x.subject_id=d.subject_id AND x.hadm_id=d.hadm_id AND x.stay_id=d.stay_id
       AND d.day_start < d.icu_out
       AND x.charttime >= d.day_start AND x.charttime < least(d.day_end, d.icu_out)
      GROUP BY x.subject_id,x.hadm_id,x.stay_id,d.day_num
    """)

    # ------------------------------------------------------------------
    # 4. Daily creatinine: last measured value in each ICU-relative day
    # ------------------------------------------------------------------
    con.execute(f"""
      CREATE OR REPLACE TABLE creatinine_daily AS
      WITH labs AS (
        SELECT try_cast(l.subject_id AS BIGINT) AS subject_id,
               try_cast(l.hadm_id AS BIGINT) AS hadm_id,
               try_cast(l.charttime AS TIMESTAMP) AS charttime,
               try_cast(l.valuenum AS DOUBLE) AS creatinine_mg_dl,
               lower(coalesce(CAST(dl.label AS VARCHAR), '')) AS lab_label
        FROM read_csv_auto('{hp('labevents.csv.gz')}', header=true,
                           union_by_name=true, null_padding=true, all_varchar=true) l
        INNER JOIN (SELECT DISTINCT subject_id_num AS subject_id,
                                    hadm_id_num AS hadm_id FROM c) cc
          ON try_cast(l.subject_id AS BIGINT)=cc.subject_id
         AND try_cast(l.hadm_id AS BIGINT)=cc.hadm_id
        LEFT JOIN read_csv_auto('{hp('d_labitems.csv.gz')}', header=true,
                                union_by_name=true, null_padding=true, all_varchar=true) dl
          ON try_cast(l.itemid AS BIGINT)=try_cast(dl.itemid AS BIGINT)
        WHERE try_cast(l.charttime AS TIMESTAMP) IS NOT NULL
          AND try_cast(l.valuenum AS DOUBLE) IS NOT NULL
          AND lower(coalesce(CAST(dl.label AS VARCHAR), '')) = 'creatinine'
      ), positioned AS (
        SELECT labs.*, d.stay_id, d.day_num,
          row_number() OVER (
            PARTITION BY d.stay_id,d.day_num
            ORDER BY labs.charttime DESC
          ) AS rn
        FROM labs
        INNER JOIN icu_days d
          ON labs.subject_id=d.subject_id AND labs.hadm_id=d.hadm_id
         AND d.day_start < d.icu_out
         AND labs.charttime >= d.day_start AND labs.charttime < least(d.day_end, d.icu_out)
      )
      SELECT subject_id,hadm_id,stay_id,day_num,charttime AS creatinine_time,
             creatinine_mg_dl
      FROM positioned
      WHERE rn=1
    """)

    # ------------------------------------------------------------------
    # 5. Daily IV loop dose in furosemide equivalents
    # ------------------------------------------------------------------
    con.execute(f"""
      CREATE OR REPLACE TABLE loop_dose_alloc AS
      WITH x AS (
        SELECT i.*,
          CASE
            WHEN regexp_matches(item_label, 'bumetanide|bumex') THEN 40.0
            WHEN regexp_matches(item_label, 'torsemide|demadex') THEN 2.0
            WHEN regexp_matches(item_label, 'furosemide|lasix|ethacrynic|edecrin') THEN 1.0
            ELSE NULL
          END AS eq_factor,
          CASE
            WHEN regexp_matches(amount_uom, '(^|[^a-z])(mcg|ug|microgram|micrograms)([^a-z]|$)')
              THEN amount / 1000.0
            WHEN regexp_matches(amount_uom, '(^|[^a-z])(g|gram|grams)([^a-z]|$)')
              THEN amount * 1000.0
            WHEN regexp_matches(amount_uom, '(^|[^a-z])(mg|milligram|milligrams)([^a-z]|$)')
              THEN amount
            ELSE NULL
          END AS amount_mg,
          CASE
            WHEN regexp_matches(rate_uom, '(^|[^a-z])(mcg|ug|microgram|micrograms).*(/|per|\\s)h')
              THEN rate / 1000.0
            WHEN regexp_matches(rate_uom, '(^|[^a-z])(g|gram|grams).*(/|per|\\s)h')
              THEN rate * 1000.0
            WHEN regexp_matches(rate_uom, '(^|[^a-z])(mg|milligram|milligrams).*(/|per|\\s)h')
              THEN rate
            ELSE NULL
          END AS rate_mg_h
        FROM input_raw i
        WHERE regexp_matches(item_label, '{LOOP_RE}')
      ), overlap AS (
        SELECT x.*, d.day_num,d.day_start,d.day_end,d.icu_out,
          (x.endtime IS NOT NULL AND x.endtime > x.starttime) AS valid_interval,
          greatest(0, date_diff('second', greatest(x.starttime, d.day_start),
                                  least(x.endtime, d.day_end, d.icu_out))) / 3600.0 AS overlap_hours
        FROM x
        INNER JOIN icu_days d
          ON x.subject_id=d.subject_id AND x.hadm_id=d.hadm_id AND x.stay_id=d.stay_id
         AND d.day_start < d.icu_out
         AND (
              (x.endtime IS NOT NULL AND x.endtime > x.starttime
               AND x.starttime < least(d.day_end, d.icu_out) AND x.endtime > d.day_start)
              OR ((x.endtime IS NULL OR x.endtime <= x.starttime)
                  AND x.starttime >= d.day_start AND x.starttime < least(d.day_end, d.icu_out))
         )
      )
      SELECT subject_id,hadm_id,stay_id,day_num,itemid,item_label,starttime,endtime,
        amount_uom,rate_uom,amount_mg,rate_mg_h,
        CASE
          WHEN eq_factor IS NULL THEN NULL
          WHEN valid_interval AND rate_mg_h IS NOT NULL
            THEN rate_mg_h * overlap_hours * eq_factor
          WHEN NOT valid_interval AND starttime >= day_start AND starttime < least(day_end, icu_out)
               AND amount_mg IS NOT NULL
            THEN amount_mg * eq_factor
          WHEN valid_interval AND amount_mg IS NOT NULL AND rate_mg_h IS NULL
            THEN amount_mg * overlap_hours /
                 NULLIF(date_diff('second', starttime, endtime) / 3600.0, 0) * eq_factor
          ELSE NULL
        END AS dose_mg_eq
      FROM overlap
    """)
    con.execute("""
      CREATE OR REPLACE TABLE loop_dose_daily AS
      SELECT subject_id,hadm_id,stay_id,day_num,
             sum(dose_mg_eq) AS iv_loop_dose_mg_eq,
             count(*) AS loop_event_n,
             count(dose_mg_eq) AS loop_quantified_event_n,
             min(starttime) AS first_loop_time
      FROM loop_dose_alloc
      GROUP BY subject_id,hadm_id,stay_id,day_num
    """)

    # ------------------------------------------------------------------
    # 6. Assemble five rows per stay and calculate the four markers
    # ------------------------------------------------------------------
    days = con.execute("SELECT * FROM icu_days ORDER BY stay_id,day_num").df()
    keys = ["subject_id", "hadm_id", "stay_id", "day_num"]
    for name in ["input_daily", "input_presence_daily", "output_daily", "urine_daily",
                 "creatinine_daily", "loop_dose_daily"]:
        frame = con.execute(f"SELECT * FROM {name}").df()
        days = days.merge(frame, on=keys, how="left", validate="one_to_one")

    for col in ["input_volume_ml", "output_volume_ml", "urine_output_ml",
                "creatinine_mg_dl", "iv_loop_dose_mg_eq"]:
        days[col] = pd.to_numeric(days[col], errors="coerce")
    for col in ["input_event_n", "input_quantified_event_n", "input_unquantified_event_n",
                "input_presence_event_n", "output_event_n",
                "output_quantified_event_n", "urine_event_n", "urine_quantified_event_n",
                "loop_event_n", "loop_quantified_event_n"]:
        if col not in days:
            days[col] = 0
        days[col] = pd.to_numeric(days[col], errors="coerce").fillna(0).astype(int)

    days["fluid_balance_observed"] = (
        (days["input_event_n"] + days["output_event_n"]) > 0
    ).astype(int)
    days["fluid_balance_complete"] = (
        (days["fluid_balance_observed"] == 1)
        & (days["input_event_n"] == days["input_quantified_event_n"])
        & (days["output_event_n"] == days["output_quantified_event_n"])
    ).astype(int)
    days["daily_net_fluid_balance_ml"] = np.where(
        days["fluid_balance_complete"] == 1,
        days["input_volume_ml"].fillna(0.0) - days["output_volume_ml"].fillna(0.0),
        np.nan,
    )
    days = days.sort_values(["stay_id", "day_num"]).reset_index(drop=True)
    # Cumulative balance is only valid while every preceding ICU day has a
    # complete fluid balance. This prevents an unobserved day from being
    # silently treated as zero and carried into later cumulative values.
    running_balance = (
        days["daily_net_fluid_balance_ml"].fillna(0.0)
        .groupby(days["stay_id"])
        .cumsum()
    )
    complete_through_day = (
        days["fluid_balance_complete"].astype(bool)
        .groupby(days["stay_id"])
        .cummin()
    )
    days["cumulative_net_fluid_balance_ml"] = running_balance.where(
        complete_through_day, np.nan
    )
    # A no-loop day is a measured zero only when there is at least one other
    # actual inputevent in that ICU day. Days outside the ICU or with no input
    # record remain unobserved; a loop event with an unquantifiable amount also
    # remains missing and is retained in the audit table.
    days["loop_dose_observed"] = (
        ((days["loop_event_n"] > 0) & days["iv_loop_dose_mg_eq"].notna())
        | ((days["loop_event_n"] == 0) & (days["input_presence_event_n"] > 0))
    ).astype(int)
    days.loc[
        (days["loop_event_n"] == 0) & (days["input_presence_event_n"] > 0),
        "iv_loop_dose_mg_eq",
    ] = 0.0
    days["loop_dose_complete"] = (
        (days["loop_event_n"] == 0)
        | (days["loop_event_n"] == days["loop_quantified_event_n"])
    ).astype(int)
    days["n_markers_observed"] = (
        days["cumulative_net_fluid_balance_ml"].notna().astype(int)
        + days["urine_output_ml"].notna().astype(int)
        + days["creatinine_mg_dl"].notna().astype(int)
        + ((days["loop_dose_observed"] == 1)
           & days["iv_loop_dose_mg_eq"].notna()).astype(int)
    )
    days["any_marker_observed"] = (days["n_markers_observed"] > 0).astype(int)

    marker_cols = [
        "cumulative_net_fluid_balance_ml",
        "urine_output_ml",
        "creatinine_mg_dl",
        "iv_loop_dose_mg_eq",
    ]

    # ------------------------------------------------------------------
    # 7. QC and day-5 landmark outcomes
    # ------------------------------------------------------------------
    completeness = (
        days.groupby(["subject_id", "hadm_id", "stay_id"], as_index=False)
        .agg(
            n_days_with_any_marker=("any_marker_observed", "sum"),
            n_days_fluid=("fluid_balance_observed", "sum"),
            n_days_fluid_complete=("fluid_balance_complete", "sum"),
            n_days_urine=("urine_output_ml", lambda x: int(x.notna().sum())),
            n_days_creatinine=("creatinine_mg_dl", lambda x: int(x.notna().sum())),
            n_days_loop=("loop_event_n", lambda x: int((x > 0).sum())),
            n_days_loop_observed=("loop_dose_observed", "sum"),
            n_days_loop_dose_incomplete=("loop_dose_complete", lambda x: int((x == 0).sum())),
        )
    )
    completeness["meets_two_day_marker_rule"] = (
        completeness["n_days_with_any_marker"] >= 2
    ).astype(int)
    days = days.merge(completeness, on=["subject_id", "hadm_id", "stay_id"], how="left")

    # `dod` is a patient-level field in MIMIC-IV (patients.csv.gz), not an
    # admissions.csv.gz field. Read the two files separately and merge them;
    # dynamic column checks keep the script compatible with minor schema
    # revisions while still failing clearly if hadm_id is absent.
    con.execute(f"""
      CREATE OR REPLACE TABLE admissions_raw AS
      SELECT *
      FROM read_csv_auto('{hp('admissions.csv.gz')}', header=true,
                         union_by_name=true, null_padding=true, all_varchar=true)
    """)
    admission_columns = {row[0] for row in con.execute("DESCRIBE admissions_raw").fetchall()}
    if "hadm_id" not in admission_columns:
        raise ValueError("admissions.csv.gz is missing hadm_id")

    def optional_cast(column: str, sql_type: str) -> str:
        if column not in admission_columns:
            return "NULL"
        return f"try_cast({column} AS {sql_type})"

    admissions = con.execute(f"""
      SELECT {optional_cast('hadm_id', 'BIGINT')} AS hadm_id,
             {optional_cast('admittime', 'TIMESTAMP')} AS hosp_admittime,
             {optional_cast('dischtime', 'TIMESTAMP')} AS hosp_dischtime,
             {optional_cast('deathtime', 'TIMESTAMP')} AS deathtime,
             {optional_cast('hospital_expire_flag', 'INTEGER')} AS hospital_expire_flag
      FROM admissions_raw
    """).df()
    patients_path = hosp / "patients.csv.gz"
    con.execute(f"""
      CREATE OR REPLACE TABLE patients_raw AS
      SELECT *
      FROM read_csv_auto('{sql_path(patients_path)}', header=true,
                         union_by_name=true, null_padding=true, all_varchar=true)
    """)
    patient_columns = {row[0] for row in con.execute("DESCRIBE patients_raw").fetchall()}
    if "subject_id" in patient_columns and "dod" in patient_columns:
        patients = con.execute("""
          SELECT try_cast(subject_id AS BIGINT) AS subject_id,
                 try_cast(dod AS DATE) AS dod
          FROM patients_raw
        """).df()
    else:
        patients = pd.DataFrame(columns=["subject_id", "dod"])
    stays = con.execute("""
      SELECT subject_id_num AS subject_id, hadm_id_num AS hadm_id,
             stay_id_num AS stay_id, icu_in, dis, icu_out
      FROM c
    """).df()
    outcomes = stays.merge(admissions, on="hadm_id", how="left")
    outcomes = outcomes.merge(patients, on="subject_id", how="left")
    for time_col in ["icu_in", "icu_out", "dis", "hosp_admittime",
                     "hosp_dischtime", "deathtime", "dod"]:
        if time_col in outcomes:
            outcomes[time_col] = pd.to_datetime(outcomes[time_col], errors="coerce")
    outcomes["day5_time"] = outcomes["icu_in"] + pd.to_timedelta(120, unit="h")
    outcomes["day90_time"] = outcomes["icu_in"] + pd.to_timedelta(90, unit="d")
    outcomes["death_before_day5"] = (
        outcomes["deathtime"].notna() & (outcomes["deathtime"] < outcomes["day5_time"])
    )
    dod_before_day5 = (
        outcomes["dod"].notna()
        & (outcomes["dod"] < outcomes["day5_time"].dt.normalize())
    )
    outcomes["death_before_day5"] = (
        outcomes["death_before_day5"] | dod_before_day5
    ).astype(int)
    outcomes["day5_landmark_eligible"] = (
        (outcomes["death_before_day5"] == 0)
        & outcomes["dis"].notna()
        & (outcomes["dis"] >= outcomes["day5_time"])
    ).astype(int)
    death_by_90 = (
        (outcomes["deathtime"].notna() & (outcomes["deathtime"] <= outcomes["day90_time"]))
        | (outcomes["dod"].notna() & (outcomes["dod"] <= outcomes["day90_time"].dt.normalize()))
    )
    outcomes["mortality_90d"] = death_by_90.astype(int)
    outcomes["in_hospital_mortality"] = (
        (outcomes["hospital_expire_flag"].fillna(0) == 1)
        | (
            outcomes["deathtime"].notna()
            & outcomes["hosp_dischtime"].notna()
            & (outcomes["deathtime"] <= outcomes["hosp_dischtime"])
        )
    ).astype(int)
    outcomes["death_time_for_analysis"] = outcomes["deathtime"]
    outcomes.loc[
        outcomes["death_time_for_analysis"].isna(), "death_time_for_analysis"
    ] = pd.to_datetime(outcomes.loc[
        outcomes["death_time_for_analysis"].isna(), "dod"
    ], errors="coerce")
    outcomes["time_to_death_or_90d_days"] = np.where(
        outcomes["death_time_for_analysis"].notna(),
        (outcomes["death_time_for_analysis"] - outcomes["icu_in"]).dt.total_seconds() / 86400.0,
        90.0,
    )
    outcomes["time_to_death_or_90d_days"] = outcomes["time_to_death_or_90d_days"].clip(lower=0, upper=90)

    # ------------------------------------------------------------------
    # 8. Winsorization and marker-specific z-scores
    # ------------------------------------------------------------------
    scale_rows: list[dict] = []
    for marker in marker_cols:
        values = pd.to_numeric(days[marker], errors="coerce")
        low = qpercentile(values, a.winsor_lower)
        high = qpercentile(values, a.winsor_upper)
        clipped = values.clip(lower=low, upper=high) if np.isfinite(low) else values
        mean = float(clipped.mean()) if clipped.notna().any() else float("nan")
        sd = float(clipped.std(ddof=0)) if clipped.notna().any() else float("nan")
        scaled_name = {
            "cumulative_net_fluid_balance_ml": "cumulative_net_fluid_balance_z",
            "urine_output_ml": "urine_output_z",
            "creatinine_mg_dl": "creatinine_z",
            "iv_loop_dose_mg_eq": "iv_loop_dose_z",
        }[marker]
        clipped_name = marker + "_winsorized"
        days[clipped_name] = clipped
        if np.isfinite(sd) and sd > 0:
            days[scaled_name] = (clipped - mean) / sd
        elif clipped.notna().any():
            days[scaled_name] = clipped * 0.0
        else:
            days[scaled_name] = np.nan
        scale_rows.append({
            "marker": marker,
            "n_observed": int(values.notna().sum()),
            "winsor_lower_probability": a.winsor_lower,
            "winsor_upper_probability": a.winsor_upper,
            "winsor_lower_value": low,
            "winsor_upper_value": high,
            "winsorized_mean": mean,
            "winsorized_sd": sd,
        })
    scaling = pd.DataFrame(scale_rows)

    # ------------------------------------------------------------------
    # 9. Write auditable files
    # ------------------------------------------------------------------
    raw_cols = [
        "subject_id", "hadm_id", "stay_id", "day_num", "day_start", "day_end",
        "icu_hours_in_day", "day_in_icu", "fluid_balance_observed",
        "fluid_balance_complete", "input_volume_ml", "output_volume_ml",
        "daily_net_fluid_balance_ml", "cumulative_net_fluid_balance_ml",
        "input_presence_event_n", "input_event_n", "input_quantified_event_n",
        "input_unquantified_event_n", "output_event_n",
        "output_quantified_event_n", "urine_output_ml", "urine_event_n",
        "urine_quantified_event_n", "creatinine_mg_dl", "creatinine_time",
        "iv_loop_dose_mg_eq", "loop_event_n", "loop_quantified_event_n",
        "loop_dose_observed", "loop_dose_complete", "first_loop_time", "n_markers_observed",
        "any_marker_observed", "n_days_with_any_marker", "n_days_fluid",
        "n_days_urine", "n_days_creatinine", "n_days_loop",
        "n_days_loop_dose_incomplete", "meets_two_day_marker_rule",
    ]
    days[raw_cols].to_csv(out / "trajectory_daily_raw_v1.csv", index=False)
    scaled_cols = raw_cols + [
        "cumulative_net_fluid_balance_ml_winsorized",
        "urine_output_ml_winsorized", "creatinine_mg_dl_winsorized",
        "iv_loop_dose_mg_eq_winsorized", "cumulative_net_fluid_balance_z",
        "urine_output_z", "creatinine_z", "iv_loop_dose_z",
    ]
    days[scaled_cols].to_csv(out / "trajectory_daily_scaled_v1.csv", index=False)
    scaling.to_csv(out / "trajectory_scaling_parameters_v1.csv", index=False)
    completeness.to_csv(out / "trajectory_completeness_v1.csv", index=False)
    outcomes.to_csv(out / "landmark_outcomes_v1.csv", index=False)

    long = days[["subject_id", "hadm_id", "stay_id", "day_num"] + [
        "cumulative_net_fluid_balance_z", "urine_output_z", "creatinine_z", "iv_loop_dose_z"
    ]].melt(
        id_vars=["subject_id", "hadm_id", "stay_id", "day_num"],
        var_name="marker", value_name="value_z",
    )
    long.to_csv(out / "trajectory_long_scaled_v1.csv", index=False)

    matrix = days[["subject_id", "hadm_id", "stay_id", "day_num"] + [
        "cumulative_net_fluid_balance_z", "urine_output_z", "creatinine_z", "iv_loop_dose_z"
    ]].copy()
    matrix = matrix.pivot_table(
        index=["subject_id", "hadm_id", "stay_id"],
        columns="day_num", values=[
            "cumulative_net_fluid_balance_z", "urine_output_z", "creatinine_z", "iv_loop_dose_z"
        ], aggfunc="first",
    )
    matrix.columns = [f"{marker}_d{day}" for marker, day in matrix.columns]
    matrix = matrix.reset_index()
    matrix.to_csv(out / "trajectory_matrix_scaled_v1.csv", index=False)

    # Marker-level and day-level missingness/outlier QC.
    qc_rows: list[dict] = []
    for day_num in range(5):
        subset = days[days["day_num"] == day_num]
        for marker in marker_cols:
            qc_rows.append({
                "day_num": day_num,
                "marker": marker,
                "n_rows": int(len(subset)),
                "n_observed": int(subset[marker].notna().sum()),
                "n_missing": int(subset[marker].isna().sum()),
                "missing_percent": float(100 * subset[marker].isna().mean()),
                "median": float(subset[marker].median()) if subset[marker].notna().any() else np.nan,
                "p01": qpercentile(subset[marker], 0.01),
                "p99": qpercentile(subset[marker], 0.99),
            })
    pd.DataFrame(qc_rows).to_csv(out / "trajectory_marker_qc_v1.csv", index=False)

    unquantified = days[(days["loop_event_n"] > 0) & (days["loop_dose_complete"] == 0)][
        ["subject_id", "hadm_id", "stay_id", "day_num", "loop_event_n", "loop_quantified_event_n", "first_loop_time"]
    ]
    unquantified.to_csv(out / "loop_dose_unquantified_days_v1.csv", index=False)

    fluid_unquantified = con.execute("""
      SELECT subject_id,hadm_id,stay_id,day_num,itemid,item_label,starttime,endtime,
             amount_uom,rate_uom,amount_ml,rate_ml_h,allocated_input_ml
      FROM input_fluid_alloc
      WHERE is_fluid_candidate AND allocated_input_ml IS NULL
      ORDER BY stay_id,day_num,starttime,itemid
    """).df()
    fluid_unquantified.to_csv(out / "fluid_input_unquantified_events_v1.csv", index=False)

    trajectory_recheck = completeness.copy()
    trajectory_recheck["cohort_trajectory_rule_pass"] = (
        trajectory_recheck["meets_two_day_marker_rule"] == 1
    ).astype(int)
    trajectory_recheck.to_csv(out / "trajectory_cohort_recheck_v1.csv", index=False)

    # A compact run manifest makes the cohort/trajectory boundary explicit.
    meta = {
        "script": "prepare_adhf_trajectory_v1.py",
        "mimic_version": "2.2",
        "cohort_path": str(cohort_path),
        "cohort_rows": int(cohort_n),
        "cohort_boundary": "input cohort is assumed to have already passed rebuild_adhf_cohort_v3.py; this script does not reapply eligibility or exclusion rules",
        "trajectory_days": [0, 1, 2, 3, 4],
        "day_definition": "[ICU intime + 24*d, ICU intime + 24*(d+1))",
        "primary_markers": marker_cols,
        "urine_rule": "outputevents dictionary label matches urine/foley/void/condom-cath/ileoconduit/suprapubic/nephrostomy/straight-cath/ureteral-stent/incontinent concepts; missing daily urine remains missing",
        "creatinine_rule": "dictionary label equals creatinine; last measurement in each ICU-relative day",
        "fluid_rule": "recognized mL/L input and output volumes only; interval records split by overlap; unquantified units audited",
        "loop_rule": "actual ICU inputevents matching loop names; dose converted to furosemide equivalents",
        "equivalence": {
            "furosemide": 1,
            "bumetanide": 40,
            "torsemide": 2,
            "ethacrynic_acid": 1,
        },
        "missingness_rule": "urine and creatinine are not recoded to zero; fluid cumulative balance is missing until each preceding day is complete; loop no-event days are exposure zero only when another actual inputevent documents that ICU day",
        "scaling_rule": f"pooled marker-specific {a.winsor_lower:.4f}/{a.winsor_upper:.4f} winsorization followed by z-score",
        "landmark_rule": "day-5 landmark requires survival and remaining in hospital at ICU intime + 120h",
        "outputs": [
            "trajectory_daily_raw_v1.csv",
            "trajectory_daily_scaled_v1.csv",
            "trajectory_long_scaled_v1.csv",
            "trajectory_matrix_scaled_v1.csv",
            "trajectory_scaling_parameters_v1.csv",
            "trajectory_completeness_v1.csv",
            "trajectory_marker_qc_v1.csv",
            "trajectory_cohort_recheck_v1.csv",
            "fluid_input_unquantified_events_v1.csv",
            "loop_dose_unquantified_days_v1.csv",
            "landmark_outcomes_v1.csv",
        ],
    }
    (out / "RUN_METADATA.json").write_text(json.dumps(meta, indent=2) + "\n")
    con.close()
    print(json.dumps(meta, indent=2))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
