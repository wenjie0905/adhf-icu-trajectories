#!/usr/bin/env python3
"""Rebuild the ADHF-ICU cohort using the plan's two ICU medication windows (v4).

MIMIC-IV v2.2.  The script is intentionally explicit and non-destructive:
- ICU ``inputevents`` is the source for actual IV loop administrations in both
  ICU 0--24 h and ICU 24--48 h windows;
- the default primary treatment rule is the prespecified two-window
  operationalization: observed initiation (0--24 h has no administration and
  24--48 h has an administration) OR a quantified increase in daily
  furosemide-equivalent dose from 0--24 h to 24--48 h (increase > 0);
- a second, broader ``first 48 h any administration`` rule is also calculated
  for sensitivity/audit, but is not the default primary rule;
- no pre-ICU baseline is used, so initiation is labelled ``observed`` rather
  than confirmed clinical initiation;
- prescriptions are used only to describe intended orders for trajectory
  availability, never as actual dose;
- ICU observation completeness and dose quantification completeness are
  recorded separately and are never silently converted to zero;
- the old primary cohort's ESRD/dialysis, cardiac-surgery, durable-MCS,
  CMO, and trajectory-data exclusion definitions are retained as explicit,
  auditable flags.

The input --candidate-cohort is the upstream HF+ICU candidate file (normally the
old 5,378-row source cohort). It must have one or more rows per stay and include
subject_id, hadm_id, stay_id, admittime, dischtime, intime, outtime, age_at_admission,
and icu_los_hours. Extra columns are retained in the outputs.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import duckdb
import pandas as pd

LOOP_RE = r"(furosemide|lasix|bumetanide|bumex|torsemide|demadex|ethacrynic|edecrin)"


def args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mimic-dir', required=True, type=Path, help='MIMIC-IV-2.2 root containing hosp/ and icu/')
    p.add_argument('--candidate-cohort', required=True, type=Path, help='upstream HF+ICU candidates CSV')
    p.add_argument('--old-cohort', type=Path, default=None,
                   help=('Optional previous primary cohort CSV. When supplied, the script writes '
                         'old_new_cohort_changes.csv with stay-level retained/removed/added reasons.'))
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument(
        '--primary-treatment-rule',
        choices=['initiation_or_escalation_24_48', 'first_48h_any_actual'],
        default='initiation_or_escalation_24_48',
        help=(
            'Primary acute-treatment gate. The default follows the two-window '
            'operationalization: 0-24 h baseline, 24-48 h observed initiation '
            'or dose increase >0. first_48h_any_actual is a broader sensitivity rule.'
        ),
    )
    return p.parse_args()


def sql_path(p: Path)->str:
    return str(p.expanduser().resolve()).replace("'", "''")

def main():
    a=args(); root=a.mimic_dir.expanduser().resolve(); cand=a.candidate_cohort.expanduser().resolve(); out=a.output_dir.expanduser().resolve()
    old_path=a.old_cohort.expanduser().resolve() if a.old_cohort is not None else None
    # Fail before mkdir when example placeholders or missing inputs are supplied.
    placeholder_tokens = ('/path/to/', 'path/to/')
    supplied = (str(root), str(cand), str(out), str(old_path) if old_path else '')
    if any(token in value for value in supplied for token in placeholder_tokens):
        raise SystemExit(
            'The command still contains an example placeholder (/path/to/...). '
            'Replace --mimic-dir, --candidate-cohort, and --output-dir with real absolute paths.'
        )
    hosp=root/'hosp'; icu=root/'icu'
    required=[
        cand,
        hosp/'admissions.csv.gz', hosp/'diagnoses_icd.csv.gz',
        hosp/'d_icd_diagnoses.csv.gz', hosp/'services.csv.gz',
        hosp/'procedures_icd.csv.gz', hosp/'d_icd_procedures.csv.gz',
        hosp/'prescriptions.csv.gz', hosp/'labevents.csv.gz',
        hosp/'d_labitems.csv.gz',
        icu/'inputevents.csv.gz', icu/'outputevents.csv.gz',
        icu/'d_items.csv.gz', icu/'chartevents.csv.gz'
    ]
    missing=[str(x) for x in required if not x.exists()]
    if old_path is not None and not old_path.exists():
        missing.append(str(old_path))
    if missing:
        detail = 'Missing required files (check --mimic-dir and --candidate-cohort):\n' + '\n'.join(missing)
        if str(icu/'chartevents.csv.gz') in missing:
            detail += ('\n\nchartevents.csv.gz is required for the plan-defined comfort-measures-only '
                       'exclusion during the first 24 ICU hours; the script will not silently '
                       'skip this exclusion.')
        raise FileNotFoundError(detail)
    out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(out/'rebuild.duckdb'))
    cp=sql_path(cand)
    # Read candidate rows and choose first qualifying ICU stay per patient only after flags are built.
    con.execute(f"""
      CREATE OR REPLACE TABLE candidate AS
      SELECT * FROM read_csv_auto('{cp}', header=true, union_by_name=true, null_padding=true)
    """)
    cols={r[0] for r in con.execute('DESCRIBE candidate').fetchall()}
    need=['subject_id','hadm_id','stay_id','admittime','dischtime','intime','outtime','age_at_admission','icu_los_hours']
    absent=[x for x in need if x not in cols]
    if absent: raise ValueError(f'candidate-cohort missing columns: {absent}')

    hp=lambda n: sql_path(hosp/n); ip=lambda n: sql_path(icu/n)
    # Normalize fields once; timestamps are explicitly cast.
    con.execute("""CREATE OR REPLACE TABLE c AS SELECT *, try_cast(admittime AS TIMESTAMP) adm, try_cast(dischtime AS TIMESTAMP) dis,
      try_cast(intime AS TIMESTAMP) icu_in, try_cast(outtime AS TIMESTAMP) icu_out,
      try_cast(age_at_admission AS DOUBLE) age, try_cast(icu_los_hours AS DOUBLE) los_h FROM candidate""")

    # Candidate first-stay deduplication is performed after all treatment and
    # exclusion flags have been built.  Inputevents is read as VARCHAR so a
    # malformed value such as "___" cannot abort the complete cohort rebuild.
    con.execute(f"""
    CREATE OR REPLACE TABLE iv_icu_events AS
    SELECT try_cast(i.subject_id AS BIGINT) subject_id,
      try_cast(i.hadm_id AS BIGINT) hadm_id,
      try_cast(i.stay_id AS BIGINT) stay_id,
      try_cast(i.starttime AS TIMESTAMP) starttime,
      coalesce(try_cast(i.endtime AS TIMESTAMP), try_cast(i.starttime AS TIMESTAMP)) endtime,
      try_cast(i.itemid AS BIGINT) itemid,
      lower(coalesce(CAST(d.label AS VARCHAR),'')) AS label,
      CASE WHEN regexp_matches(lower(coalesce(CAST(d.label AS VARCHAR),'')), '{LOOP_RE}') THEN 1 ELSE 0 END AS is_loop,
      try_cast(i.amount AS DOUBLE) AS amount,
      lower(trim(coalesce(CAST(i.amountuom AS VARCHAR),''))) amountuom,
      try_cast(i.rate AS DOUBLE) AS rate,
      lower(trim(coalesce(CAST(i.rateuom AS VARCHAR),''))) rateuom,
      lower(trim(coalesce(CAST(i.statusdescription AS VARCHAR),''))) statusdescription,
      CAST(i.orderid AS VARCHAR) orderid, CAST(i.linkorderid AS VARCHAR) linkorderid
    FROM read_csv_auto('{ip('inputevents.csv.gz')}', header=true, union_by_name=true, null_padding=true, all_varchar=true) i
    LEFT JOIN read_csv_auto('{ip('d_items.csv.gz')}', header=true, union_by_name=true, null_padding=true, all_varchar=true) d
      ON try_cast(i.itemid AS BIGINT)=try_cast(d.itemid AS BIGINT)
    WHERE regexp_matches(lower(coalesce(CAST(d.label AS VARCHAR),'')), '{LOOP_RE}')
      AND NOT regexp_matches(lower(coalesce(CAST(i.statusdescription AS VARCHAR),'')), 'not started|not given|refus|hold|off schedule|cancel|discontinu')
      AND try_cast(i.starttime AS TIMESTAMP) IS NOT NULL
    """)
    # Split each actual administration into ICU-relative windows.  Infusion
    # rates are integrated only when the rate unit is a drug-mass unit; mL/hour
    # rates are deliberately left unquantified because concentration is not
    # recoverable from inputevents alone.  Bolus amounts are converted from mg,
    # g, mcg/ug when the amount unit is explicit.
    con.execute("""
    CREATE OR REPLACE TABLE doses AS
    WITH windows AS (
      SELECT c.subject_id,c.hadm_id,c.stay_id,c.icu_in,c.icu_out,
        c.icu_in + INTERVAL '24 hours' post0_end,
        c.icu_in + INTERVAL '48 hours' post1_end
      FROM c
    ), icu_windowed AS (
      SELECT w.subject_id,w.hadm_id,w.stay_id,'icu_inputevents' source_table,
        e.starttime event_start,e.endtime event_end,lower(e.label) med_label,
        e.amount,e.amountuom,e.rate,e.rateuom,
        'icu_0_24h' window_name,w.icu_in window_start,w.post0_end window_end
      FROM windows w JOIN iv_icu_events e ON e.stay_id=w.stay_id
      WHERE e.is_loop=1 AND e.endtime>w.icu_in AND e.starttime<w.post0_end
      UNION ALL
      SELECT w.subject_id,w.hadm_id,w.stay_id,'icu_inputevents',
        e.starttime,e.endtime,lower(e.label),e.amount,e.amountuom,e.rate,e.rateuom,
        'icu_24_48h',w.post0_end,w.post1_end
      FROM windows w JOIN iv_icu_events e ON e.stay_id=w.stay_id
      WHERE e.is_loop=1 AND e.endtime>w.post0_end AND e.starttime<w.post1_end
    ), parsed AS (
      SELECT *,
        CASE
          WHEN amount IS NULL THEN NULL
          WHEN amountuom LIKE '%mcg%' OR amountuom LIKE '%ug%' OR amountuom LIKE '%microgram%' THEN amount/1000.0
          WHEN amountuom LIKE '%gram%' OR amountuom IN ('g','gm','gms') THEN amount*1000.0
          WHEN amountuom LIKE '%mg%' THEN amount
          ELSE NULL
        END amount_mg,
        greatest(0,date_diff('second',greatest(event_start,window_start),least(event_end,window_end))) overlap_seconds
      FROM icu_windowed
    ), quantified AS (
      SELECT *,
        CASE
          WHEN rate IS NOT NULL AND rate>0 AND event_end>event_start
               AND (rateuom LIKE '%mg/hr%' OR rateuom LIKE '%mg/hour%' OR rateuom IN ('mg/h','mg/hr','mg/hour'))
            THEN rate*overlap_seconds/3600.0
          WHEN rate IS NOT NULL AND rate>0 AND event_end>event_start
               AND (rateuom LIKE '%mg/min%' OR rateuom LIKE '%mg per min%')
            THEN rate*overlap_seconds/60.0
          WHEN rate IS NOT NULL AND rate>0 AND event_end>event_start
               AND (rateuom LIKE '%g/hr%' OR rateuom LIKE '%g/hour%' OR rateuom IN ('g/h','g/hr','g/hour'))
            THEN rate*1000.0*overlap_seconds/3600.0
          WHEN rate IS NOT NULL AND rate>0 AND event_end>event_start
               AND (rateuom LIKE '%g/min%' OR rateuom LIKE '%g per min%')
            THEN rate*1000.0*overlap_seconds/60.0
          WHEN rate IS NOT NULL AND rate>0 AND event_end>event_start
               AND (rateuom LIKE '%mcg/min%' OR rateuom LIKE '%ug/min%' OR rateuom LIKE '%microgram/min%')
            THEN rate*overlap_seconds/60000.0
          WHEN rate IS NOT NULL AND rate>0 AND event_end>event_start
               AND (rateuom LIKE '%mcg/hr%' OR rateuom LIKE '%ug/hr%' OR rateuom LIKE '%microgram/hr%'
                    OR rateuom LIKE '%mcg/hour%' OR rateuom LIKE '%ug/hour%')
            THEN rate*overlap_seconds/3600000.0
          WHEN event_start>=window_start AND event_start<window_end THEN amount_mg
          ELSE NULL
        END dose_raw_mg
      FROM parsed
    )
    SELECT subject_id,hadm_id,stay_id,source_table,event_start,event_end,med_label,
      amount,amountuom,rate,rateuom,window_name,window_start,window_end,
      try_cast(dose_raw_mg AS DOUBLE) dose_raw_mg
    FROM quantified
    """)
    # Convert to IV furosemide equivalents. An administered medication with an
    # unquantifiable amount/rate remains exposure-positive and is explicitly
    # counted as unquantified; it is not silently assigned zero.
    con.execute("""
    CREATE OR REPLACE TABLE dose_norm AS
    SELECT *, CASE
      WHEN regexp_matches(med_label,'bumetanide|bumex') THEN dose_raw_mg*40
      WHEN regexp_matches(med_label,'torsemide|demadex') THEN dose_raw_mg*2
      WHEN regexp_matches(med_label,'furosemide|lasix') THEN dose_raw_mg
      WHEN regexp_matches(med_label,'ethacrynic|edecrin') THEN dose_raw_mg
      ELSE NULL END dose_mg_eq,
      dose_raw_mg IS NOT NULL AS dose_observed
    FROM doses
    """)
    con.execute("""
    CREATE OR REPLACE TABLE dose_summary AS
    SELECT c.subject_id,c.hadm_id,c.stay_id,
      count(*) FILTER (WHERE window_name='icu_0_24h') post0_admin_n,
      count(*) FILTER (WHERE window_name='icu_24_48h') post1_admin_n,
      sum(dose_mg_eq) FILTER (WHERE window_name='icu_0_24h') post0_dose_mg_eq,
      sum(dose_mg_eq) FILTER (WHERE window_name='icu_24_48h') post1_dose_mg_eq,
      min(event_start) FILTER (WHERE window_name='icu_0_24h') first_post0_time,
      min(event_start) FILTER (WHERE window_name='icu_24_48h') first_post1_time,
      string_agg(DISTINCT source_table, ';') dose_sources,
      count(*) FILTER (WHERE window_name='icu_0_24h' AND dose_observed) post0_quantified_n,
      count(*) FILTER (WHERE window_name='icu_24_48h' AND dose_observed) post1_quantified_n,
      count(*) FILTER (WHERE window_name='icu_0_24h' AND NOT dose_observed) post0_unquantified_n,
      count(*) FILTER (WHERE window_name='icu_24_48h' AND NOT dose_observed) post1_unquantified_n
    FROM c LEFT JOIN dose_norm USING(subject_id,hadm_id,stay_id) GROUP BY ALL
    """)
    # The ICU LOS bounds define whether the two ICU-relative windows are complete.
    # Treatment decisions are deliberately separated into administration-based
    # initiation and dose-based escalation.  The latter is not evaluable when
    # either window contains an administered but unquantified event.
    con.execute("""
    CREATE OR REPLACE TABLE audit AS
    SELECT c.*, s.* EXCLUDE(subject_id,hadm_id,stay_id),
      greatest(0, date_diff('second', icu_in,
        least(icu_in + INTERVAL '24 hours', coalesce(icu_out,icu_in))))/3600.0 icu_0_24_observed_hours,
      greatest(0, date_diff('second', icu_in,
        least(icu_in + INTERVAL '48 hours', coalesce(icu_out,icu_in))))/3600.0 icu_0_48_observed_hours,
      greatest(0, date_diff('second', icu_in + INTERVAL '24 hours',
        least(icu_in + INTERVAL '48 hours', coalesce(icu_out,icu_in))))/3600.0 icu_24_48_observed_hours,
      (coalesce(icu_out,icu_in) >= icu_in + INTERVAL '24 hours') AS icu_0_24_complete,
      (coalesce(icu_out,icu_in) >= icu_in + INTERVAL '48 hours') AS icu_24_48_complete,
      (coalesce(icu_out,icu_in) >= icu_in + INTERVAL '48 hours') AS icu_0_48_complete,
      (coalesce(post0_admin_n,0)>0) AS iv_loop_0_24_any,
      (coalesce(post1_admin_n,0)>0) AS iv_loop_24_48_any,
      (coalesce(post0_admin_n,0)+coalesce(post1_admin_n,0)>0) AS iv_loop_0_48_any,
      coalesce(post0_dose_mg_eq,0.0) AS iv_loop_0_24_dose_mg_eq,
      coalesce(post1_dose_mg_eq,0.0) AS iv_loop_24_48_dose_mg_eq,
      (coalesce(post0_unquantified_n,0)=0) AS iv_loop_0_24_dose_quantified,
      (coalesce(post1_unquantified_n,0)=0) AS iv_loop_24_48_dose_quantified,
      (coalesce(icu_out,icu_in) >= icu_in + INTERVAL '48 hours')
        AND coalesce(post0_unquantified_n,0)=0
        AND coalesce(post1_unquantified_n,0)=0 AS dose_change_0_24_to_24_48_evaluable,
      (coalesce(icu_out,icu_in) >= icu_in + INTERVAL '48 hours')
        AND coalesce(post0_admin_n,0)=0 AND coalesce(post1_admin_n,0)>0 AS observed_initiation_24_48,
      (coalesce(icu_out,icu_in) >= icu_in + INTERVAL '48 hours')
        AND coalesce(post0_dose_mg_eq,0)>0
        AND coalesce(post0_unquantified_n,0)=0
        AND coalesce(post1_unquantified_n,0)=0
        AND coalesce(post1_dose_mg_eq,0)>coalesce(post0_dose_mg_eq,0)
        AS observed_escalation_24_48,
      ((coalesce(icu_out,icu_in) >= icu_in + INTERVAL '48 hours')
        AND coalesce(post0_admin_n,0)=0 AND coalesce(post1_admin_n,0)>0)
        OR ((coalesce(icu_out,icu_in) >= icu_in + INTERVAL '48 hours')
          AND coalesce(post0_dose_mg_eq,0)>0
          AND coalesce(post0_unquantified_n,0)=0
          AND coalesce(post1_unquantified_n,0)=0
          AND coalesce(post1_dose_mg_eq,0)>coalesce(post0_dose_mg_eq,0))
        AS initiation_or_escalation_24_48,
      (coalesce(post0_admin_n,0)+coalesce(post1_admin_n,0)>0) AS first_48h_any_actual,
      CASE
        WHEN coalesce(icu_out,icu_in) < icu_in + INTERVAL '48 hours' THEN 'incomplete_0_48h_observation'
        WHEN coalesce(post0_admin_n,0)=0 AND coalesce(post1_admin_n,0)>0 THEN 'observed_initiation_0_to_1_24_48h'
        WHEN coalesce(post0_admin_n,0)>0 AND coalesce(post0_unquantified_n,0)>0 THEN 'escalation_not_evaluable_unquantified_0_24h'
        WHEN coalesce(post0_admin_n,0)>0 AND coalesce(post1_unquantified_n,0)>0 THEN 'escalation_not_evaluable_unquantified_24_48h'
        WHEN coalesce(post0_dose_mg_eq,0)>0 AND coalesce(post1_dose_mg_eq,0)>coalesce(post0_dose_mg_eq,0) THEN 'observed_escalation_dose_increase_gt0_24_48h'
        WHEN coalesce(post0_admin_n,0)>0 AND coalesce(post1_admin_n,0)=0 THEN 'no_post_24_48h_iv_loop'
        WHEN coalesce(post0_admin_n,0)>0 THEN 'no_dose_increase_24_48h'
        ELSE 'no_iv_loop_0_48h' END decision_reason
    FROM c LEFT JOIN dose_summary s USING(subject_id,hadm_id,stay_id)
    """)

    # Plan-defined trajectory sufficiency check: within ICU days 0--4, retain
    # a stay only when at least two distinct days contain >=1 measured marker.
    # This is an availability check, not a requirement that every marker be
    # present on every day.
    con.execute(f"""
    CREATE OR REPLACE TABLE trajectory_day_counts AS
    WITH input_days AS (
      SELECT DISTINCT c.subject_id,c.hadm_id,c.stay_id,
        CAST(FLOOR(date_diff('hour',c.icu_in,try_cast(i.starttime AS TIMESTAMP))/24.0) AS INTEGER) day_num,
        'input' marker
      FROM c JOIN read_csv_auto('{ip('inputevents.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) i
        ON try_cast(i.subject_id AS BIGINT)=c.subject_id AND try_cast(i.hadm_id AS BIGINT)=c.hadm_id AND try_cast(i.stay_id AS BIGINT)=c.stay_id
      WHERE try_cast(i.starttime AS TIMESTAMP)>=c.icu_in AND try_cast(i.starttime AS TIMESTAMP)<c.icu_in+INTERVAL '5 days'
    ), output_days AS (
      SELECT DISTINCT c.subject_id,c.hadm_id,c.stay_id,
        CAST(FLOOR(date_diff('hour',c.icu_in,try_cast(o.charttime AS TIMESTAMP))/24.0) AS INTEGER) day_num,
        'output' marker
      FROM c JOIN read_csv_auto('{ip('outputevents.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) o
        ON try_cast(o.subject_id AS BIGINT)=c.subject_id AND try_cast(o.hadm_id AS BIGINT)=c.hadm_id AND try_cast(o.stay_id AS BIGINT)=c.stay_id
      WHERE try_cast(o.charttime AS TIMESTAMP)>=c.icu_in AND try_cast(o.charttime AS TIMESTAMP)<c.icu_in+INTERVAL '5 days'
    ), urine_days AS (
      SELECT DISTINCT c.subject_id,c.hadm_id,c.stay_id,
        CAST(FLOOR(date_diff('hour',c.icu_in,try_cast(o.charttime AS TIMESTAMP))/24.0) AS INTEGER) day_num,
        'urine_output' marker
      FROM c JOIN read_csv_auto('{ip('outputevents.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) o
        ON try_cast(o.subject_id AS BIGINT)=c.subject_id AND try_cast(o.hadm_id AS BIGINT)=c.hadm_id AND try_cast(o.stay_id AS BIGINT)=c.stay_id
      LEFT JOIN read_csv_auto('{ip('d_items.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) di
        ON try_cast(o.itemid AS BIGINT)=try_cast(di.itemid AS BIGINT)
      WHERE try_cast(o.charttime AS TIMESTAMP)>=c.icu_in AND try_cast(o.charttime AS TIMESTAMP)<c.icu_in+INTERVAL '5 days'
        AND regexp_matches(lower(coalesce(CAST(di.label AS VARCHAR),'')),'urine')
    ), creatinine_days AS (
      SELECT DISTINCT c.subject_id,c.hadm_id,c.stay_id,
        CAST(FLOOR(date_diff('hour',c.icu_in,try_cast(l.charttime AS TIMESTAMP))/24.0) AS INTEGER) day_num,
        'creatinine' marker
      FROM c JOIN read_csv_auto('{hp('labevents.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) l
        ON try_cast(l.subject_id AS BIGINT)=c.subject_id AND try_cast(l.hadm_id AS BIGINT)=c.hadm_id
      LEFT JOIN read_csv_auto('{hp('d_labitems.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) dl
        ON try_cast(l.itemid AS BIGINT)=try_cast(dl.itemid AS BIGINT)
      WHERE try_cast(l.charttime AS TIMESTAMP)>=c.icu_in AND try_cast(l.charttime AS TIMESTAMP)<c.icu_in+INTERVAL '5 days'
        AND lower(coalesce(CAST(dl.label AS VARCHAR),''))='creatinine'
        AND try_cast(l.valuenum AS DOUBLE) IS NOT NULL
    ), loop_input_days AS (
      SELECT DISTINCT c.subject_id,c.hadm_id,c.stay_id,
        CAST(FLOOR(date_diff('hour',c.icu_in,e.starttime)/24.0) AS INTEGER) day_num,
        'loop_diuretic' marker
      FROM c JOIN iv_icu_events e ON e.subject_id=c.subject_id AND e.hadm_id=c.hadm_id AND e.stay_id=c.stay_id
      WHERE e.starttime>=c.icu_in AND e.starttime<c.icu_in+INTERVAL '5 days'
    ), loop_prescription_days AS (
      SELECT DISTINCT c.subject_id,c.hadm_id,c.stay_id,
        CAST(FLOOR(date_diff('hour',c.icu_in,try_cast(p.starttime AS TIMESTAMP))/24.0) AS INTEGER) day_num,
        'loop_diuretic' marker
      FROM c JOIN read_csv_auto('{hp('prescriptions.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) p
        ON try_cast(p.subject_id AS BIGINT)=c.subject_id AND try_cast(p.hadm_id AS BIGINT)=c.hadm_id
      WHERE try_cast(p.starttime AS TIMESTAMP)>=c.icu_in AND try_cast(p.starttime AS TIMESTAMP)<c.icu_in+INTERVAL '5 days'
        AND regexp_matches(lower(coalesce(CAST(p.drug AS VARCHAR),'')),'{LOOP_RE}')
    ), all_marker_days AS (
      SELECT * FROM input_days WHERE day_num BETWEEN 0 AND 4
      UNION SELECT * FROM output_days WHERE day_num BETWEEN 0 AND 4
      UNION SELECT * FROM urine_days WHERE day_num BETWEEN 0 AND 4
      UNION SELECT * FROM creatinine_days WHERE day_num BETWEEN 0 AND 4
      UNION SELECT * FROM loop_input_days WHERE day_num BETWEEN 0 AND 4
      UNION SELECT * FROM loop_prescription_days WHERE day_num BETWEEN 0 AND 4
    )
    SELECT subject_id,hadm_id,stay_id,count(DISTINCT day_num) n_days_with_any_marker,
      string_agg(DISTINCT marker,'; ') available_marker_types
    FROM all_marker_days GROUP BY subject_id,hadm_id,stay_id
    """)
    # Strict, plan-defined CMO evidence. Generic words such as "comfort",
    # "comfortable", or "provided comfort" are deliberately not sufficient.
    con.execute(f"""
    CREATE OR REPLACE TABLE cmo_evidence AS
    SELECT a.subject_id,a.hadm_id,a.stay_id,
      try_cast(ce.charttime AS TIMESTAMP) charttime,
      try_cast(ce.itemid AS BIGINT) itemid,
      lower(coalesce(CAST(di.label AS VARCHAR),'')) item_label,
      lower(coalesce(CAST(ce.value AS VARCHAR),'')) chart_value,
      CASE WHEN lower(coalesce(CAST(ce.value AS VARCHAR),'')) LIKE '%comfort measures%' THEN 'comfort_measures_text'
           WHEN lower(coalesce(CAST(ce.value AS VARCHAR),'')) LIKE '%comfort care%' THEN 'comfort_care_text'
           WHEN lower(trim(coalesce(CAST(ce.value AS VARCHAR),'')))='cmo' THEN 'cmo_exact'
           WHEN lower(coalesce(CAST(di.label AS VARCHAR),'')) LIKE '%code status%'
             AND lower(coalesce(CAST(ce.value AS VARCHAR),'')) LIKE '%comfort%' THEN 'code_status_comfort'
           ELSE 'unclassified' END cmo_match_rule
    FROM audit a
    JOIN read_csv_auto('{ip('chartevents.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) ce
      ON try_cast(ce.subject_id AS BIGINT)=a.subject_id
     AND try_cast(ce.hadm_id AS BIGINT)=a.hadm_id
     AND try_cast(ce.stay_id AS BIGINT)=a.stay_id
    LEFT JOIN read_csv_auto('{ip('d_items.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) di
      ON try_cast(ce.itemid AS BIGINT)=try_cast(di.itemid AS BIGINT)
    WHERE try_cast(ce.charttime AS TIMESTAMP)>=a.icu_in
      AND try_cast(ce.charttime AS TIMESTAMP)<a.icu_in+INTERVAL '24 hours'
      AND (lower(coalesce(CAST(ce.value AS VARCHAR),'')) LIKE '%comfort measures%'
        OR lower(coalesce(CAST(ce.value AS VARCHAR),'')) LIKE '%comfort care%'
        OR lower(trim(coalesce(CAST(ce.value AS VARCHAR),'')))='cmo'
        OR (lower(coalesce(CAST(di.label AS VARCHAR),'')) LIKE '%code status%'
            AND lower(coalesce(CAST(ce.value AS VARCHAR),'')) LIKE '%comfort%'))
    """)

    primary_treatment_column = (
        'initiation_or_escalation_24_48'
        if a.primary_treatment_rule == 'initiation_or_escalation_24_48'
        else 'first_48h_any_actual'
    )

    # Reapply plan exclusions sequentially. The candidate file is assumed to
    # represent the upstream HF+ICU source, but every criterion is still audited.
    con.execute(f"""
    CREATE OR REPLACE TABLE flags AS
    WITH dx AS (
      SELECT try_cast(subject_id AS BIGINT) subject_id,try_cast(hadm_id AS BIGINT) hadm_id,
        try_cast(icd_version AS INTEGER) icd_version,CAST(icd_code AS VARCHAR) icd_code,
        replace(CAST(icd_code AS VARCHAR),'.','') code
      FROM read_csv_auto('{hp('diagnoses_icd.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true)
    ), dxt AS (
      SELECT dx.subject_id,dx.hadm_id,lower(coalesce(CAST(d.long_title AS VARCHAR),'')) title
      FROM dx LEFT JOIN read_csv_auto('{hp('d_icd_diagnoses.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) d
        ON dx.icd_version=try_cast(d.icd_version AS INTEGER) AND dx.icd_code=CAST(d.icd_code AS VARCHAR)
    ), esrd AS (
      SELECT DISTINCT subject_id,hadm_id FROM dxt
      WHERE regexp_matches(title,'end stage renal disease|dependence on renal dialysis|encounter for dialysis')
         OR EXISTS(SELECT 1 FROM dx z WHERE z.subject_id=dxt.subject_id AND z.hadm_id=dxt.hadm_id
           AND ((z.icd_version=9 AND (z.code IN ('5856','V4511') OR z.code LIKE 'V56%'))
             OR (z.icd_version=10 AND (z.code IN ('N186','Z992') OR z.code LIKE 'Z49%'))))
    ), fs AS (
      SELECT try_cast(subject_id AS BIGINT) subject_id,try_cast(hadm_id AS BIGINT) hadm_id,
        CAST(curr_service AS VARCHAR) curr_service,
        row_number() OVER(PARTITION BY subject_id,hadm_id ORDER BY try_cast(transfertime AS TIMESTAMP)) rn
      FROM read_csv_auto('{hp('services.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true)
    ), surg AS (
      SELECT DISTINCT subject_id,hadm_id FROM fs WHERE rn=1 AND upper(coalesce(curr_service,''))='CSURG'
    ), hf AS (
      SELECT DISTINCT subject_id,hadm_id FROM dx
      WHERE (icd_version=9 AND code LIKE '428%') OR (icd_version=10 AND code LIKE 'I50%')
    ), mcs_dx AS (
      SELECT DISTINCT subject_id,hadm_id FROM dxt
      WHERE regexp_matches(title,'ventricular assist|heart assist device|lvad|left ventricular assist')
    ), mcs_proc AS (
      SELECT DISTINCT try_cast(p.subject_id AS BIGINT) subject_id,try_cast(p.hadm_id AS BIGINT) hadm_id
      FROM read_csv_auto('{hp('procedures_icd.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) p
      LEFT JOIN read_csv_auto('{hp('d_icd_procedures.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) q
        ON try_cast(p.icd_version AS INTEGER)=try_cast(q.icd_version AS INTEGER) AND CAST(p.icd_code AS VARCHAR)=CAST(q.icd_code AS VARCHAR)
      WHERE (try_cast(p.icd_version AS INTEGER)=9 AND CAST(p.icd_code AS VARCHAR) IN ('3766','3768'))
         OR (try_cast(p.icd_version AS INTEGER)=10 AND (CAST(p.icd_code AS VARCHAR) LIKE '02HA%' OR CAST(p.icd_code AS VARCHAR) LIKE '02PA%'))
         OR regexp_matches(lower(coalesce(CAST(q.long_title AS VARCHAR),'')),'ventricular assist|heart assist|lvad|left ventricular assist')
    ), mcs AS (
      SELECT * FROM mcs_dx UNION SELECT * FROM mcs_proc
    ), proc AS (
      SELECT DISTINCT try_cast(p.subject_id AS BIGINT) subject_id,try_cast(p.hadm_id AS BIGINT) hadm_id
      FROM read_csv_auto('{hp('procedures_icd.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) p
      LEFT JOIN read_csv_auto('{hp('d_icd_procedures.csv.gz')}',header=true,union_by_name=true,null_padding=true,all_varchar=true) q
        ON try_cast(p.icd_version AS INTEGER)=try_cast(q.icd_version AS INTEGER) AND CAST(p.icd_code AS VARCHAR)=CAST(q.icd_code AS VARCHAR)
      WHERE (try_cast(p.icd_version AS INTEGER)=9 AND (CAST(p.icd_code AS VARCHAR) LIKE '35%' OR CAST(p.icd_code AS VARCHAR) LIKE '361%'))
         OR (try_cast(p.icd_version AS INTEGER)=10 AND (CAST(p.icd_code AS VARCHAR) LIKE '0210%' OR CAST(p.icd_code AS VARCHAR) LIKE '0211%' OR CAST(p.icd_code AS VARCHAR) LIKE '0212%' OR CAST(p.icd_code AS VARCHAR) LIKE '0213%' OR CAST(p.icd_code AS VARCHAR) LIKE '02RF%' OR CAST(p.icd_code AS VARCHAR) LIKE '02RG%' OR CAST(p.icd_code AS VARCHAR) LIKE '02RH%' OR CAST(p.icd_code AS VARCHAR) LIKE '02RJ%' OR CAST(p.icd_code AS VARCHAR) LIKE '02Q%'))
         OR regexp_matches(lower(coalesce(CAST(q.long_title AS VARCHAR),'')),'coronary artery bypass|aortocoronary bypass|heart valve|cardiac valve|mitral valve|aortic valve|tricuspid valve')
    ), cmo AS (
      SELECT DISTINCT subject_id,hadm_id,stay_id FROM cmo_evidence
    ), source AS (
      SELECT a.*,
        (h.subject_id IS NOT NULL) hf_pass,
        (e.subject_id IS NOT NULL) exclude_esrd,
        (s.subject_id IS NOT NULL OR p.subject_id IS NOT NULL) exclude_cardiac_surgery,
        (m.subject_id IS NOT NULL) exclude_durable_mcs,
        (cm.subject_id IS NOT NULL) exclude_cmo,
        coalesce(t.n_days_with_any_marker,0) n_days_with_any_marker,
        coalesce(t.available_marker_types,'') available_marker_types
      FROM audit a
      LEFT JOIN hf h USING(subject_id,hadm_id)
      LEFT JOIN esrd e USING(subject_id,hadm_id)
      LEFT JOIN surg s USING(subject_id,hadm_id)
      LEFT JOIN proc p USING(subject_id,hadm_id)
      LEFT JOIN mcs m USING(subject_id,hadm_id)
      LEFT JOIN cmo cm USING(subject_id,hadm_id,stay_id)
      LEFT JOIN trajectory_day_counts t USING(subject_id,hadm_id,stay_id)
    )
    SELECT *,
      (age>=18) age_pass,
      (los_h>=48) icu_los_pass,
      ({primary_treatment_column}) acute_treatment_plan_pass,
      (n_days_with_any_marker<2) exclude_insufficient_trajectory_data
    FROM source
    """)
    # Sequential first qualifying hospitalization/stay. Stable tie-breakers are explicit.
    con.execute("""
    CREATE OR REPLACE TABLE final_cohort AS
    SELECT * EXCLUDE(rn_patient,rn_hadm) FROM (
      SELECT f.*, row_number() OVER(PARTITION BY subject_id ORDER BY adm,icu_in,stay_id) rn_patient,
        row_number() OVER(PARTITION BY subject_id,hadm_id ORDER BY icu_in,stay_id) rn_hadm
      FROM flags f WHERE hf_pass AND age_pass AND icu_los_pass AND acute_treatment_plan_pass
        AND NOT exclude_esrd AND NOT exclude_cardiac_surgery AND NOT exclude_durable_mcs AND NOT exclude_cmo
        AND NOT exclude_insufficient_trajectory_data
    ) z WHERE rn_patient=1 AND rn_hadm=1
    """)

    # Optional stay-level comparison with the previous primary cohort. The
    # comparison is intentionally done after all flags are available so each
    # removed stay receives the first applicable updated-rule reason.
    if old_path is not None:
      old_df=pd.read_csv(old_path)
      old_need=[x for x in ['subject_id','hadm_id','stay_id'] if x not in old_df.columns]
      if old_need:
        raise ValueError(f'old-cohort missing columns: {old_need}')
      old_ids=old_df[['subject_id','hadm_id','stay_id']].copy()
      old_ids['old_primary']=True
      flag_df=con.execute("SELECT subject_id,hadm_id,stay_id,hf_pass,age_pass,icu_los_pass,acute_treatment_plan_pass,initiation_or_escalation_24_48,first_48h_any_actual,observed_initiation_24_48,observed_escalation_24_48,exclude_esrd,exclude_cardiac_surgery,exclude_durable_mcs,exclude_cmo,exclude_insufficient_trajectory_data,decision_reason FROM flags").df()
      new_ids=con.execute("SELECT subject_id,hadm_id,stay_id FROM final_cohort").df()
      new_ids['updated_primary']=True
      keys=['subject_id','hadm_id','stay_id']
      for frame in (old_ids,flag_df,new_ids):
        for col in keys:
          frame[col]=pd.to_numeric(frame[col],errors='coerce').astype('Int64')
      change=flag_df.merge(old_ids,on=keys,how='outer').merge(new_ids,on=keys,how='outer')
      for col in ('old_primary','updated_primary'):
        change[col]=change[col].astype('boolean').fillna(False).astype(bool)
      def change_reason(row):
        if row.old_primary and row.updated_primary: return 'retained_in_updated_primary'
        if (not row.old_primary) and row.updated_primary: return 'added_under_updated_definition'
        if not row.old_primary and not row.updated_primary: return 'not_in_either_primary_cohort'
        if pd.isna(row.hf_pass): return 'old_cohort_not_in_candidate_source'
        checks=[('hf_pass','removed_no_hf_diagnosis'),('age_pass','removed_age_lt18'),('icu_los_pass','removed_icu_los_lt48h'),('acute_treatment_plan_pass','removed_no_plan_defined_initiation_or_escalation'),('exclude_esrd','removed_esrd_or_dialysis'),('exclude_cardiac_surgery','removed_cardiac_surgery'),('exclude_durable_mcs','removed_durable_mcs'),('exclude_cmo','removed_cmo'),('exclude_insufficient_trajectory_data','removed_insufficient_trajectory_data')]
        for col,reason in checks:
          if col in ('hf_pass','age_pass','icu_los_pass','acute_treatment_plan_pass') and not bool(row.get(col,False)): return reason
          if col.startswith('exclude_') and bool(row.get(col,False)): return reason
        return 'removed_not_first_qualifying_record'
      change['change_reason']=change.apply(change_reason,axis=1)
      change.to_csv(out/'old_new_cohort_changes.csv',index=False)
      change.groupby('change_reason',dropna=False).size().reset_index(name='n').to_csv(out/'old_new_cohort_change_counts.csv',index=False)
    # Sequential flow is generated in Python so n_before, n_excluded, and n_after
    # are directly auditable and duplicate-patient removal is visible.
    current = con.execute("SELECT * FROM flags").df()
    flow_rows = [{"step":"candidate source", "n_before":None, "n_excluded_at_step":None, "n_after":len(current), "n_patients_after":current.subject_id.nunique(), "n_hadm_after":current.hadm_id.nunique(), "n_stays_after":current.stay_id.nunique()}]
    treatment_step_label = (
      "plan treatment: observed initiation (0-24h none -> 24-48h any) or dose increase >0"
      if a.primary_treatment_rule == 'initiation_or_escalation_24_48'
      else "plan sensitivity treatment: any actual IV loop in ICU 0-48h"
    )
    ordered_steps = [
      ("HF diagnosis ICD-9 428.x or ICD-10 I50.x", "hf_pass"),
      ("age >=18", "age_pass"), ("ICU LOS >=48h", "icu_los_pass"),
      (treatment_step_label, "acute_treatment_plan_pass"),
      ("exclude ESRD/dialysis", "exclude_esrd", True),
      ("exclude cardiac surgery", "exclude_cardiac_surgery", True),
      ("exclude durable MCS", "exclude_durable_mcs", True),
      ("exclude CMO", "exclude_cmo", True),
      ("exclude insufficient trajectory data (<2 marker-days)", "exclude_insufficient_trajectory_data", True),
    ]
    for item in ordered_steps:
      step,col,*exclude = item; before=len(current)
      keep = current[col].astype(bool) if not exclude else ~current[col].astype(bool)
      current=current.loc[keep].copy()
      flow_rows.append({"step":step,"n_before":before,"n_excluded_at_step":before-len(current),"n_after":len(current),"n_patients_after":current.subject_id.nunique(),"n_hadm_after":current.hadm_id.nunique(),"n_stays_after":current.stay_id.nunique()})
    before=len(current)
    current=current.sort_values(["subject_id","adm","icu_in","stay_id"]).drop_duplicates("subject_id",keep="first")
    flow_rows.append({"step":"first qualifying hospitalization and ICU stay per patient","n_before":before,"n_excluded_at_step":before-len(current),"n_after":len(current),"n_patients_after":current.subject_id.nunique(),"n_hadm_after":current.hadm_id.nunique(),"n_stays_after":current.stay_id.nunique()})
    flow=pd.DataFrame(flow_rows)
    con.register('flow_df',flow)
    con.execute('CREATE OR REPLACE TABLE flow AS SELECT * FROM flow_df')

    # Total flag counts mirror the old primary exclusion audit. These are
    # marginal counts, not sequential counts, so overlaps can be inspected
    # separately from the flow table.
    flag_count_map = [
      ('exclude_esrd_dialysis', 'exclude_esrd'),
      ('exclude_primary_cardiac_surgery', 'exclude_cardiac_surgery'),
      ('exclude_durable_mcs', 'exclude_durable_mcs'),
      ('exclude_comfort_measures_24h', 'exclude_cmo'),
      ('exclude_insufficient_trajectory_data', 'exclude_insufficient_trajectory_data'),
    ]

    flag_snapshot = con.execute("SELECT * FROM flags").df()
    flag_counts = pd.DataFrame([
      {'exclusion_flag': label, 'n_flagged_total': int(flag_snapshot[col].fillna(False).astype(bool).sum())}
      for label, col in flag_count_map
    ])
    flag_counts.to_csv(out/'exclusion_flag_counts_v4.csv', index=False)

    for table,file in [('audit','medication_window_audit_v4.csv'),('flags','cohort_flags_v4.csv'),('final_cohort','adhf_icu_cohort_v4.csv'),('flow','cohort_flow_v4.csv'),('cmo_evidence','cmo_evidence_v4.csv'),('trajectory_day_counts','trajectory_day_counts_v4.csv'),('dose_norm','iv_loop_dose_events_v4.csv')]:
      con.execute(f"COPY {table} TO '{sql_path(out/file)}' (HEADER, DELIMITER ',')")
    # Old candidate comparison by patient/stay identity; do not imply old and new
    # criteria are equivalent.
    old=con.execute("SELECT count(*) FROM c").fetchone()[0]; new=con.execute("SELECT count(*) FROM final_cohort").fetchone()[0]
    reasons=con.execute("SELECT decision_reason,count(*) n FROM flags GROUP BY decision_reason ORDER BY decision_reason").fetchall()
    decision_counts=con.execute("SELECT decision_reason,count(*) n FROM flags GROUP BY decision_reason ORDER BY decision_reason").fetchall()
    treatment_counts=con.execute("""
      SELECT
        count(*) AS candidate_n,
        count(*) FILTER (WHERE iv_loop_0_24_any) AS iv_loop_0_24_any_n,
        count(*) FILTER (WHERE iv_loop_24_48_any) AS iv_loop_24_48_any_n,
        count(*) FILTER (WHERE first_48h_any_actual) AS first_48h_any_actual_n,
        count(*) FILTER (WHERE observed_initiation_24_48) AS observed_initiation_24_48_n,
        count(*) FILTER (WHERE observed_escalation_24_48) AS observed_escalation_24_48_n,
        count(*) FILTER (WHERE initiation_or_escalation_24_48) AS initiation_or_escalation_24_48_n,
        count(*) FILTER (WHERE dose_change_0_24_to_24_48_evaluable) AS dose_change_evaluable_n,
        count(*) FILTER (WHERE NOT dose_change_0_24_to_24_48_evaluable) AS dose_change_not_evaluable_n,
        count(*) FILTER (WHERE post0_unquantified_n>0) AS post0_unquantified_stay_n,
        count(*) FILTER (WHERE post1_unquantified_n>0) AS post1_unquantified_stay_n
      FROM audit
    """).fetchone()
    treatment_summary={
      'candidate_n':int(treatment_counts[0]),
      'iv_loop_0_24_any_n':int(treatment_counts[1]),
      'iv_loop_24_48_any_n':int(treatment_counts[2]),
      'first_48h_any_actual_n':int(treatment_counts[3]),
      'observed_initiation_24_48_n':int(treatment_counts[4]),
      'observed_escalation_24_48_n':int(treatment_counts[5]),
      'initiation_or_escalation_24_48_n':int(treatment_counts[6]),
      'dose_change_evaluable_n':int(treatment_counts[7]),
      'dose_change_not_evaluable_n':int(treatment_counts[8]),
      'post0_unquantified_stay_n':int(treatment_counts[9]),
      'post1_unquantified_stay_n':int(treatment_counts[10]),
    }
    pd.DataFrame([{'metric':k,'n':v} for k,v in treatment_summary.items()]).to_csv(out/'treatment_decision_counts_v4.csv',index=False)
    if a.primary_treatment_rule == 'initiation_or_escalation_24_48':
      primary_window_meta = 'ICU 0-24 h baseline and ICU 24-48 h observation'
      primary_rule_desc_meta = ('observed initiation: no actual IV loop administration in ICU 0-24 h and at least one in ICU 24-48 h; '
                                'OR observed escalation: quantified 24-48 h furosemide-equivalent dose minus quantified 0-24 h dose > 0')
      secondary_window_meta = 'combined 0-48 h any actual administration retained as a sensitivity/audit flag'
      initiation_rule_meta = '0-24 h administration count = 0 and 24-48 h administration count > 0, with both ICU windows observed'
      escalation_rule_meta = 'quantified 24-48 h total furosemide-equivalent dose minus quantified 0-24 h total > 0; no arbitrary absolute or relative threshold'
    else:
      primary_window_meta = 'combined ICU 0-48 h treatment window'
      primary_rule_desc_meta = 'at least one actual IV loop administration in the combined ICU 0-48 h window'
      secondary_window_meta = '0-24 h and 24-48 h administration counts/doses retained for treatment-pattern audit; initiation/escalation flags are descriptive only'
      initiation_rule_meta = 'descriptive only: 0-24 h administration count = 0 and 24-48 h administration count > 0'
      escalation_rule_meta = 'descriptive only: quantified 24-48 h total furosemide-equivalent dose minus quantified 0-24 h total > 0'
    meta={'script':'rebuild_adhf_cohort_v4.py','version':'strict_plan_two_window_old_exclusions','mimic_version':'2.2','candidate_n':old,'updated_n':new,'delta':new-old,
          'primary_treatment_window':primary_window_meta,
          'primary_treatment_rule':a.primary_treatment_rule,
          'primary_treatment_rule_description':primary_rule_desc_meta,
          'secondary_treatment_window':secondary_window_meta,
          'pre_icu_window':'not used; initiation is observed ICU-window initiation and cannot establish outpatient/pre-ICU treatment initiation',
          'initiation_rule':initiation_rule_meta,
          'escalation_rule':escalation_rule_meta,
          'exclusion_rules':'retained from the old primary step2.py: ESRD/chronic dialysis, first service CSURG or specified cardiac-surgery procedure/title, durable MCS diagnosis/procedure/title, explicit CMO within ICU first 24h, and fewer than 2 marker-days in ICU days 0-4',
          'cmo_rule':'explicit comfort measures/comfort care/CMO or code-status comfort wording within ICU first 24h; generic comfort wording excluded',
          'trajectory_rule':'at least 2 ICU days (days 0-4) with >=1 measured marker',
          'dose_sources':'ICU inputevents for ICU 0-24h and ICU 24-48h; prescriptions are trajectory-availability only, never actual dose',
          'equivalence':{'furosemide':1,'bumetanide':40,'torsemide':2,'ethacrynic_acid':1},
          'treatment_decision_counts':treatment_summary,
          'decision_reason_counts':{str(k):int(v) for k,v in decision_counts},
          'cmo_evidence_rows':int(con.execute('SELECT count(*) FROM cmo_evidence').fetchone()[0]),
          'old_cohort_path':str(old_path) if old_path is not None else None,
          'old_new_comparison_written':old_path is not None}
    (out/'RUN_METADATA.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2)); print(f'Outputs: {out}')

if __name__=='__main__': main()
