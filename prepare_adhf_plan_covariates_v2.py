#!/usr/bin/env python3
"""Assemble and audit the covariates required by the ADHF trajectory plan.

The previous covariate extractor intentionally stopped at demographics, ICD
comorbidities, first-24-hour laboratories, and treatment flags.  This script
is the *plan gate*: it joins that table to independently generated score,
echocardiography, vital-sign, biomarker, and chronic-HF-therapy exports and
does not silently replace an unavailable plan variable with zero.

Each supplemental file must contain ``stay_id`` and be one row per stay.  A
file may contain ``subject_id``/``hadm_id`` as well; those columns are used to
check the join.  Column aliases are normalised to stable analysis names, and
an audit table records the source and missingness of every plan field.

Examples
--------
python prepare_adhf_plan_covariates_v2.py \\
  --cohort /path/adhf_icu_cohort_v4.csv \\
  --base-covariates /path/analysis_covariates_newcohort_v1.csv \\
  --supplement sofa=/path/sofa_first24.csv \\
  --supplement sapsii=/path/sapsii_first24.csv \\
  --supplement ef=/path/ef_category.csv \\
  --supplement hemodynamics=/path/admission_vitals.csv \\
  --supplement hf_therapy=/path/chronic_hf_therapy.csv \\
  --output-dir /path/plan_covariates_v2

The default is strict and exits if a required field group is absent.  Use
``--allow-missing-plan-fields`` only to produce an audit while collecting the
missing exports; that output must not be used as the final adjusted model.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Aliases are deliberately conservative.  A similarly named field is not
# guessed unless it is listed here, which prevents accidental use of a
# post-baseline or differently defined variable.
ALIASES = {
    "age_per10": ["age_per10", "age_10", "age_at_admission_per10"],
    "age_for_analysis": ["age_for_analysis", "age_at_admission", "age"],
    "sex_female": ["sex_female", "female", "gender_female"],
    "anchor_year_group": ["anchor_year_group", "care_era", "era", "anchor_year"],
    "charlson_index": ["charlson_index", "charlson", "cci", "charlson_comorbidity_index"],
    "elixhauser_score": ["elixhauser_score", "elixhauser", "van_walraven", "elixhauser_van_walraven"],
    "sofa_first24h": ["sofa_first24h", "sofa_24h", "first_day_sofa", "sofa_score", "sofa"],
    "sapsii": ["sapsii", "saps_ii", "sapsii_score", "saps2", "saps_ii_score"],
    "admission_sbp": ["admission_sbp", "first_sbp", "sbp_first", "admit_sbp", "systolic_bp"],
    "admission_hr": ["admission_hr", "first_hr", "hr_first", "admit_hr", "heart_rate_first"],
    "admission_rhythm": ["admission_rhythm", "rhythm_first", "initial_rhythm", "rhythm"],
    "baseline_ntprobnp": [
        "baseline_ntprobnp", "baseline_nt_pro_bnp", "baseline_ntprobnP",
        "baseline_bnp", "ntprobnp_first24h", "nt_pro_bnp_first24h", "bnp_first24h",
    ],
    "ef_category": ["ef_category", "lvef_category", "ef_group", "ejection_fraction_category"],
    "chronic_hf_loop": ["chronic_hf_loop", "home_loop_diuretic", "home_loop", "prior_loop_diuretic"],
    "chronic_hf_raas": ["chronic_hf_raas", "home_raas", "home_ace_arb_arni", "prior_raas"],
    "chronic_hf_beta_blocker": ["chronic_hf_beta_blocker", "home_beta_blocker", "prior_beta_blocker"],
    "chronic_hf_mra": ["chronic_hf_mra", "home_mra", "prior_mra", "home_mineralocorticoid"],
    "chronic_hf_sglt2": ["chronic_hf_sglt2", "home_sglt2", "prior_sglt2", "home_sglt2_inhibitor"],
    "chronic_hf_therapy": ["chronic_hf_therapy", "home_hf_therapy", "any_chronic_hf_therapy", "home_guideline_directed_therapy"],
}

# These are the plan's prespecified clinical confounders.  The first v1
# extractor already produces the ICD-derived fields; they are repeated here
# so the final audit is self-contained.
COMORBIDITY_FIELDS = [
    "diabetes", "ckd", "copd", "atrial_fibrillation", "prior_mi",
    "hypertension", "valvular_disease", "anemia",
]
FIRST24_FIELDS = [
    "vasopressor_first24h", "inotrope_first24h",
    "mechanical_ventilation_first24h", "rrt_first24h",
]
LAB_FIELDS = [
    "baseline_creatinine", "baseline_bun", "baseline_sodium",
    "baseline_potassium", "baseline_bicarbonate", "baseline_chloride",
    "baseline_hemoglobin",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cohort", required=True, type=Path)
    p.add_argument("--base-covariates", required=True, type=Path,
                   help="Output of prepare_adhf_mortality_covariates_v2.py or an equivalent table")
    p.add_argument("--supplement", action="append", default=[], metavar="NAME=CSV",
                   help="One-row-per-stay supplement, e.g. sofa=/path/sofa.csv; repeatable")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--allow-missing-plan-fields", action="store_true",
                   help="Write an audit without failing; not valid for the final adjusted model")
    p.add_argument("--ef-reference", default="preserved",
                   choices=["reduced", "midrange", "preserved"],
                   help="Reference EF category used when creating EF indicators")
    return p.parse_args()


def norm_name(x: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    d = pd.read_csv(path, low_memory=False)
    if d.empty:
        raise ValueError(f"{label} is empty: {path}")
    return d


def normalize_ids(d: pd.DataFrame, label: str) -> pd.DataFrame:
    out = d.copy()
    if "stay_id" not in out.columns:
        raise ValueError(f"{label} must contain stay_id")
    for c in [x for x in ID_COLS if x in out.columns]:
        out[c] = pd.to_numeric(out[c], errors="raise").astype("int64")
    out["stay_id"] = pd.to_numeric(out["stay_id"], errors="raise").astype("int64")
    if out["stay_id"].duplicated().any():
        dup = out.loc[out["stay_id"].duplicated(keep=False), "stay_id"].head(10).tolist()
        raise ValueError(f"{label} has multiple rows per stay_id (examples: {dup})")
    return out


def canonicalize_columns(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename the first matching alias to the canonical analysis name."""
    out = d.copy()
    lower = {norm_name(c): c for c in out.columns}
    source: dict[str, str] = {}
    for canonical, aliases in ALIASES.items():
        if canonical in out.columns:
            source[canonical] = canonical
            continue
        candidates = [canonical, *aliases]
        hit = next((lower[norm_name(x)] for x in candidates if norm_name(x) in lower), None)
        if hit is not None:
            out = out.rename(columns={hit: canonical})
            source[canonical] = hit
            lower[norm_name(canonical)] = canonical
    return out, source


def coerce_binary(d: pd.DataFrame, columns: Iterable[str]) -> None:
    for c in columns:
        if c not in d.columns:
            continue
        x = d[c]
        if x.dtype == object or str(x.dtype).startswith("string"):
            y = x.astype("string").str.strip().str.lower()
            d[c] = y.map({"true": 1, "false": 0, "yes": 1, "no": 0,
                          "y": 1, "n": 0, "female": 1, "male": 0,
                          "f": 1, "m": 0, "1": 1, "0": 0}).where(y.notna(), np.nan)
        d[c] = pd.to_numeric(d[c], errors="coerce")
        bad = d[c].notna() & ~d[c].isin([0, 1])
        if bad.any():
            raise ValueError(f"{c} contains non-binary values: {d.loc[bad, c].head().tolist()}")


def make_ef_indicators(d: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, str]:
    out = d.copy()
    if "ef_category" in out.columns:
        x = out["ef_category"].astype("string").str.strip().str.lower()
        # Accept common clinical labels, but keep unknown/missing explicit.
        mapped = x.replace({
            "hfrEF": "reduced", "hfref": "reduced", "reduced": "reduced",
            "systolic": "reduced", "hfmref": "midrange", "hfmrEF": "midrange",
            "hfmr ef": "midrange", "mid-range": "midrange", "midrange": "midrange",
            "preserved": "preserved", "diastolic": "preserved", "hfpef": "preserved",
        })
        # Numeric EF values are converted using standard clinical cut points;
        # this is only a fallback when a category was not already supplied.
        num = pd.to_numeric(x, errors="coerce")
        mapped = mapped.where(mapped.isin(["reduced", "midrange", "preserved"]), np.nan)
        mapped = mapped.where(mapped.notna(), pd.cut(num, [-np.inf, 39, 49, np.inf], labels=["reduced", "midrange", "preserved"]))
        mapped = mapped.astype("string").fillna("unknown")
        out["ef_category"] = mapped
        for cat in ["reduced", "midrange", "preserved", "unknown"]:
            out[f"ef_{cat}"] = (mapped == cat).astype(float)
    else:
        indicator_cols = [f"ef_{x}" for x in ["reduced", "midrange", "preserved", "unknown"]]
        if not any(c in out.columns for c in indicator_cols):
            return out, "missing"
        for c in indicator_cols:
            if c not in out.columns:
                out[c] = 0.0
        # Ensure exactly one indicator when multiple were supplied.
        vals = out[indicator_cols].apply(pd.to_numeric, errors="coerce")
        out[indicator_cols] = vals
        out["ef_category"] = vals.idxmax(axis=1).str.replace("ef_", "", regex=False)
        out.loc[vals.sum(axis=1).fillna(0).eq(0), "ef_category"] = "unknown"
    # Drop the reference indicator from the model only later; retain all four
    # in the prepared table to make the coding auditable.
    return out, reference


def parse_supplements(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--supplement must be NAME=CSV, got {item!r}")
        name, path = item.split("=", 1)
        name = norm_name(name)
        if not name or not path:
            raise ValueError(f"Invalid supplement specification: {item!r}")
        out.append((name, Path(path)))
    return out


def merge_one(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    right = normalize_ids(right, label)
    # Verify overlapping IDs when present.  This catches accidental joining of
    # a table from a different cohort version.
    # stay_id is the join key and has already been validated as unique above;
    # checking it here would select ``stay_id`` twice and turn the Series into
    # a 2-D DataFrame.  Only compare the auxiliary identifiers.
    for c in [x for x in ID_COLS if x != "stay_id" and x in left.columns and x in right.columns]:
        r = right[["stay_id", c]].dropna().set_index("stay_id")[c]
        l = left[["stay_id", c]].dropna().set_index("stay_id")[c]
        common = l.index.intersection(r.index)
        if len(common) and (l.loc[common].astype(str) != r.loc[common].astype(str)).any():
            raise ValueError(f"{label} has inconsistent {c} for overlapping stay_id values")
    overlap = [c for c in right.columns if c != "stay_id" and c in left.columns]
    if overlap:
        # Identical duplicate fields are harmless; conflicting duplicates are
        # not silently overwritten.
        for c in overlap:
            a = left.set_index("stay_id")[c]
            b = right.set_index("stay_id")[c]
            common = a.index.intersection(b.index)
            aa = pd.to_numeric(a.loc[common], errors="coerce")
            bb = pd.to_numeric(b.loc[common], errors="coerce")
            both_num = aa.notna() & bb.notna()
            conflict = (both_num & (aa - bb).abs().gt(1e-8)) | (
                ~both_num & a.loc[common].notna() & b.loc[common].notna() &
                (a.loc[common].astype(str) != b.loc[common].astype(str))
            )
            if conflict.any():
                raise ValueError(f"{label} conflicts with existing column {c}")
        right = right.drop(columns=overlap)
    return left.merge(right, on="stay_id", how="left", validate="one_to_one")


def main() -> None:
    a = parse_args()
    cohort_path = a.cohort.expanduser().resolve()
    base_path = a.base_covariates.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    cohort = normalize_ids(read_csv(cohort_path, "cohort"), "cohort")
    cohort = cohort.drop_duplicates("stay_id").copy()
    base = normalize_ids(read_csv(base_path, "base covariates"), "base covariates")
    d = cohort.merge(base, on="stay_id", how="left", suffixes=("", "_base"), validate="one_to_one")
    # Prefer the cohort's identifiers, but preserve all base fields.  If the
    # base table also has subject/hadm values, check them explicitly.
    for c in ["subject_id", "hadm_id"]:
        bc = f"{c}_base"
        if bc in d.columns and c in d.columns:
            l = pd.to_numeric(d[c], errors="coerce")
            r = pd.to_numeric(d[bc], errors="coerce")
            bad = l.notna() & r.notna() & (l != r)
            if bad.any():
                raise ValueError(f"Base covariates disagree with cohort {c} for stay_id examples {d.loc[bad, 'stay_id'].head().tolist()}")
            d = d.drop(columns=[bc])

    source_files = {"cohort": str(cohort_path), "base_covariates": str(base_path)}
    source_map: dict[str, str] = {}
    d, src = canonicalize_columns(d)
    source_map.update({k: "base/cohort:" + v for k, v in src.items()})
    for name, path in parse_supplements(a.supplement):
        s = normalize_ids(read_csv(path.expanduser().resolve(), f"supplement {name}"), f"supplement {name}")
        s, src_s = canonicalize_columns(s)
        d = merge_one(d, s, f"supplement {name}")
        source_files[name] = str(path.expanduser().resolve())
        for c, original in src_s.items():
            source_map[c] = f"{name}:{original}"
        for c in s.columns:
            source_map.setdefault(c, f"{name}:{c}")

    # Canonical age and sex are created from v1 names if needed.
    if "age_per10" not in d.columns and "age_for_analysis" in d.columns:
        d["age_per10"] = pd.to_numeric(d["age_for_analysis"], errors="coerce") / 10.0
        source_map["age_per10"] = "derived:age_for_analysis/10"
    if "sex_female" not in d.columns and "sex" in d.columns:
        d["sex_female"] = d["sex"].astype("string").str.upper().eq("F").astype(float)
        source_map["sex_female"] = "derived:sex==F"

    d, ef_ref = make_ef_indicators(d, a.ef_reference)
    for c in ["age_per10", "age_for_analysis", "sex_female", "charlson_index", "elixhauser_score",
              "sofa_first24h", "sapsii", "admission_sbp", "admission_hr", "baseline_ntprobnp"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    coerce_binary(d, COMORBIDITY_FIELDS + FIRST24_FIELDS + [
        "ef_reduced", "ef_midrange", "ef_preserved", "ef_unknown",
        "chronic_hf_loop", "chronic_hf_raas", "chronic_hf_beta_blocker",
        "chronic_hf_mra", "chronic_hf_sglt2", "chronic_hf_therapy",
    ])

    # Stable era string used for temporal partitioning.  Numeric anchor_year
    # values are retained; the validation script extracts the first 4 digits.
    if "anchor_year_group" not in d.columns:
        d["anchor_year_group"] = pd.NA

    # Required groups reflect the plan.  SOFA and SAPS-II are retained as
    # alternative severity specifications; at least one is needed for a
    # runnable adjusted model, while the audit distinguishes one vs both.
    required_single = [
        "age_per10", "sex_female", "baseline_creatinine", "baseline_ntprobnp",
        "admission_sbp", "admission_hr", "admission_rhythm",
    ] + COMORBIDITY_FIELDS + FIRST24_FIELDS
    ef_group = ["ef_reduced", "ef_midrange", "ef_preserved", "ef_unknown"]
    severity_group = ["sofa_first24h", "sapsii"]
    therapy_group = ["chronic_hf_loop", "chronic_hf_raas", "chronic_hf_beta_blocker", "chronic_hf_mra", "chronic_hf_sglt2", "chronic_hf_therapy"]
    missing_required = [c for c in required_single if c not in d.columns]
    missing_ef = not any(c in d.columns for c in ef_group) or all(d[c].isna().all() for c in ef_group if c in d.columns)
    observed_severity = [c for c in severity_group if c in d.columns and d[c].notna().any()]
    observed_therapy = [c for c in therapy_group if c in d.columns and d[c].notna().any()]

    audit_fields = required_single + ef_group + severity_group + therapy_group + ["anchor_year_group"]
    audit_rows = []
    for c in audit_fields:
        present = c in d.columns
        n_obs = int(d[c].notna().sum()) if present else 0
        audit_rows.append({
            "field": c,
            "present": int(present),
            "n_rows": int(len(d)),
            "n_observed": n_obs,
            "missing_percent": 100.0 * (1.0 - n_obs / len(d)) if len(d) else np.nan,
            "source": source_map.get(c, "not_found"),
            "plan_role": "required_single" if c in required_single else (
                "ef_category" if c in ef_group else ("severity_alternative" if c in severity_group else (
                    "chronic_hf_therapy" if c in therapy_group else "temporal_era")))
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "plan_covariate_field_audit_v2.csv", index=False)

    # Numeric coercion for every canonical analysis field; rhythm/EF category
    # stay as strings and are encoded below in the model-input script.
    d.to_csv(out / "analysis_covariates_plan_v2.csv", index=False)
    pd.DataFrame({"variable": d.columns,
                  "n_missing": [int(d[c].isna().sum()) for c in d.columns],
                  "missing_percent": [100.0 * d[c].isna().mean() for c in d.columns]}).to_csv(
        out / "analysis_covariates_missingness_plan_v2.csv", index=False)

    errors = []
    if missing_required:
        errors.append("missing required fields: " + ", ".join(missing_required))
    if missing_ef:
        errors.append("no observed EF category/indicator field")
    if not observed_severity:
        errors.append("neither sofa_first24h nor sapsii is observed")
    if not observed_therapy:
        errors.append("no chronic HF therapy flag is observed")
    # anchor_year_group is required only for temporal validation; permit an
    # audit-only run to proceed, but make the omission explicit.
    if d["anchor_year_group"].notna().sum() == 0:
        errors.append("anchor_year_group is absent/entirely missing")

    metadata = {
        "script": "prepare_adhf_plan_covariates_v2.py",
        "version": "plan_covariate_gate_no_silent_imputation",
        "cohort": str(cohort_path),
        "base_covariates": str(base_path),
        "supplements": source_files,
        "n_stays": int(len(d)),
        "ef_reference": ef_ref,
        "severity_fields_observed": observed_severity,
        "therapy_fields_observed": observed_therapy,
        "strict_pass": len(errors) == 0,
        "audit_warnings_or_errors": errors,
        "outputs": ["analysis_covariates_plan_v2.csv", "plan_covariate_field_audit_v2.csv", "analysis_covariates_missingness_plan_v2.csv"],
    }
    (out / "RUN_METADATA.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    if errors and not a.allow_missing_plan_fields:
        raise ValueError("Plan covariate gate failed:\n- " + "\n- ".join(errors) +
                         "\nUse --allow-missing-plan-fields only to create an audit while obtaining the missing exports.")
    print(json.dumps(metadata, indent=2, default=str))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
