#!/usr/bin/env python3
"""Strict plan Cox/validation layer for the ADHF trajectory cohort.

This script consumes a mortality table and the outcome-blind temporal GBMT
assignments produced by ``fit_adhf_gbmt_temporal_v2.R``.  It fits the base and
base-plus-phenotype Cox models in 2008--2013, freezes both coefficients and
the development-era design matrix in 2014--2019, and reports the quantities
specified in the plan: phenotype HRs, Schoenfeld PH tests, landmark KM tables,
Harrell C, 85-day AUC, integrated Brier score, calibration data, LR test,
continuous NRI/IDI, decision-curve net benefit, and phenotype-by-EF
interaction.

The default covariate gate is intentionally strict.  The old 21-variable
table is not a plan-complete adjusted model because it has no EF, severity,
admission hemodynamics/rhythm, NT-proBNP/congestion, or chronic HF therapy.
Prepare and audit those fields with ``prepare_adhf_plan_covariates_v2.py``
before invoking this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2

from adhf_survival_utils import (
    HORIZON_DAYS,
    calibration_slope,
    continuous_nri_idi,
    evaluate_predictions,
    fit_cox,
    ipcw_components,
    ipcw_auc_brier,
    require_columns,
    write_json,
)


ID_COLS = ["subject_id", "hadm_id", "stay_id"]
COMORBIDITY = ["diabetes", "ckd", "copd", "atrial_fibrillation", "prior_mi", "hypertension", "valvular_disease", "anemia"]
FIRST24 = ["vasopressor_first24h", "inotrope_first24h", "mechanical_ventilation_first24h", "rrt_first24h"]
THERAPY = ["chronic_hf_loop", "chronic_hf_raas", "chronic_hf_beta_blocker", "chronic_hf_mra", "chronic_hf_sglt2"]
EF_INDICATORS = ["ef_reduced", "ef_midrange", "ef_preserved", "ef_unknown"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mortality-input", required=True, type=Path,
                   help="mortality_input_GE3_K4_v1.csv, augmented with strict-plan covariates")
    p.add_argument("--assignments", required=True, type=Path,
                   help="temporal_gbmt_assignments_*_v2.csv from the R script")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--severity-score", choices=["sofa", "sapsii", "both"], default="sofa")
    p.add_argument("--reference-class", type=int, default=1)
    p.add_argument("--era-column", default="anchor_year_group")
    p.add_argument("--dev-cutoff", type=int, default=2013)
    p.add_argument("--validation-start", type=int, default=2014)
    p.add_argument("--allow-missing-plan-fields", action="store_true",
                   help="Only write the design audit; do not fit models")
    return p.parse_args()


def read_table(path: Path, label: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    d = pd.read_csv(path, low_memory=False)
    if d.empty:
        raise ValueError(f"{label} is empty: {path}")
    return d


def numeric(d: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")


def parse_year(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x.astype("string").str.extract(r"(\d{4})", expand=False), errors="coerce")


def require_unique_ids(d: pd.DataFrame, label: str) -> pd.DataFrame:
    require_columns(d, ["stay_id"], label)
    out = d.copy()
    for c in ID_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="raise").astype("int64")
    out["stay_id"] = pd.to_numeric(out["stay_id"], errors="raise").astype("int64")
    if out["stay_id"].duplicated().any():
        raise ValueError(f"{label} has duplicate stay_id")
    return out


def rhythm_indicators(d: pd.DataFrame, column: str = "admission_rhythm") -> tuple[pd.DataFrame, list[str], str]:
    if column not in d.columns:
        return d, [], "missing"
    x = d[column].astype("string").str.lower().fillna("unknown")
    # Fixed, outcome-blind buckets avoid learning categories from the later
    # era.  A record can belong to one bucket only.
    y = pd.Series("other", index=d.index, dtype="string")
    y[x.str.contains("sinus", na=False)] = "sinus"
    y[x.str.contains("atrial fibr|afib|a-fib|flutter", na=False)] = "af"
    y[x.str.contains("paced|pacemaker", na=False)] = "paced"
    y[x.str.contains("unknown|missing|not recorded", na=False)] = "unknown"
    out = d.copy()
    cols = []
    for cat in ["sinus", "af", "paced", "other", "unknown"]:
        c = f"admission_rhythm_{cat}"
        out[c] = (y == cat).astype(float)
        cols.append(c)
    # sinus is the prespecified clinically interpretable reference when it is
    # observed; retaining the indicator in the table keeps coding auditable.
    return out, cols, "admission_rhythm_sinus"


def ef_design(d: pd.DataFrame, reference: int) -> tuple[pd.DataFrame, list[str], str]:
    out = d.copy()
    if not all(c in out.columns for c in EF_INDICATORS) and "ef_category" in out.columns:
        x = out["ef_category"].astype("string").str.strip().str.lower()
        num = pd.to_numeric(x, errors="coerce")
        x = x.where(x.isin(["reduced", "midrange", "preserved"]), np.nan)
        x = x.where(x.notna(), pd.cut(num, [-np.inf, 39, 49, np.inf], labels=["reduced", "midrange", "preserved"]))
        x = x.astype("string").fillna("unknown")
        for cat in ["reduced", "midrange", "preserved", "unknown"]:
            out[f"ef_{cat}"] = (x == cat).astype(float)
    if not all(c in out.columns for c in EF_INDICATORS):
        raise ValueError(f"Missing EF indicators: {EF_INDICATORS}; run the plan covariate gate first")
    numeric(out, EF_INDICATORS)
    # Use preserved as the default reference if it exists; otherwise use the
    # first observed indicator.  Unknown remains an explicit category.
    ref = "ef_preserved" if "ef_preserved" in out.columns else EF_INDICATORS[0]
    vars_ = [c for c in EF_INDICATORS if c != ref]
    for c in vars_:
        out[c] = out[c].fillna(0.0)
    return out, vars_, ref


def add_temporal_split(d: pd.DataFrame, era_column: str, cutoff: int, start: int) -> pd.DataFrame:
    out = d.copy()
    if era_column not in out.columns:
        raise ValueError(f"No era column {era_column!r}; temporal validation requires anchor_year_group")
    out["era_year"] = parse_year(out[era_column])
    out["temporal_partition"] = np.select(
        [out.era_year.le(cutoff), out.era_year.ge(start)],
        ["development_2008_2013", "validation_2014_2019"], default="outside_requested_eras")
    return out


def base_candidates(d: pd.DataFrame, severity: str) -> tuple[list[str], list[str]]:
    required = ["age_per10", "sex_female", *COMORBIDITY, *FIRST24, "baseline_creatinine", "baseline_ntprobnp",
                "admission_sbp", "admission_hr"]
    missing = [c for c in required if c not in d.columns]
    if not any(c in d.columns for c in EF_INDICATORS) and "ef_category" not in d.columns:
        missing.append("ef indicators (ef_reduced/ef_midrange/ef_preserved/ef_unknown)")
    severity_cols = {"sofa": ["sofa_first24h"], "sapsii": ["sapsii"], "both": ["sofa_first24h", "sapsii"]}[severity]
    missing += [c for c in severity_cols if c not in d.columns]
    observed_therapy = [c for c in THERAPY if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().any()]
    if not observed_therapy and ("chronic_hf_therapy" not in d.columns or pd.to_numeric(d["chronic_hf_therapy"], errors="coerce").notna().sum() == 0):
        missing.append("at least one chronic HF therapy flag (chronic_hf_loop/RAAS/beta-blocker/MRA/SGLT2)")
    return required + severity_cols + observed_therapy, missing


def scale_baseline(d: pd.DataFrame, columns: list[str], dev_mask: pd.Series, outdir: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    out = d.copy(); rows = []; used = []
    for c in columns:
        if c not in out.columns:
            continue
        x = pd.to_numeric(out[c], errors="coerce")
        xd = x.loc[dev_mask & x.notna()]
        if len(xd) < 5:
            raise ValueError(f"Development has fewer than five observed values for {c}")
        # Laboratory/biomarker values are winsorised only in the development
        # partition; this prevents temporal leakage from later values.
        q01, q99 = np.quantile(xd, [0.01, 0.99])
        w = xd.clip(q01, q99)
        mu, sd = float(w.mean()), float(w.std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Development SD is zero/non-finite for {c}")
        zc = f"{c}_z_plan"
        out[zc] = (x.clip(q01, q99) - mu) / sd
        used.append(zc)
        rows.append({"raw_column": c, "scaled_column": zc, "q01_development": q01, "q99_development": q99,
                     "mean_winsorized_development": mu, "sd_winsorized_development": sd, "n_development_observed": len(xd)})
    params = pd.DataFrame(rows)
    params.to_csv(outdir / "plan_baseline_scaling_parameters.csv", index=False)
    return out, used, params


def make_design(raw: pd.DataFrame, severity: str, reference_class: int, dev_mask: pd.Series, outdir: Path) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    d = raw.copy()
    # Numeric fields used directly by the design.
    therapy_vars = [c for c in THERAPY if c in d.columns]
    if not therapy_vars and "chronic_hf_therapy" in d.columns:
        therapy_vars = ["chronic_hf_therapy"]
    continuous = ["age_per10", "baseline_creatinine", "baseline_ntprobnp", "admission_sbp", "admission_hr", *therapy_vars, *COMORBIDITY, *FIRST24]
    severity_cols = {"sofa": ["sofa_first24h"], "sapsii": ["sapsii"], "both": ["sofa_first24h", "sapsii"]}[severity]
    numeric(d, continuous + severity_cols + ["sex_female"])
    d, ef_vars, ef_ref = ef_design(d, reference_class)
    d, rhythm_vars, rhythm_ref = rhythm_indicators(d)
    scaled_raw = ["baseline_creatinine", "baseline_ntprobnp"]
    # z_plan fields are used for all continuous measures in the adjusted
    # model.  Admission vital signs are scaled as well for numerical stability.
    scaled_raw += ["admission_sbp", "admission_hr", *severity_cols]
    d, scaled, _ = scale_baseline(d, scaled_raw, dev_mask, outdir)
    base_vars = ["age_per10", "sex_female", *COMORBIDITY, *FIRST24, *therapy_vars, *scaled]
    ext_vars: list[str] = []
    # The reference class is kept fixed to avoid changing interpretation when
    # a later era has a different modal class prevalence.
    classes = sorted(pd.to_numeric(d["phenotype_class"], errors="coerce").dropna().astype(int).unique().tolist())
    if reference_class not in classes:
        raise ValueError(f"Reference phenotype class {reference_class} not observed; classes={classes}")
    for cls in classes:
        c = f"phenotype_class_{cls}"
        d[c] = (pd.to_numeric(d.phenotype_class, errors="coerce") == cls).astype(float)
        if cls != reference_class:
            ext_vars.append(c)
    # Add EF indicators to both models; EF reference is preserved.
    base_vars += ef_vars + [c for c in rhythm_vars if c != rhythm_ref]
    ext_vars = base_vars + ext_vars
    return d, base_vars, ext_vars, classes, ef_vars


def complete(d: pd.DataFrame, vars_: list[str]) -> pd.DataFrame:
    return d.dropna(subset=["analysis_time", "analysis_event", *vars_]).copy()


def remove_constant_in_development(dev: pd.DataFrame, base_vars: list[str], ext_vars: list[str], classes: list[int], reference: int) -> tuple[list[str], list[str], list[str]]:
    """Drop non-informative clinical columns, but never drop a phenotype class.

    Empty EF/rhythm buckets are expected in real MIMIC extracts.  Keeping a
    zero-variance dummy would make lifelines' Hessian singular, so it is
    removed with an explicit audit entry.  A phenotype class absent from the
    development era is a hard error because a frozen later-era coefficient
    cannot be estimated for it.
    """
    dropped = []
    base_used = []
    for c in base_vars:
        vals = pd.to_numeric(dev[c], errors="coerce").dropna().unique()
        if len(vals) <= 1:
            dropped.append(c)
        else:
            base_used.append(c)
    ph_terms = [c for c in ext_vars if c.startswith("phenotype_class_")]
    for c in ph_terms:
        vals = pd.to_numeric(dev[c], errors="coerce").dropna().unique()
        if len(vals) <= 1:
            cls = c.rsplit("_", 1)[-1]
            raise ValueError(f"Development era has no variation in phenotype class {cls}; cannot transport a frozen model")
    return base_used, base_used + ph_terms, dropped


def frozen_performance(model, data: pd.DataFrame, vars_: list[str]) -> tuple[dict, dict]:
    # The existing utility evaluates an already-fitted model on any data with
    # the same design columns; no refit occurs here.
    return evaluate_predictions(model, data[["analysis_time", "analysis_event", *vars_]], vars_)


def brier_at(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: float) -> float:
    t = np.asarray(time, dtype=float); e = np.asarray(event, dtype=int); r = np.asarray(risk, dtype=float)
    score_time = t.copy(); endpoint_nonevent = (e == 0) & np.isclose(score_time, HORIZON_DAYS); score_time[endpoint_nonevent] = HORIZON_DAYS + 1e-8
    km = ipcw_components(t, e, HORIZON_DAYS)
    # Recompute censoring survival at the requested time using a KM fit.
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter(); kmf.fit(score_time, event_observed=1-e)
    def g_left(x):
        tl = np.asarray(kmf.timeline, dtype=float); sf = np.asarray(kmf.survival_function_.iloc[:,0], dtype=float)
        ix = np.where(tl < x)[0]; return 1.0 if len(ix)==0 else float(sf[ix[-1]])
    def g_at(x):
        tl = np.asarray(kmf.timeline, dtype=float); sf = np.asarray(kmf.survival_function_.iloc[:,0], dtype=float)
        ix = np.where(tl <= x)[0]; return 1.0 if len(ix)==0 else float(sf[ix[-1]])
    cases = (e == 1) & (t <= horizon + 1e-8); controls = score_time > horizon
    censored = (e == 0) & (score_time < horizon - 1e-8)
    if not cases.any() or not controls.any(): return float("nan")
    wc = np.array([1.0 / max(g_left(x), 1e-12) for x in t[cases]])
    gh = max(g_at(horizon), 1e-12); wn = np.full(int(controls.sum()), 1.0 / gh)
    return float((np.sum(wc * (1-r[cases])**2) + np.sum(wn * r[controls]**2)) / len(t))


def integrated_brier(model, data: pd.DataFrame, vars_: list[str], max_time: float = HORIZON_DAYS, n_grid: int = 50) -> tuple[float, pd.DataFrame]:
    t = data.analysis_time.to_numpy(float); e = data.analysis_event.to_numpy(int)
    grid = np.linspace(0.0, max_time, n_grid)
    rows = []
    for tt in grid:
        sf = model.predict_survival_function(data[vars_], times=[float(tt)])
        risk = 1.0 - sf.T.iloc[:,0].to_numpy(float)
        rows.append({"time_from_landmark_days": float(tt), "brier": brier_at(t, e, risk, float(tt)), "n": len(data), "events": int(e.sum())})
    curve = pd.DataFrame(rows).dropna(subset=["brier"])
    if len(curve) < 2: return float("nan"), curve
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(curve.brier, curve.time_from_landmark_days) / max_time), curve


def calibration_table(model, data: pd.DataFrame, vars_: list[str], partition: str, n_bins: int = 10) -> pd.DataFrame:
    sf = model.predict_survival_function(data[vars_], times=[HORIZON_DAYS])
    pred = np.clip(1.0 - sf.T.iloc[:,0].to_numpy(float), 1e-8, 1-1e-8)
    x = data[["analysis_time", "analysis_event"]].copy(); x["predicted_risk_85d"] = pred
    # qcut can produce fewer bins when ties are abundant; duplicates=drop is
    # deterministic and the output records the actual bin count.
    x["risk_decile"] = pd.qcut(x.predicted_risk_85d, q=min(n_bins, x.predicted_risk_85d.nunique()), labels=False, duplicates="drop") + 1
    from lifelines import KaplanMeierFitter
    rows = []
    for b, g in x.groupby("risk_decile", dropna=True):
        km = KaplanMeierFitter(); tt = g.analysis_time.to_numpy(float); ee = g.analysis_event.to_numpy(int)
        km.fit(tt, event_observed=ee)
        try: observed = float(1.0 - km.predict(HORIZON_DAYS))
        except Exception: observed = float("nan")
        rows.append({"partition": partition, "risk_decile": int(b), "n": len(g), "events": int(g.analysis_event.sum()),
                     "mean_predicted_risk_85d": float(g.predicted_risk_85d.mean()), "observed_KM_risk_85d": observed,
                     "calibration_difference": observed - float(g.predicted_risk_85d.mean())})
    return pd.DataFrame(rows)


def ph_table(model, data: pd.DataFrame, vars_: list[str], model_name: str, partition: str) -> pd.DataFrame:
    from lifelines.statistics import proportional_hazard_test
    try:
        test = proportional_hazard_test(model, data[["analysis_time", "analysis_event", *vars_]], time_transform="rank")
        out = test.summary.reset_index().rename(columns={"index": "term"})
        if "term" not in out.columns: out = out.rename(columns={out.columns[0]: "term"})
        pcol = next((c for c in ["p", "p_value"] if c in out.columns), None)
        if pcol and pcol != "p_value": out = out.rename(columns={pcol: "p_value"})
        out["model"], out["partition"] = model_name, partition
        return out
    except Exception as exc:
        return pd.DataFrame([{"term": "__PH_TEST_ERROR__", "p_value": np.nan, "error": str(exc), "model": model_name, "partition": partition}])


def km_by_class(data: pd.DataFrame, partition: str) -> pd.DataFrame:
    from lifelines import KaplanMeierFitter
    rows = []
    for cls, g in data.groupby("phenotype_class"):
        km = KaplanMeierFitter(); km.fit(g.analysis_time, event_observed=g.analysis_event, label=f"class_{int(cls)}")
        for tt in [30.0, 60.0, 85.0]:
            rows.append({"partition": partition, "phenotype_class": int(cls), "time_from_landmark_days": tt,
                         "n": len(g), "events": int(g.analysis_event.sum()), "survival": float(km.predict(tt)),
                         "mortality": float(1-km.predict(tt))})
    return pd.DataFrame(rows)


def decision_curve(time: np.ndarray, event: np.ndarray, risk: np.ndarray, model: str, partition: str) -> pd.DataFrame:
    cpt = ipcw_components(time, event, HORIZON_DAYS)
    status, w = cpt["status"], cpt["weight"]
    grid = np.linspace(0.05, 0.80, 16)
    rows = []
    for p in grid:
        high = risk >= p
        # IPCW net benefit: weighted true positives/n minus weighted false
        # positives/n multiplied by odds of the threshold.
        tp = np.sum(w[(status == 1) & high]); fp = np.sum(w[(status == 0) & high])
        nb = (tp - fp * p / (1-p)) / len(risk)
        rows.append({"partition": partition, "model": model, "threshold": p, "net_benefit": float(nb), "n": len(risk), "events": int(event.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    a = parse_args()
    out = a.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    d = require_unique_ids(read_table(a.mortality_input, "mortality input"), "mortality input")
    ass = require_unique_ids(read_table(a.assignments, "temporal assignments"), "temporal assignments")
    require_columns(d, ["analysis_time", "analysis_event", "day5_landmark_eligible"], "mortality input")
    require_columns(ass, ["partition", "phenotype_class", "max_posterior"], "temporal assignments")
    # Keep outcome rows exactly once and use the GBMT assignment generated
    # without outcomes.
    d = d.merge(ass[["stay_id", "partition", "phenotype_class", "max_posterior"] + [c for c in ass.columns if c.startswith("posterior_class_")]], on="stay_id", how="inner", validate="one_to_one")
    d = d.loc[pd.to_numeric(d.day5_landmark_eligible, errors="coerce").eq(1)].copy()
    numeric(d, ["analysis_time", "analysis_event", "phenotype_class", "max_posterior"])
    if d.analysis_time.isna().any() or d.analysis_event.isna().any(): raise ValueError("Eligible outcome rows have missing analysis_time/event")
    d["analysis_event"] = d.analysis_event.astype(int)
    if not set(d.analysis_event.unique()).issubset({0,1}): raise ValueError("analysis_event must be binary")
    d = add_temporal_split(d, a.era_column, a.dev_cutoff, a.validation_start)
    # The partition written by R is authoritative; compare it to the era
    # split and fail on any mismatch rather than silently transporting rows.
    mismatch = d.partition.ne(d.temporal_partition) & d.temporal_partition.ne("outside_requested_eras")
    if mismatch.any(): raise ValueError("GBMT assignment partition disagrees with era split")
    d["temporal_partition"] = d["partition"]
    dev_mask = d.temporal_partition.eq("development_2008_2013")
    val_mask = d.temporal_partition.eq("validation_2014_2019")
    requested, missing = base_candidates(d, a.severity_score)
    # Admission rhythm is required by the plan but represented categorically.
    if "admission_rhythm" not in d.columns: missing.append("admission_rhythm")
    audit = pd.DataFrame({"field": sorted(set(missing)), "missing": 1}) if missing else pd.DataFrame(columns=["field","missing"])
    audit.to_csv(out / "plan_survival_missing_fields.csv", index=False)
    if missing:
        meta = {"script": "validate_adhf_plan_survival_v2.py", "strict_plan_pass": False, "missing_fields": sorted(set(missing)), "severity_score": a.severity_score}
        write_json(out / "RUN_METADATA.json", meta)
        if not a.allow_missing_plan_fields:
            raise ValueError("Strict plan survival gate failed; missing:\n- " + "\n- ".join(sorted(set(missing))))
        print(json.dumps(meta, indent=2)); return

    d, base_vars_requested, ext_vars_requested, classes, ef_vars = make_design(d, a.severity_score, a.reference_class, dev_mask, out)
    # Complete cases are defined before dropping zero-variance dummies so the
    # missingness denominator is transparent; the dropped-column audit is
    # written below.
    dev_all = complete(d.loc[dev_mask].copy(), ext_vars_requested)
    val_all = complete(d.loc[val_mask].copy(), ext_vars_requested)
    base_vars, ext_vars, dropped_constant = remove_constant_in_development(dev_all, base_vars_requested, ext_vars_requested, classes, a.reference_class)
    dev = dev_all.dropna(subset=["analysis_time", "analysis_event", *ext_vars]).copy()
    val = val_all.dropna(subset=["analysis_time", "analysis_event", *ext_vars]).copy()
    pd.DataFrame({"variable": dropped_constant, "reason": "constant_or_zero_variance_in_development"}).to_csv(out / "plan_constant_covariates_removed.csv", index=False)
    if len(dev) < 50 or len(val) < 50 or dev.analysis_event.sum() < 5 or val.analysis_event.sum() < 5:
        raise ValueError(f"Temporal complete-case samples too small: development n={len(dev)}, events={int(dev.analysis_event.sum())}; validation n={len(val)}, events={int(val.analysis_event.sum())}")

    base_fit, dev_base_df = fit_cox(dev, base_vars)
    ext_fit, dev_ext_df = fit_cox(dev, ext_vars)
    bp_dev, bpred_dev = evaluate_predictions(base_fit, dev_base_df, base_vars)
    ep_dev, epred_dev = evaluate_predictions(ext_fit, dev_ext_df, ext_vars)
    bp_val, bpred_val = frozen_performance(base_fit, val, base_vars)
    ep_val, epred_val = frozen_performance(ext_fit, val, ext_vars)

    rows = []
    for partition, b, e in [("development_2008_2013", bp_dev, ep_dev), ("validation_2014_2019", bp_val, ep_val)]:
        rows += [{"partition": partition, "model": "Base clinical model", **b}, {"partition": partition, "model": "Base + phenotype", **e},
                 {"partition": partition, "model": "Difference (extended - base)", "n": e["n"], "events": e["events"], "Harrell_C": e["Harrell_C"]-b["Harrell_C"], "IPCW_AUC85": e["IPCW_AUC85"]-b["IPCW_AUC85"], "IPCW_Brier85": e["IPCW_Brier85"]-b["IPCW_Brier85"], "Cox_linear_predictor_calibration_slope": e["Cox_linear_predictor_calibration_slope"]-b["Cox_linear_predictor_calibration_slope"]}]
    pd.DataFrame(rows).to_csv(out / "plan_temporal_performance.csv", index=False)

    # LR test and continuous reclassification in development and transport.
    lr = 2.0 * (float(ext_fit.log_likelihood_) - float(base_fit.log_likelihood_)); lr_df = len(ext_vars)-len(base_vars)
    pd.DataFrame([{"partition": "development_2008_2013", "LR_statistic": lr, "df": lr_df, "p_value": chi2.sf(lr, lr_df), "n": len(dev), "events": int(dev.analysis_event.sum())}]).to_csv(out / "plan_lr_test_development.csv", index=False)
    nri_rows = []
    for part, bpred, epred in [("development_2008_2013", bpred_dev, epred_dev), ("validation_2014_2019", bpred_val, epred_val)]:
        z = continuous_nri_idi(bpred["time"], bpred["event"], bpred["risk"], epred["risk"])
        nri_rows.append({"partition": part, **z})
    pd.DataFrame(nri_rows).to_csv(out / "plan_continuous_NRI_IDI_85d.csv", index=False)

    # Schoenfeld tests use the development fit only.  Validation is a frozen
    # transport check, so a new PH test would be a different estimand.
    ph = pd.concat([ph_table(base_fit, dev_base_df, base_vars, "Base clinical model", "development_2008_2013"), ph_table(ext_fit, dev_ext_df, ext_vars, "Base + phenotype", "development_2008_2013")], ignore_index=True)
    ph.to_csv(out / "plan_schoenfeld_PH_tests.csv", index=False)

    cal = pd.concat([calibration_table(base_fit, dev_base_df, base_vars, "development_2008_2013"), calibration_table(ext_fit, dev_ext_df, ext_vars, "development_2008_2013"), calibration_table(base_fit, val, base_vars, "validation_2014_2019"), calibration_table(ext_fit, val, ext_vars, "validation_2014_2019")], ignore_index=True)
    cal.to_csv(out / "plan_calibration_85d.csv", index=False)
    ibs_rows, ibs_curves = [], []
    for part, model_name, fit, data, vars_ in [("development_2008_2013","Base clinical model",base_fit,dev_base_df,base_vars),("development_2008_2013","Base + phenotype",ext_fit,dev_ext_df,ext_vars),("validation_2014_2019","Base clinical model",base_fit,val,base_vars),("validation_2014_2019","Base + phenotype",ext_fit,val,ext_vars)]:
        ibs, curve = integrated_brier(fit, data, vars_); ibs_rows.append({"partition":part,"model":model_name,"integrated_Brier_0_85d":ibs,"n":len(data),"events":int(data.analysis_event.sum())}); curve["partition"], curve["model"] = part, model_name; ibs_curves.append(curve)
    pd.DataFrame(ibs_rows).to_csv(out / "plan_integrated_Brier_0_85d.csv", index=False)
    pd.concat(ibs_curves, ignore_index=True).to_csv(out / "plan_Brier_curve_0_85d.csv", index=False)

    dca = pd.concat([decision_curve(bpred_dev["time"], bpred_dev["event"], bpred_dev["risk"], "Base clinical model", "development_2008_2013"), decision_curve(epred_dev["time"], epred_dev["event"], epred_dev["risk"], "Base + phenotype", "development_2008_2013"), decision_curve(bpred_val["time"], bpred_val["event"], bpred_val["risk"], "Base clinical model", "validation_2014_2019"), decision_curve(epred_val["time"], epred_val["event"], epred_val["risk"], "Base + phenotype", "validation_2014_2019")], ignore_index=True)
    dca.to_csv(out / "plan_decision_curve_net_benefit_85d.csv", index=False)

    # Landmark survival summaries and phenotype HRs.
    km = pd.concat([km_by_class(dev, "development_2008_2013"), km_by_class(val, "validation_2014_2019")], ignore_index=True)
    km.to_csv(out / "plan_landmark_KM_by_phenotype.csv", index=False)
    ext_fit.summary.to_csv(out / "plan_extended_cox_summary.csv")
    h = ext_fit.summary.reset_index().rename(columns={"covariate":"term"})
    if "term" not in h.columns: h = h.rename(columns={h.columns[0]:"term"})
    h[h.term.str.startswith("phenotype_class_")].assign(reference_class=a.reference_class).to_csv(out / "plan_phenotype_HR_development.csv", index=False)

    # Phenotype x EF interaction, fitted only in development.  The joint LR
    # test is the prespecified effect-modification test.
    int_vars = list(ext_vars); interaction_terms = []
    ph_terms = [v for v in ext_vars if v.startswith("phenotype_class_")]
    for pterm in ph_terms:
        for ef in ef_vars:
            t = f"{pterm}__x__{ef}"; dev[t] = dev[pterm] * dev[ef]; interaction_terms.append(t)
    int_fit = None
    if interaction_terms:
        # Empty phenotype-by-EF cells can make the unpenalised Cox Hessian
        # singular.  Keep the prespecified interaction attempt auditable and
        # report the failure instead of aborting all primary validation files.
        try:
            int_fit, _ = fit_cox(dev, ext_vars + interaction_terms)
            stat = 2.0 * (float(int_fit.log_likelihood_) - float(ext_fit.log_likelihood_)); df_int = len(interaction_terms)
            pd.DataFrame([{"LR_statistic":stat,"df":df_int,"p_value":chi2.sf(stat,df_int),"n":len(dev),"events":int(dev.analysis_event.sum()),"interaction_terms":";".join(interaction_terms),"fit_error":""}]).to_csv(out / "plan_phenotype_EF_interaction_LR.csv", index=False)
            s = int_fit.summary.reset_index().rename(columns={"covariate":"term"});
            if "term" not in s.columns: s=s.rename(columns={s.columns[0]:"term"})
            s[s.term.isin(interaction_terms)].to_csv(out / "plan_phenotype_EF_interaction_coefficients.csv", index=False)
        except Exception as exc:
            pd.DataFrame([{"LR_statistic":np.nan,"df":len(interaction_terms),"p_value":np.nan,"n":len(dev),"events":int(dev.analysis_event.sum()),"interaction_terms":";".join(interaction_terms),"fit_error":str(exc)}]).to_csv(out / "plan_phenotype_EF_interaction_LR.csv", index=False)
            pd.DataFrame(columns=["term","fit_error"]).to_csv(out / "plan_phenotype_EF_interaction_coefficients.csv", index=False)

    # Save complete-case datasets and model summaries for reproducibility.
    dev.to_csv(out / "plan_development_complete_case.csv", index=False); val.to_csv(out / "plan_validation_complete_case.csv", index=False)
    base_fit.summary.to_csv(out / "plan_base_cox_summary.csv")
    write_json(out / "RUN_METADATA.json", {
        "script":"validate_adhf_plan_survival_v2.py", "version":"strict_plan_temporal_frozen_cox_day5_landmark",
        "mortality_input":str(a.mortality_input.expanduser().resolve()), "assignments":str(a.assignments.expanduser().resolve()),
        "severity_score":a.severity_score, "reference_class":a.reference_class,
        "development_n":len(dev), "development_events":int(dev.analysis_event.sum()), "validation_n":len(val), "validation_events":int(val.analysis_event.sum()),
        "development_rule":f"era_year <= {a.dev_cutoff}", "validation_rule":f"era_year >= {a.validation_start}",
        "landmark_origin":"ICU day 5", "horizon_days_from_landmark":HORIZON_DAYS,
        "phenotype_assignment_frozen":True, "gbmt_refit_validation":False,
        "scaling":"continuous baseline variables winsorized/z-scored using development partition only",
        "base_covariates":base_vars, "extended_covariates":ext_vars,
        "plan_strict_pass":True,
    })
    print(pd.DataFrame(rows).to_string(index=False)); print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
