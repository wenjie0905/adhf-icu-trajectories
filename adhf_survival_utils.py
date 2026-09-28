#!/usr/bin/env python3
"""Shared utilities for the post-cohort-change ADHF mortality analyses.

The utilities deliberately use the day-5 landmark scale: time is measured from
ICU day 5 and the 90-day-from-ICU endpoint is therefore an 85-day horizon.
Phenotype assignment is treated as fixed in the Cox/validation layer; GBMT is
never refit by these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2


HORIZON_DAYS = 85.0
ID_COLS = ["subject_id", "hadm_id", "stay_id"]


def require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_ids(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    require_columns(out, ID_COLS, label)
    for col in ID_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")
    if out[ID_COLS].isna().any().any():
        raise ValueError(f"{label} contains missing identifiers")
    return out


def key_for(df: pd.DataFrame) -> pd.Series:
    return (
        df["subject_id"].astype(str)
        + "|"
        + df["hadm_id"].astype(str)
        + "|"
        + df["stay_id"].astype(str)
    )


def chi_square_p(statistic: float, df: int) -> float:
    if not np.isfinite(statistic) or df <= 0:
        return float("nan")
    return float(chi2.sf(statistic, df))


def censoring_km(time: np.ndarray, event: np.ndarray):
    """Kaplan–Meier estimate of the censoring survival function."""
    from lifelines import KaplanMeierFitter

    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    score_time = t.copy()
    # Administrative non-events at the horizon are not censoring events.
    endpoint_nonevent = (e == 0) & np.isclose(score_time, HORIZON_DAYS)
    score_time[endpoint_nonevent] = HORIZON_DAYS + 1e-8
    km = KaplanMeierFitter()
    km.fit(score_time, event_observed=1 - e)
    return km


def g_left(km, value: float) -> float:
    timeline = np.asarray(km.timeline, dtype=float)
    survival = np.asarray(km.survival_function_[km.survival_function_.columns[0]], dtype=float)
    if len(timeline) == 0:
        return 1.0
    idx = np.where(timeline < value)[0]
    return 1.0 if len(idx) == 0 else float(survival[idx[-1]])


def g_at(km, value: float) -> float:
    timeline = np.asarray(km.timeline, dtype=float)
    survival = np.asarray(km.survival_function_[km.survival_function_.columns[0]], dtype=float)
    if len(timeline) == 0:
        return 1.0
    idx = np.where(timeline <= value)[0]
    return 1.0 if len(idx) == 0 else float(survival[idx[-1]])


def ipcw_components(time: np.ndarray, event: np.ndarray, horizon: float = HORIZON_DAYS):
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(e)):
        raise ValueError("IPCW input contains non-finite time/event values")
    if not set(np.unique(e)).issubset({0, 1}):
        raise ValueError("Event must be binary")
    score_time = t.copy()
    endpoint_nonevent = (e == 0) & np.isclose(score_time, horizon)
    score_time[endpoint_nonevent] = horizon + 1e-8
    km = censoring_km(t, e)
    cases = (e == 1) & (t <= horizon + 1e-8)
    controls = score_time > horizon
    early_censored = (e == 0) & (score_time < horizon - 1e-8)
    if not cases.any() or not controls.any():
        raise ValueError("No cases or controls at the 85-day horizon")
    g_case = np.asarray([g_left(km, x) for x in t[cases]], dtype=float)
    g_horizon = float(g_at(km, horizon))
    if np.any(g_case <= 0) or not np.isfinite(g_horizon) or g_horizon <= 0:
        raise ValueError("Censoring survival reached zero before the horizon")
    weight = np.zeros(len(t), dtype=float)
    status = np.full(len(t), np.nan, dtype=float)
    weight[cases] = 1.0 / g_case
    weight[controls] = 1.0 / g_horizon
    status[cases] = 1.0
    status[controls] = 0.0
    return {
        "cases": cases,
        "controls": controls,
        "early_censored": early_censored,
        "status": status,
        "weight": weight,
        "g_horizon": g_horizon,
        "max_case_weight": float(np.max(1.0 / g_case)),
        "control_weight": float(1.0 / g_horizon),
        "endpoint_nonevents_shifted": int(endpoint_nonevent.sum()),
    }


def weighted_pair_auc(case_risk, case_weight, control_risk, control_weight) -> float:
    case_risk = np.asarray(case_risk, dtype=float)
    case_weight = np.asarray(case_weight, dtype=float)
    control_risk = np.asarray(control_risk, dtype=float)
    control_weight = np.asarray(control_weight, dtype=float)
    denominator = case_weight.sum() * control_weight.sum()
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    order = np.argsort(control_risk, kind="mergesort")
    sr = control_risk[order]
    sw = control_weight[order]
    cumulative = np.concatenate([[0.0], np.cumsum(sw)])
    numerator = 0.0
    for risk, weight in zip(case_risk, case_weight):
        left = np.searchsorted(sr, risk, side="left")
        right = np.searchsorted(sr, risk, side="right")
        numerator += weight * (cumulative[left] + 0.5 * (cumulative[right] - cumulative[left]))
    return float(numerator / denominator)


def ipcw_auc_brier(time, event, risk, horizon: float = HORIZON_DAYS):
    risk = np.clip(np.asarray(risk, dtype=float), 1e-8, 1 - 1e-8)
    cpt = ipcw_components(time, event, horizon)
    cases = cpt["cases"]
    controls = cpt["controls"]
    weights = cpt["weight"]
    auc = weighted_pair_auc(
        risk[cases], weights[cases], risk[controls], weights[controls]
    )
    brier = float(
        (np.sum(weights[cases] * (1.0 - risk[cases]) ** 2)
         + np.sum(weights[controls] * risk[controls] ** 2)) / len(risk)
    )
    return {
        "IPCW_AUC85": auc,
        "IPCW_Brier85": brier,
        "n_cases_at_85": int(cases.sum()),
        "n_controls_at_85": int(controls.sum()),
        "early_censored_non_events": int(cpt["early_censored"].sum()),
        "G_hat_85": cpt["g_horizon"],
        "maximum_case_IPCW": cpt["max_case_weight"],
        "control_IPCW": cpt["control_weight"],
    }


def continuous_nri_idi(time, event, risk_base, risk_extended, horizon: float = HORIZON_DAYS):
    """IPCW continuous NRI and IDI at the prespecified horizon.

    The estimates are descriptive reclassification summaries; no category
    cut-points are introduced. Higher risk is favorable for cases and lower
    risk is favorable for controls.
    """
    rb = np.asarray(risk_base, dtype=float)
    re = np.asarray(risk_extended, dtype=float)
    cpt = ipcw_components(time, event, horizon)
    cases, controls, w = cpt["cases"], cpt["controls"], cpt["weight"]
    wc = w[cases]
    wn = w[controls]
    d_cases = re[cases] - rb[cases]
    d_controls = re[controls] - rb[controls]
    case_up = float(np.sum(wc * (d_cases > 0)) / np.sum(wc))
    case_down = float(np.sum(wc * (d_cases < 0)) / np.sum(wc))
    control_down = float(np.sum(wn * (d_controls < 0)) / np.sum(wn))
    control_up = float(np.sum(wn * (d_controls > 0)) / np.sum(wn))
    case_mean_delta = float(np.sum(wc * d_cases) / np.sum(wc))
    control_mean_delta = float(np.sum(wn * d_controls) / np.sum(wn))
    return {
        "continuous_NRI85": (case_up - case_down) + (control_down - control_up),
        "IDI85": case_mean_delta - control_mean_delta,
        "case_up": case_up,
        "case_down": case_down,
        "control_down": control_down,
        "control_up": control_up,
        "case_mean_risk_delta": case_mean_delta,
        "control_mean_risk_delta": control_mean_delta,
    }


def fit_cox(df: pd.DataFrame, variables: list[str], robust: bool = True):
    from lifelines import CoxPHFitter

    cols = ["analysis_time", "analysis_event", *variables]
    require_columns(df, cols, "Cox data")
    model_df = df[cols].copy()
    model_df["analysis_time"] = pd.to_numeric(model_df["analysis_time"], errors="coerce")
    model_df["analysis_event"] = pd.to_numeric(model_df["analysis_event"], errors="coerce").astype("Int64")
    model_df = model_df.dropna().copy()
    model_df["analysis_event"] = model_df["analysis_event"].astype(int)
    if len(model_df) == 0 or model_df["analysis_event"].sum() == 0:
        raise ValueError("Cox model has no complete rows or no events")
    if (model_df["analysis_time"] < 0).any() or (model_df["analysis_time"] > HORIZON_DAYS + 1e-8).any():
        raise ValueError("analysis_time must be on the 0–85 day landmark scale")
    model = CoxPHFitter()
    model.fit(
        model_df,
        duration_col="analysis_time",
        event_col="analysis_event",
        robust=robust,
        show_progress=False,
    )
    return model, model_df


def predictions(model, model_df: pd.DataFrame, variables: list[str], horizon: float = HORIZON_DAYS):
    from lifelines.utils import concordance_index

    lp = np.log(np.clip(model.predict_partial_hazard(model_df[variables]).to_numpy(dtype=float).reshape(-1), 1e-300, None))
    sf = model.predict_survival_function(model_df[variables], times=[horizon])
    risk = 1.0 - sf.T.iloc[:, 0].to_numpy(dtype=float)
    t = model_df["analysis_time"].to_numpy(dtype=float)
    e = model_df["analysis_event"].to_numpy(dtype=int)
    cindex = float(concordance_index(t, -lp, e))
    return {"lp": lp, "risk": risk, "cindex": cindex, "time": t, "event": e}


def calibration_slope(time, event, lp) -> float:
    from lifelines import CoxPHFitter

    d = pd.DataFrame({"analysis_time": time, "analysis_event": event, "lp": lp})
    try:
        fit = CoxPHFitter().fit(d, "analysis_time", "analysis_event", robust=True)
        return float(fit.params_["lp"])
    except Exception:
        return float("nan")


def evaluate_predictions(model, model_df: pd.DataFrame, variables: list[str]):
    pred = predictions(model, model_df, variables)
    ipcw = ipcw_auc_brier(pred["time"], pred["event"], pred["risk"])
    out = {
        "n": int(len(model_df)),
        "events": int(pred["event"].sum()),
        "Harrell_C": pred["cindex"],
        **ipcw,
        "Cox_linear_predictor_calibration_slope": calibration_slope(pred["time"], pred["event"], pred["lp"]),
    }
    return out, pred


def model_risk_at_horizon(model, df: pd.DataFrame, variables: list[str], horizon: float = HORIZON_DAYS):
    sf = model.predict_survival_function(df[variables], times=[horizon])
    return 1.0 - sf.T.iloc[:, 0].to_numpy(dtype=float)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

