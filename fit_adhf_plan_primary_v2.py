#!/usr/bin/env python3
"""Fit the plan-complete primary day-5 landmark Cox models.

This is the non-temporal counterpart of ``validate_adhf_plan_survival_v2.py``:
the strict 1,458-stay cohort is analysed with the pooled outcome-blind GBMT
assignment (GE3-K4 in the current plan), while the base clinical model
contains EF, admission severity, hemodynamics/rhythm, NT-proBNP/congestion,
comorbidity, first-24-hour support, and chronic HF therapy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

from adhf_survival_utils import continuous_nri_idi, evaluate_predictions, fit_cox, require_columns, write_json
from validate_adhf_plan_survival_v2 import (
    base_candidates, complete, decision_curve, ef_design, integrated_brier,
    calibration_table, km_by_class, make_design, ph_table, remove_constant_in_development,
    require_unique_ids, read_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="plan-complete mortality_input_plan_v2.csv")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--severity-score", choices=["sofa", "sapsii", "both"], default="sofa")
    p.add_argument("--reference-class", type=int, default=1)
    return p.parse_args()


def main() -> None:
    a = parse_args(); out = a.output_dir.expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    d = require_unique_ids(read_table(a.input, "mortality input"), "mortality input")
    require_columns(d, ["phenotype_class", "day5_landmark_eligible", "analysis_time", "analysis_event"], "mortality input")
    d = d.loc[pd.to_numeric(d.day5_landmark_eligible, errors="coerce").eq(1)].copy()
    for c in ["phenotype_class", "analysis_time", "analysis_event"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    requested, missing = base_candidates(d, a.severity_score)
    if "admission_rhythm" not in d.columns: missing.append("admission_rhythm")
    if missing: raise ValueError("Strict plan primary gate failed; missing: " + ", ".join(sorted(set(missing))))
    all_mask = pd.Series(True, index=d.index)
    d, base_req, ext_req, classes, ef_vars = make_design(d, a.severity_score, a.reference_class, all_mask, out)
    all_complete = complete(d, ext_req)
    base_vars, ext_vars, dropped = remove_constant_in_development(all_complete, base_req, ext_req, classes, a.reference_class)
    analysis = all_complete.dropna(subset=["analysis_time", "analysis_event", *ext_vars]).reset_index(drop=True)
    pd.DataFrame({"variable": dropped, "reason": "constant_or_zero_variance"}).to_csv(out / "plan_constant_covariates_removed.csv", index=False)
    if len(analysis) < 50 or analysis.analysis_event.sum() < 5: raise ValueError("Plan primary complete-case sample is too small")
    bf, bdf = fit_cox(analysis, base_vars); ef, edf = fit_cox(analysis, ext_vars)
    bp, bpred = evaluate_predictions(bf, bdf, base_vars); ep, epred = evaluate_predictions(ef, edf, ext_vars)
    rows = [{"model":"Base clinical model",**bp},{"model":"Base + phenotype",**ep},{"model":"Difference (extended - base)","n":len(analysis),"events":int(analysis.analysis_event.sum()),"Harrell_C":ep["Harrell_C"]-bp["Harrell_C"],"IPCW_AUC85":ep["IPCW_AUC85"]-bp["IPCW_AUC85"],"IPCW_Brier85":ep["IPCW_Brier85"]-bp["IPCW_Brier85"],"Cox_linear_predictor_calibration_slope":ep["Cox_linear_predictor_calibration_slope"]-bp["Cox_linear_predictor_calibration_slope"]}]
    pd.DataFrame(rows).to_csv(out / "plan_primary_performance.csv", index=False)
    lr = 2*(float(ef.log_likelihood_)-float(bf.log_likelihood_)); df_lr=len(ext_vars)-len(base_vars)
    pd.DataFrame([{"LR_statistic":lr,"df":df_lr,"p_value":chi2.sf(lr,df_lr),"n":len(analysis),"events":int(analysis.analysis_event.sum())}]).to_csv(out/"plan_primary_LR_test.csv",index=False)
    pd.DataFrame([continuous_nri_idi(bpred["time"],bpred["event"],bpred["risk"],epred["risk"]) | {"n":len(analysis),"events":int(analysis.analysis_event.sum())}]).to_csv(out/"plan_primary_continuous_NRI_IDI_85d.csv",index=False)
    ph_table(ef, edf, ext_vars, "Base + phenotype", "primary").to_csv(out/"plan_primary_schoenfeld_PH_tests.csv",index=False)
    pd.concat([calibration_table(bf,bdf,base_vars,"primary"),calibration_table(ef,edf,ext_vars,"primary")],ignore_index=True).to_csv(out/"plan_primary_calibration_85d.csv",index=False)
    ibs_rows=[]; curves=[]
    for name,fit,dat,vars_ in [("Base clinical model",bf,bdf,base_vars),("Base + phenotype",ef,edf,ext_vars)]:
        ibs,curve=integrated_brier(fit,dat,vars_); ibs_rows.append({"model":name,"integrated_Brier_0_85d":ibs,"n":len(dat),"events":int(dat.analysis_event.sum())}); curve["model"]=name; curves.append(curve)
    pd.DataFrame(ibs_rows).to_csv(out/"plan_primary_integrated_Brier_0_85d.csv",index=False); pd.concat(curves,ignore_index=True).to_csv(out/"plan_primary_Brier_curve_0_85d.csv",index=False)
    pd.concat([decision_curve(bpred["time"],bpred["event"],bpred["risk"],"Base clinical model","primary"),decision_curve(epred["time"],epred["event"],epred["risk"],"Base + phenotype","primary")],ignore_index=True).to_csv(out/"plan_primary_decision_curve_85d.csv",index=False)
    km_by_class(analysis,"primary").to_csv(out/"plan_primary_KM_by_phenotype.csv",index=False)
    bf.summary.to_csv(out/"plan_primary_base_cox_summary.csv"); ef.summary.to_csv(out/"plan_primary_extended_cox_summary.csv")
    h=ef.summary.reset_index().rename(columns={"covariate":"term"});
    if "term" not in h.columns: h=h.rename(columns={h.columns[0]:"term"})
    h[h.term.str.startswith("phenotype_class_")].assign(reference_class=a.reference_class).to_csv(out/"plan_primary_phenotype_HR.csv",index=False)
    # Interaction with EF category.
    terms=[]; ph_terms=[v for v in ext_vars if v.startswith("phenotype_class_")]
    for pterm in ph_terms:
        for efv in ef_vars:
            name=f"{pterm}__x__{efv}"; analysis[name]=analysis[pterm]*analysis[efv]; terms.append(name)
    try:
        int_fit,_=fit_cox(analysis,ext_vars+terms); stat=2*(float(int_fit.log_likelihood_)-float(ef.log_likelihood_)); pd.DataFrame([{"LR_statistic":stat,"df":len(terms),"p_value":chi2.sf(stat,len(terms)),"n":len(analysis),"events":int(analysis.analysis_event.sum()),"fit_error":""}]).to_csv(out/"plan_primary_phenotype_EF_interaction_LR.csv",index=False)
    except Exception as exc:
        pd.DataFrame([{"LR_statistic":np.nan,"df":len(terms),"p_value":np.nan,"n":len(analysis),"events":int(analysis.analysis_event.sum()),"fit_error":str(exc)}]).to_csv(out/"plan_primary_phenotype_EF_interaction_LR.csv",index=False)
    analysis.to_csv(out/"plan_primary_complete_case.csv",index=False)
    write_json(out/"RUN_METADATA.json",{"script":"fit_adhf_plan_primary_v2.py","version":"strict_plan_primary_day5_landmark","input":str(a.input.expanduser().resolve()),"severity_score":a.severity_score,"n_landmark_eligible":int(len(d)),"n_complete_case":len(analysis),"events":int(analysis.analysis_event.sum()),"horizon_days_from_landmark":85,"base_covariates_used":base_vars,"extended_covariates":ext_vars,"phenotype_assignment_frozen":True,"gbmt_refit":False,"no_imputation":True})
    print(pd.DataFrame(rows).to_string(index=False)); print(f"Outputs: {out}")


if __name__ == "__main__": main()
