#!/usr/bin/env python3
"""Build the plan-complete day-5 landmark mortality-analysis table.

The preferred direct mode joins three outcome-blind sources from the current
pipeline: the strict cohort, day-5 landmark outcomes, and frozen GBMT modal
assignments. Plan covariates are then joined by ``stay_id``. Time is placed on
the landmark scale (90 days from ICU admission = 85 days after day 5).

For reproducibility with an already completed run, the script also accepts a
previously prepared mortality input through ``--mortality-input``. In either
mode it performs one-to-one identifier checks and never imputes covariates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = ["subject_id", "hadm_id", "stay_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cohort",
        type=Path,
        help="Preferred direct mode: strict cohort from rebuild_adhf_cohort_v4.py",
    )
    source.add_argument(
        "--mortality-input",
        type=Path,
        help="Compatibility mode: an existing one-row-per-stay mortality input",
    )
    p.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Direct mode: directory containing landmark_outcomes_v1.csv",
    )
    p.add_argument(
        "--gbmt-dir",
        type=Path,
        help="Direct mode: directory containing <SET>/gbmt_<SET>_K<K>_assignments.csv",
    )
    p.add_argument("--plan-covariates", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model-set", choices=["GE2", "GE3", "FULL5"], default="GE3")
    p.add_argument("--k", type=int, default=4)
    p.add_argument(
        "--assignments",
        type=Path,
        default=None,
        help="Compatibility mode only: optional frozen assignment file that replaces an old assignment",
    )
    return p.parse_args()


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read(path: Path, label: str, *, unique_stay: bool = True) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = pd.read_csv(path, low_memory=False)
    if data.empty:
        raise ValueError(f"{label} is empty: {path}")
    require_columns(data, ["stay_id"], label)
    for column in ID_COLS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="raise").astype("int64")
    if unique_stay and data["stay_id"].duplicated().any():
        raise ValueError(f"{label} has duplicate stay_id")
    return data


def check_ids(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    for column in ["subject_id", "hadm_id"]:
        if column not in left.columns or column not in right.columns:
            continue
        a = left.set_index("stay_id")[column]
        b = right.set_index("stay_id")[column]
        common = a.index.intersection(b.index)
        bad = a.loc[common].notna() & b.loc[common].notna() & a.loc[common].ne(b.loc[common])
        if bad.any():
            examples = bad[bad].index[:5].tolist()
            raise ValueError(f"{label} identifier mismatch in {column}; example stay_id={examples}")


def strict_join(left: pd.DataFrame, right: pd.DataFrame, label: str) -> pd.DataFrame:
    """Join one-row-per-stay tables without silently creating _x/_y fields."""
    check_ids(left, right, label)
    left = left.copy()
    right = right.copy()
    overlap = [column for column in right.columns if column != "stay_id" and column in left.columns]
    for column in overlap:
        a = left.set_index("stay_id")[column]
        b = right.set_index("stay_id")[column]
        common = a.index.intersection(b.index)
        if len(common):
            av = a.loc[common]
            bv = b.loc[common]
            both = av.notna() & bv.notna()
            av_num = pd.to_numeric(av, errors="coerce")
            bv_num = pd.to_numeric(bv, errors="coerce")
            numeric_pair = both & av_num.notna() & bv_num.notna()
            bad_numeric = numeric_pair & ~np.isclose(
                av_num.astype(float), bv_num.astype(float), equal_nan=True
            )
            bad_text = (
                both
                & ~numeric_pair
                & av.astype("string").str.strip().ne(bv.astype("string").str.strip())
            )
            bad = bad_numeric | bad_text
            if bad.any():
                examples = bad[bad].index[:5].tolist()
                raise ValueError(
                    f"{label} conflicts with existing column {column}; example stay_id={examples}"
                )
        # Preserve the canonical left copy but fill a left-side missing value
        # from an observed right-side value before dropping the duplicate.
        fill = left["stay_id"].map(b)
        left[column] = left[column].where(left[column].notna(), fill)
        right = right.drop(columns=[column])
    return left.merge(right, on="stay_id", how="left", validate="one_to_one")


def direct_input(a: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    if a.trajectory_dir is None or a.gbmt_dir is None:
        raise ValueError("Direct mode requires both --trajectory-dir and --gbmt-dir")
    if a.assignments is not None:
        raise ValueError("--assignments is only used with --mortality-input compatibility mode")

    cohort = read(a.cohort, "strict cohort")
    outcomes_path = a.trajectory_dir.expanduser().resolve() / "landmark_outcomes_v1.csv"
    assignments_path = (
        a.gbmt_dir.expanduser().resolve()
        / a.model_set
        / f"gbmt_{a.model_set}_K{a.k}_assignments.csv"
    )
    outcomes = read(outcomes_path, "landmark outcomes")
    assignments = read(assignments_path, "frozen GBMT assignments")
    require_columns(
        outcomes,
        [*ID_COLS, "day5_landmark_eligible", "mortality_90d", "time_to_death_or_90d_days"],
        "landmark outcomes",
    )
    require_columns(
        assignments,
        [*ID_COLS, "assigned_class", "max_posterior"],
        "frozen GBMT assignments",
    )

    outcome_keep = [
        *ID_COLS,
        "day5_landmark_eligible",
        "mortality_90d",
        "time_to_death_or_90d_days",
    ]
    for optional in ["in_hospital_mortality", "death_time"]:
        if optional in outcomes.columns:
            outcome_keep.append(optional)
    outcomes = outcomes[outcome_keep].copy()
    outcomes["analysis_time"] = (
        pd.to_numeric(outcomes["time_to_death_or_90d_days"], errors="coerce") - 5.0
    ).clip(lower=0.0, upper=85.0)
    outcomes["analysis_event"] = pd.to_numeric(outcomes["mortality_90d"], errors="coerce")

    posterior = [column for column in assignments.columns if column.startswith("posterior_class_")]
    assignments = assignments[
        [*ID_COLS, "assigned_class", "max_posterior", *posterior]
    ].rename(columns={"assigned_class": "phenotype_class"})
    assignments["phenotype_class"] = pd.to_numeric(
        assignments["phenotype_class"], errors="raise"
    ).astype(int)

    mortality = assignments.merge(outcomes, on=ID_COLS, how="inner", validate="one_to_one")
    if len(mortality) != len(assignments):
        raise ValueError("At least one frozen GBMT assignment has no landmark outcome row")

    extras = [
        column
        for column in cohort.columns
        if column not in ID_COLS and column not in mortality.columns
    ]
    if extras:
        mortality = strict_join(
            mortality,
            cohort[[*ID_COLS, *extras]],
            "cohort",
        )
    mortality["model_set"] = a.model_set
    mortality["K"] = int(a.k)
    return mortality, {
        "mode": "direct_from_frozen_gbmt_and_landmark_outcomes",
        "cohort": str(a.cohort.expanduser().resolve()),
        "landmark_outcomes": str(outcomes_path),
        "assignments": str(assignments_path),
        "model_set": a.model_set,
        "K": int(a.k),
    }


def compatibility_input(a: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    mortality = read(a.mortality_input, "mortality input")
    assignments_path = None
    if a.assignments is not None:
        assignments = read(a.assignments, "replacement assignments")
        class_col = "phenotype_class" if "phenotype_class" in assignments.columns else "assigned_class"
        require_columns(assignments, ["stay_id", class_col, "max_posterior"], "replacement assignments")
        keep = [
            column
            for column in [
                "stay_id",
                class_col,
                "max_posterior",
                *[x for x in assignments.columns if x.startswith("posterior_class_")],
            ]
            if column in assignments.columns
        ]
        assignments = assignments[keep].rename(columns={class_col: "phenotype_class"})
        old = [column for column in keep if column != "stay_id" and column in mortality.columns]
        mortality = mortality.drop(columns=old, errors="ignore").merge(
            assignments, on="stay_id", how="inner", validate="one_to_one"
        )
        assignments_path = str(a.assignments.expanduser().resolve())
    return mortality, {
        "mode": "augment_existing_mortality_input",
        "mortality_input": str(a.mortality_input.expanduser().resolve()),
        "assignments": assignments_path,
    }


def main() -> None:
    a = parse_args()
    out = a.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mortality, source_meta = direct_input(a) if a.cohort is not None else compatibility_input(a)
    covariates = read(a.plan_covariates, "plan covariates")
    joined = strict_join(mortality, covariates, "plan covariates")
    require_columns(
        joined,
        ["phenotype_class", "day5_landmark_eligible", "analysis_time", "analysis_event"],
        "joined mortality input",
    )

    missingness = pd.DataFrame(
        {
            "variable": joined.columns,
            "n_rows": len(joined),
            "n_missing": [int(joined[column].isna().sum()) for column in joined.columns],
            "missing_percent": [100.0 * joined[column].isna().mean() for column in joined.columns],
        }
    )
    missingness.to_csv(out / "plan_mortality_input_missingness_v2.csv", index=False)
    output = out / "mortality_input_plan_v2.csv"
    joined.to_csv(output, index=False)
    meta = {
        "script": "prepare_adhf_plan_mortality_input_v2.py",
        "version": "strict_plan_direct_or_compatibility_join_no_imputation",
        **source_meta,
        "plan_covariates": str(a.plan_covariates.expanduser().resolve()),
        "n_rows": int(len(joined)),
        "n_landmark_eligible": int(
            pd.to_numeric(joined["day5_landmark_eligible"], errors="coerce").eq(1).sum()
        ),
        "landmark_origin": "ICU day 5",
        "horizon_days_from_landmark": 85,
        "no_imputation": True,
        "outputs": [str(output), str(out / "plan_mortality_input_missingness_v2.csv")],
    }
    (out / "RUN_METADATA.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, default=str))
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
