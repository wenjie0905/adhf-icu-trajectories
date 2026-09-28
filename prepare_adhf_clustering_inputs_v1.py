#!/usr/bin/env python3
"""Prepare GE2/GE3/FULL5 model inputs from one trajectory run.

This script is deliberately downstream of ``rebuild_adhf_cohort_v4.py`` and
``prepare_adhf_trajectory_v1.py``.  It does not change cohort eligibility,
recalculate medication exposure, impute missing markers, or refit a trajectory
model.  It only derives model-ready stay sets from the already audited daily
trajectory table.

The completeness rule used here is stricter than the cohort sufficiency gate:
one complete marker-day requires all four plan markers to be observed on that
ICU-relative day (cumulative net fluid balance, urine output, creatinine, and
IV loop dose in furosemide equivalents).  GE2/GE3/FULL5 require at least 2/3/5
such days, respectively.  All five ICU-relative days are retained in each
selected set so that downstream trajectory models can handle intermittent
missingness without silently treating post-ICU days as zero.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLS = ["subject_id", "hadm_id", "stay_id"]
RAW_MARKERS = [
    "cumulative_net_fluid_balance_ml",
    "urine_output_ml",
    "creatinine_mg_dl",
    "iv_loop_dose_mg_eq",
]
Z_MARKERS = [
    "cumulative_net_fluid_balance_z",
    "urine_output_z",
    "creatinine_z",
    "iv_loop_dose_z",
]
SETS = {"GE2": 2, "GE3": 3, "FULL5": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GE2, GE3, and FULL5 trajectory inputs from an audited "
            "prepare_adhf_trajectory_v1.py output directory."
        )
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="Directory containing trajectory_daily_scaled_v1.csv and QC outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for model-ready files and QC summaries.",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="Optional rebuilt one-row-per-stay cohort for an ID consistency check.",
    )
    parser.add_argument(
        "--primary-set",
        choices=sorted(SETS),
        default="GE3",
        help="Set labelled primary in the manifest; all three sets are always written.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        fail(f"{name} is missing required columns: {', '.join(missing)}")


def read_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        fail(f"Required {name} was not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - error text is user-facing
        fail(f"Could not read {name} ({path}): {exc}")


def normalise_ids(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    require_columns(out, ID_COLS, name)
    for column in ID_COLS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
        if out[column].isna().any():
            fail(f"{name}.{column} contains missing or non-numeric identifiers")
    return out


def id_key(frame: pd.DataFrame) -> pd.Series:
    # Strings avoid any float formatting if a CSV was read with mixed integer
    # types.  The identifiers themselves remain integer-like in output files.
    return frame[ID_COLS].astype("string").agg("|".join, axis=1)


def validate_unique(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if frame.duplicated(columns).any():
        dup = frame.loc[frame.duplicated(columns, keep=False), columns].head(5)
        fail(f"{name} has duplicate keys {columns}; examples: {dup.to_dict('records')}")


def make_complete_marker_day(daily: pd.DataFrame) -> pd.Series:
    """Return the plan's all-four-markers complete-day flag.

    The trajectory preparation script already makes cumulative fluid balance
    missing until preceding fluid days are complete and keeps unquantified loop
    doses missing.  The explicit flags below provide an additional defensive
    check against a non-null value being treated as complete by accident.
    """

    observed = daily[RAW_MARKERS].notna().all(axis=1)
    if "fluid_balance_complete" in daily.columns:
        observed &= pd.to_numeric(
            daily["fluid_balance_complete"], errors="coerce"
        ).eq(1)
    if "loop_dose_complete" in daily.columns:
        observed &= pd.to_numeric(daily["loop_dose_complete"], errors="coerce").eq(1)
    return observed.astype("int8")


def build_wide_day_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Create one row per stay with fixed marker-day columns (20 columns)."""

    keys = daily[ID_COLS].drop_duplicates().sort_values(ID_COLS).reset_index(drop=True)
    wide = keys.copy()
    for marker in Z_MARKERS:
        for day in range(5):
            selected = daily.loc[daily["day_num"].eq(day), ID_COLS + [marker]]
            selected = selected.rename(columns={marker: f"{marker}_d{day}"})
            wide = wide.merge(selected, on=ID_COLS, how="left", validate="one_to_one")

    # Add a compact day-completeness representation for R/Python QC.
    complete = daily.groupby(ID_COLS, as_index=False)["complete_marker_day"].sum()
    wide = wide.merge(complete, on=ID_COLS, how="left", validate="one_to_one")
    return wide


def write_set_outputs(
    set_name: str,
    selected_keys: pd.DataFrame,
    daily: pd.DataFrame,
    wide: pd.DataFrame,
    long: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Write daily, long, wide, and matrix files for one model-ready set."""

    selected_key_values = set(id_key(selected_keys))
    daily_out = daily.loc[id_key(daily).isin(selected_key_values)].copy()
    daily_out = daily_out.sort_values(ID_COLS + ["day_num"]).reset_index(drop=True)

    wide_out = wide.loc[id_key(wide).isin(selected_key_values)].copy()
    wide_out = wide_out.sort_values(ID_COLS).reset_index(drop=True)

    long_out = long.loc[id_key(long).isin(selected_key_values)].copy()
    long_out = long_out.sort_values(ID_COLS + ["day_num", "marker"]).reset_index(drop=True)

    paths = {
        "daily": output_dir / f"trajectory_daily_scaled_{set_name}_v1.csv",
        "long": output_dir / f"trajectory_long_scaled_{set_name}_v1.csv",
        "wide": output_dir / f"trajectory_wide_scaled_{set_name}_v1.csv",
        "matrix": output_dir / f"trajectory_matrix_scaled_{set_name}_v1.csv",
    }
    daily_out.to_csv(paths["daily"], index=False)
    long_out.to_csv(paths["long"], index=False)
    wide_out.to_csv(paths["wide"], index=False)
    # The matrix name is retained for compatibility with the previous
    # pipeline; unlike the old sparse pivot, this one has fixed d0--d4 columns.
    wide_out.to_csv(paths["matrix"], index=False)
    return {key: str(path) for key, path in paths.items()}


def make_long_table(daily: pd.DataFrame) -> pd.DataFrame:
    keep = ID_COLS + ["day_num"] + Z_MARKERS + ["complete_marker_day"]
    long = daily[keep].melt(
        id_vars=ID_COLS + ["day_num", "complete_marker_day"],
        value_vars=Z_MARKERS,
        var_name="marker",
        value_name="value_z",
    )
    return long


def main() -> None:
    args = parse_args()
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = read_csv(
        trajectory_dir / "trajectory_daily_scaled_v1.csv", "scaled daily trajectory table"
    )
    require_columns(
        daily,
        ID_COLS + ["day_num"] + RAW_MARKERS + Z_MARKERS,
        "trajectory_daily_scaled_v1.csv",
    )
    daily = normalise_ids(daily, "trajectory_daily_scaled_v1.csv")
    daily["day_num"] = pd.to_numeric(daily["day_num"], errors="coerce").astype("Int64")
    if daily["day_num"].isna().any() or (~daily["day_num"].between(0, 4)).any():
        fail("trajectory_daily_scaled_v1.csv has day_num values outside 0--4")
    validate_unique(daily, ID_COLS + ["day_num"], "trajectory_daily_scaled_v1.csv")

    daily["complete_marker_day"] = make_complete_marker_day(daily)
    daily["complete_marker_day"] = daily["complete_marker_day"].astype("int8")
    completeness = (
        daily.groupby(ID_COLS, as_index=False)
        .agg(
            n_complete_marker_days=("complete_marker_day", "sum"),
            n_days_present=("day_num", "nunique"),
        )
        .sort_values(ID_COLS)
    )
    if (completeness["n_days_present"] != 5).any():
        bad = completeness.loc[completeness["n_days_present"] != 5].head(5)
        fail(
            "Each stay must have five day rows (0--4) in the prepared table; "
            f"examples: {bad.to_dict('records')}"
        )

    # Keep the original preparation completeness fields as an audit join when
    # available, but do not use them to replace the all-four-marker definition.
    prep_completeness_path = trajectory_dir / "trajectory_completeness_v1.csv"
    if prep_completeness_path.exists():
        prep = normalise_ids(
            read_csv(prep_completeness_path, "trajectory completeness table"),
            "trajectory_completeness_v1.csv",
        )
        validate_unique(prep, ID_COLS, "trajectory_completeness_v1.csv")
        overlap = set(id_key(completeness)) & set(id_key(prep))
        if overlap != set(id_key(completeness)):
            fail("trajectory_completeness_v1.csv does not cover every prepared stay")
        completeness = completeness.merge(
            prep, on=ID_COLS, how="left", suffixes=("", "_prepare"), validate="one_to_one"
        )

    landmark_path = trajectory_dir / "landmark_outcomes_v1.csv"
    landmark = normalise_ids(read_csv(landmark_path, "landmark outcomes table"), "landmark_outcomes_v1.csv")
    validate_unique(landmark, ID_COLS, "landmark_outcomes_v1.csv")
    landmark_cols = [
        "day5_landmark_eligible",
        "mortality_90d",
        "in_hospital_mortality",
    ]
    require_columns(landmark, landmark_cols, "landmark_outcomes_v1.csv")
    landmark = landmark[ID_COLS + landmark_cols]
    flags = completeness.merge(landmark, on=ID_COLS, how="left", validate="one_to_one")
    if flags[landmark_cols].isna().any().any():
        fail("landmark_outcomes_v1.csv does not cover every prepared stay")

    # Optional cohort check.  The strict v4 rebuild is the source of truth for
    # eligibility; this check catches accidental mixing with the old 2,446 or
    # 3,223-patient cohort without changing any rows.
    cohort_check = None
    if args.cohort is not None:
        cohort = normalise_ids(read_csv(args.cohort.expanduser().resolve(), "rebuilt cohort"), "rebuilt cohort")
        validate_unique(cohort, ID_COLS, "rebuilt cohort")
        cohort_keys = set(id_key(cohort))
        daily_keys = set(id_key(completeness))
        cohort_check = pd.DataFrame(
            {
                "check": [
                    "cohort_rows",
                    "trajectory_stay_rows",
                    "cohort_ids_missing_from_trajectory",
                    "trajectory_ids_missing_from_cohort",
                ],
                "n": [
                    len(cohort_keys),
                    len(daily_keys),
                    len(cohort_keys - daily_keys),
                    len(daily_keys - cohort_keys),
                ],
            }
        )
        if cohort_keys != daily_keys:
            fail("The supplied cohort IDs do not exactly match the trajectory IDs")

    wide = build_wide_day_table(daily)
    long = make_long_table(daily)

    outputs: dict[str, str] = {}
    set_counts: list[dict[str, int | str]] = []
    for set_name, minimum_days in SETS.items():
        selected = flags.loc[flags["n_complete_marker_days"] >= minimum_days, ID_COLS].copy()
        selected_keys = set(id_key(selected))
        set_flags = flags.loc[id_key(flags).isin(selected_keys)].copy()
        files = write_set_outputs(set_name, selected, daily, wide, long, output_dir)
        outputs[set_name] = files

        eligible = set_flags["day5_landmark_eligible"].astype(int).eq(1)
        set_counts.append(
            {
                "model_set": set_name,
                "minimum_complete_marker_days": minimum_days,
                "n_stays": int(len(set_flags)),
                "n_day5_landmark_eligible": int(eligible.sum()),
                "n_90d_deaths_all_stays": int(set_flags["mortality_90d"].astype(int).sum()),
                "n_90d_deaths_day5_eligible": int(
                    set_flags.loc[eligible, "mortality_90d"].astype(int).sum()
                ),
                "n_in_hospital_deaths_all_stays": int(
                    set_flags["in_hospital_mortality"].astype(int).sum()
                ),
                "n_in_hospital_deaths_day5_eligible": int(
                    set_flags.loc[eligible, "in_hospital_mortality"].astype(int).sum()
                ),
            }
        )

    counts = pd.DataFrame(set_counts)
    counts.to_csv(output_dir / "model_ready_set_counts_v1.csv", index=False)
    flags = flags.sort_values(ID_COLS).reset_index(drop=True)
    for set_name, minimum_days in SETS.items():
        flags[f"in_{set_name.lower()}"] = (
            flags["n_complete_marker_days"] >= minimum_days
        ).astype("int8")
    flags["in_primary_set"] = flags["in_" + args.primary_set.lower()]
    flags.to_csv(output_dir / "model_ready_stay_flags_v1.csv", index=False)
    if cohort_check is not None:
        cohort_check.to_csv(output_dir / "trajectory_cohort_id_check_v1.csv", index=False)

    # Distribution is useful in the methods audit and prevents the five
    # patients with 0/1 complete days from being silently carried forward.
    distribution = (
        flags["n_complete_marker_days"]
        .value_counts()
        .rename_axis("n_complete_marker_days")
        .reset_index(name="n_stays")
        .sort_values("n_complete_marker_days")
    )
    distribution.to_csv(output_dir / "complete_marker_day_distribution_v1.csv", index=False)

    metadata = {
        "script": "prepare_adhf_clustering_inputs_v1.py",
        "version": "strict_plan_1458_model_input_sets",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_dir": str(trajectory_dir),
        "cohort_path": str(args.cohort.expanduser().resolve()) if args.cohort else None,
        "input_n_stays": int(len(flags)),
        "primary_set": args.primary_set,
        "model_sets": {name: {"minimum_complete_marker_days": n} for name, n in SETS.items()},
        "complete_marker_day_rule": (
            "all four plan markers observed on the same ICU-relative day; "
            "fluid_balance_complete=1 and loop_dose_complete=1 when those flags exist"
        ),
        "missingness_rule": (
            "no imputation; retain all five ICU-relative day rows, including "
            "post-ICU rows as missing rather than artificial zeros"
        ),
        "primary_markers": RAW_MARKERS,
        "scaled_columns": Z_MARKERS,
        "source_cohort_decision": (
            "0-24 h baseline; 24-48 h observed initiation or quantified dose increase >0; "
            "old primary exclusions retained; no pre-ICU treatment rule"
        ),
        "set_counts": set_counts,
        "outputs": {
            "set_files": outputs,
            "model_ready_set_counts": str(output_dir / "model_ready_set_counts_v1.csv"),
            "model_ready_stay_flags": str(output_dir / "model_ready_stay_flags_v1.csv"),
            "complete_marker_day_distribution": str(
                output_dir / "complete_marker_day_distribution_v1.csv"
            ),
        },
    }
    with (output_dir / "RUN_METADATA.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
