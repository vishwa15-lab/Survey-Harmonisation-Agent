"""Deterministic Validator (the analogue of a `check_extraction` step).

Rules are plain code so quality never depends on the model. Each issue can carry a
`hint` that the planner agent reads when it revises its mapping.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel

from .schema import STANDARD_VARS
from .units import CONVERSIONS, to_standard


class Issue(BaseModel):
    variable: str
    severity: Literal["error", "warning", "info"]
    check: str
    message: str
    n_affected: int = 0
    hint: str | None = None


def _unit_hint(name: str, median: float) -> str | None:
    """If the median would be plausible in some other unit, suggest it."""
    var = STANDARD_VARS[name]
    table = CONVERSIONS.get(name, {})
    for unit in table:
        if unit == var.unit:
            continue
        conv = float(to_standard(name, pd.Series([median]), unit).iloc[0])
        if var.lo <= conv <= var.hi:
            return (f"median {median:.2f} looks like unit {unit!r} "
                    f"(would be {conv:.2f} {var.unit}); check the codebook and set source_unit")
    return None


def run_checks(out: pd.DataFrame, max_oor: float = 0.01, max_missing: float = 0.5) -> list[Issue]:
    issues: list[Issue] = []
    n = len(out)

    for name, var in STANDARD_VARS.items():
        col = out[name]
        nn = int(col.notna().sum())
        if nn == 0:
            issues.append(Issue(variable=name, severity="info", check="unmapped",
                                message="no values (not mapped or not collected)"))
            continue
        miss = 1 - nn / n
        if miss > max_missing:
            issues.append(Issue(variable=name, severity="warning", check="missingness",
                                message=f"{miss:.0%} missing", n_affected=n - nn))
        vals = col.dropna()
        if var.kind == "binary":
            bad = int((~vals.isin([0, 1])).sum())
            if bad:
                issues.append(Issue(variable=name, severity="error", check="non_binary",
                                    message=f"{bad} values are not 0/1", n_affected=bad,
                                    
                                    hint="add recode entries for every raw value, or fix source column"))
        else:
            oor = int(((vals < var.lo) | (vals > var.hi)).sum())
            if oor / nn > max_oor:
                hint = _unit_hint(name, float(vals.median())) if name in CONVERSIONS else None
                issues.append(Issue(variable=name, severity="error", check="out_of_range",
                                    message=f"{oor}/{nn} values outside {var.lo}-{var.hi} {var.unit}",
                                    n_affected=oor, hint=hint))

    # logic checks
    for treated, dx in (("htn_treated", "htn_dx"), ("dm_treated", "dm_dx")):
        both = out[[treated, dx]].dropna()
        bad = int(((both[treated] == 1) & (both[dx] == 0)).sum())
        if bad:
            issues.append(Issue(variable=treated, severity="error", check="treated_gt_diagnosed",
                                message=f"{bad} people treated but not diagnosed", n_affected=bad,
                                hint="check recode direction (1/2 coding) or the gating question"))
    bp = out[["sbp", "dbp"]].dropna()
    if len(bp):
        bad = int((bp["dbp"] >= bp["sbp"]).sum())
        if bad / len(bp) > max_oor:
            issues.append(Issue(variable="dbp", severity="error", check="dbp_ge_sbp",
                                message=f"{bad} rows have diastolic >= systolic", n_affected=bad,
                                hint="systolic/diastolic columns may be swapped"))
    return issues


def format_issues(issues: list[Issue]) -> str:
    if not issues:
        return "No issues."
    return "\n".join(
        f"[{i.severity}] {i.variable}/{i.check}: {i.message}" + (f" HINT: {i.hint}" if i.hint else "")
        for i in issues
    )
