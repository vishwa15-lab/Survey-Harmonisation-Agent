"""Scoring against synthetic ground truth.

mapping_acc : right raw column chosen (null counts as right when the variable is not collected)
unit_acc    : right source unit chosen for continuous variables
value_acc   : harmonized values match truth (continuous within `tol` relative error, binary exact)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.mapping import MappingSpec
from core.schema import STANDARD_VARS


def _value_match(var: str, out: pd.Series, truth: pd.Series, tol: float) -> float:
    both_nan = out.isna() & truth.isna()
    if STANDARD_VARS[var].kind == "binary":
        ok = (out == truth) | both_nan
    else:
        rel = (out - truth).abs() / truth.abs().clip(lower=1e-9)
        ok = (rel <= tol) | both_nan
    return float(ok.mean())


def score(out: pd.DataFrame, truth_df: pd.DataFrame, spec: MappingSpec,
          truth_spec: MappingSpec, tol: float = 0.03) -> dict:
    got = {m.target: m for m in spec.mappings}
    want = {m.target: m for m in truth_spec.mappings}
    per_var: dict[str, dict] = {}
    for t, w in want.items():
        g = got.get(t)
        col_ok = g is not None and g.source_column == w.source_column
        unit_ok = None
        if STANDARD_VARS[t].kind == "continuous" and w.source_column is not None:
            unit_ok = g is not None and g.source_unit == w.source_unit
        per_var[t] = dict(column_ok=col_ok, unit_ok=unit_ok,
                          value_acc=_value_match(t, out[t], truth_df[t], tol))
    n = len(per_var)
    unit_vals = [v["unit_ok"] for v in per_var.values() if v["unit_ok"] is not None]
    return dict(
        mapping_acc=sum(v["column_ok"] for v in per_var.values()) / n,
        unit_acc=(sum(unit_vals) / len(unit_vals)) if unit_vals else 1.0,
        value_acc=float(np.mean([v["value_acc"] for v in per_var.values()])),
        per_var=per_var,
    )
