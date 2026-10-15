"""Unit conversion into standard units.

Factors are the commonly used analyte conversions
(glucose 18.016, cholesterol 38.67, triglycerides 88.57 mg/dL per mmol/L).
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

GLU = 18.016
CHOL = 38.67
TRG = 88.57


def _ifcc_to_ngsp(x: pd.Series) -> pd.Series:
    return x / 10.929 + 2.15


# target -> {source unit: multiplier or callable}
CONVERSIONS: dict[str, dict[str, float | Callable[[pd.Series], pd.Series]]] = {
    "age": {"years": 1.0},
    "height_cm": {"cm": 1.0, "m": 100.0, "in": 2.54},
    "weight_kg": {"kg": 1.0, "lb": 0.45359237},
    "sbp": {"mmHg": 1.0},
    "dbp": {"mmHg": 1.0},
    "fpg": {"mmol/L": 1.0, "mg/dL": 1 / GLU},
    "hba1c": {"%": 1.0, "mmol/mol": _ifcc_to_ngsp},
    "tc": {"mmol/L": 1.0, "mg/dL": 1 / CHOL},
    "hdl": {"mmol/L": 1.0, "mg/dL": 1 / CHOL},
    "ldl": {"mmol/L": 1.0, "mg/dL": 1 / CHOL},
    "trg": {"mmol/L": 1.0, "mg/dL": 1 / TRG},
}


def allowed_units(target: str) -> list[str]:
    return list(CONVERSIONS.get(target, {}).keys())


def to_standard(target: str, values: pd.Series, unit: str | None) -> pd.Series:
    """Convert numeric `values` from `unit` to the standard unit of `target`.

    unit=None means "already in the standard unit".
    """
    table = CONVERSIONS.get(target)
    if table is None:
        raise ValueError(f"{target!r} is not a continuous standard variable")
    if unit is None:
        return values
    if unit not in table:
        raise ValueError(f"unit {unit!r} not allowed for {target!r}; allowed: {list(table)}")
    f = table[unit]
    return f(values) if callable(f) else values * f
