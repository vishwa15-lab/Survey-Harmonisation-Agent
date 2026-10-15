"""Standard variable schema (a small, generic subset for the demo).

Units are the *standard* units every survey is harmonized into.
Plausible ranges drive the Validator; they are starting rules, not clinical truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class StdVar:
    name: str
    description: str
    unit: str | None
    kind: Literal["continuous", "binary"]
    lo: float | None = None  # plausible minimum (standard unit)
    hi: float | None = None  # plausible maximum (standard unit)


_VARS = [
    StdVar("male", "Sex: 1 = male, 0 = female", None, "binary"),
    StdVar("age", "Age in completed years", "years", "continuous", 15, 110),
    StdVar("height_cm", "Standing height", "cm", "continuous", 100, 230),
    StdVar("weight_kg", "Body weight", "kg", "continuous", 25, 250),
    StdVar("sbp", "Systolic blood pressure", "mmHg", "continuous", 60, 300),
    StdVar("dbp", "Diastolic blood pressure", "mmHg", "continuous", 30, 200),
    StdVar("fpg", "Fasting plasma glucose", "mmol/L", "continuous", 2.0, 30.0),
    StdVar("hba1c", "Glycated haemoglobin (NGSP %)", "%", "continuous", 3.0, 20.0),
    StdVar("tc", "Total cholesterol", "mmol/L", "continuous", 1.0, 15.0),
    StdVar("hdl", "HDL cholesterol", "mmol/L", "continuous", 0.2, 5.0),
    StdVar("ldl", "LDL cholesterol", "mmol/L", "continuous", 0.3, 12.0),
    StdVar("trg", "Triglycerides", "mmol/L", "continuous", 0.2, 20.0),
    StdVar("htn_dx", "Ever diagnosed with raised blood pressure (1/0)", None, "binary"),
    StdVar("htn_treated", "Currently taking BP medication (1/0)", None, "binary"),
    StdVar("dm_dx", "Ever diagnosed with diabetes (1/0)", None, "binary"),
    StdVar("dm_treated", "Currently taking diabetes medication (1/0)", None, "binary"),
]

STANDARD_VARS: dict[str, StdVar] = {v.name: v for v in _VARS}

# Mappings a human must approve even if the agent is confident
# (judgment calls: what counts as "treated" varies by questionnaire).
ALWAYS_REVIEW = {"htn_treated", "dm_treated"}
