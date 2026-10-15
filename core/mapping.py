"""Declarative mapping spec + deterministic executor.

Design choice: the LLM proposes a *declarative* MappingSpec (data), and plain code
applies it. The model never executes arbitrary code on health data, and every
transformation is auditable and replayable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .schema import STANDARD_VARS
from .units import to_standard


class ValueRecode(BaseModel):
    raw: str = Field(description="Raw value as it appears in the data, e.g. '1', 'Y', 'M'")
    std: int = Field(description="Standard value: 1 or 0")


class VarMapping(BaseModel):
    target: str = Field(description="Standard variable name")
    source_column: str | None = Field(description="Raw column name, or null if not collected")
    source_unit: str | None = Field(description="Unit of the raw column; null if already standard or binary")
    recode: list[ValueRecode] = Field(description="Value recodes for binary variables; empty otherwise")
    missing_codes: list[str] = Field(description="Raw values that mean 'missing', e.g. ['999']")
    na_to_zero_if_zero: str | None = Field(
        description="Skip-pattern rule: if this standard variable is 0, a missing value here becomes 0"
    )
    confidence: float = Field(description="0 to 1")
    needs_review: bool = Field(description="True if a human should approve this mapping")
    rationale: str = Field(description="One or two sentences citing the codebook")


class MappingSpec(BaseModel):
    survey_id: str
    mappings: list[VarMapping]


def _norm(v) -> str | None:
    if pd.isna(v):
        return None
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def apply_mapping(df: pd.DataFrame, spec: MappingSpec) -> tuple[pd.DataFrame, list[str]]:
    """Apply `spec` to raw `df`. Returns (standard dataframe, warnings)."""
    out = pd.DataFrame(np.nan, index=df.index, columns=list(STANDARD_VARS), dtype="float64")
    warnings: list[str] = []

    for m in spec.mappings:
        var = STANDARD_VARS.get(m.target)
        if var is None:
            warnings.append(f"unknown target {m.target!r} ignored")
            continue
        if m.source_column is None:
            continue
        if m.source_column not in df.columns:
            warnings.append(f"{m.target}: source column {m.source_column!r} not in data")
            continue

        s = df[m.source_column].map(_norm)
        if m.missing_codes:
            miss = {c.strip() for c in m.missing_codes}
            s = s.where(~s.isin(miss))

        if var.kind == "binary":
            if m.recode:
                lut = {r.raw.strip(): r.std for r in m.recode}
                unknown = set(s.dropna().unique()) - set(lut)
                if unknown:
                    warnings.append(f"{m.target}: values with no recode: {sorted(unknown)[:5]}")
                s = s.map(lut)
            out[m.target] = pd.to_numeric(s, errors="coerce")
        else:
            try:
                out[m.target] = to_standard(m.target, pd.to_numeric(s, errors="coerce"), m.source_unit)
            except ValueError as e:
                warnings.append(f"{m.target}: {e}")

    # second pass: skip-pattern NAs
    for m in spec.mappings:
        g = m.na_to_zero_if_zero
        if m.source_column is None or m.source_column not in df.columns:
            continue  # never invent values for an unmapped variable
        if g and m.target in out and g in out:
            fix = out[m.target].isna() & (out[g] == 0)
            out.loc[fix, m.target] = 0.0
    return out, warnings
