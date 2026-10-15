"""Synthetic 'country' surveys with cryptic columns, mixed units and odd coding.

Ground truth is known, so evals can score mapping accuracy and value accuracy.
No real survey data is used anywhere in this repo.

Three styles on purpose differ in what usually trips harmonization up:
  A: cryptic names, mg/dL lipids and glucose, 1/2 yes-no coding, HbA1c not collected
  B: letter codes (M/F, Y/N), height in inches, weight in lb, HbA1c in mmol/mol
  C: reversed sex coding (1 = female), height in metres, 999 as a missing code
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.mapping import MappingSpec, ValueRecode, VarMapping
from core.schema import ALWAYS_REVIEW, STANDARD_VARS
from core.units import CONVERSIONS, GLU, CHOL, TRG


@dataclass
class Survey:
    survey_id: str
    raw: pd.DataFrame
    codebook: dict[str, str]
    truth_df: pd.DataFrame
    truth_spec: MappingSpec


def _latent(n: int, rng: np.random.Generator) -> pd.DataFrame:
    male = rng.integers(0, 2, n)
    age = rng.integers(18, 80, n)
    height = np.clip(np.where(male == 1, rng.normal(175, 7, n), rng.normal(162, 6.5, n)), 140, 205)
    bmi = rng.normal(26, 4.5, n)
    weight = np.clip(bmi * (height / 100) ** 2, 40, 160)
    sbp = np.clip(105 + 0.55 * (age - 18) + rng.normal(0, 11, n), 85, 220)
    dbp = np.clip(0.52 * sbp + rng.normal(8, 6, n), 45, sbp - 15)
    fpg = np.clip(np.exp(rng.normal(np.log(5.2), 0.2, n)) + 0.004 * (age - 18), 3.2, 20)
    hba1c = np.clip(4.4 + 0.2 * fpg + rng.normal(0, 0.2, n), 4.0, 14.0)
    tc = np.clip(rng.normal(5.0, 1.0, n), 2.5, 9.0)
    hdl = np.clip(rng.normal(1.45 - 0.2 * male, 0.3, n), 0.6, 2.8)
    trg = np.clip(np.exp(rng.normal(np.log(1.3), 0.45, n)), 0.4, 10)
    ldl = np.clip(tc - hdl - trg / 2.2, 0.8, 7.0)
    htn_dx = ((sbp >= 140) & (rng.random(n) < 0.7)) | (rng.random(n) < 0.04)
    htn_tr = htn_dx & (rng.random(n) < 0.6)
    dm_dx = ((fpg >= 7) & (rng.random(n) < 0.6)) | (rng.random(n) < 0.02)
    dm_tr = dm_dx & (rng.random(n) < 0.5)
    df = pd.DataFrame(dict(
        male=male, age=age, height_cm=height, weight_kg=weight, sbp=sbp, dbp=dbp, fpg=fpg,
        hba1c=hba1c, tc=tc, hdl=hdl, ldl=ldl, trg=trg,
        htn_dx=htn_dx.astype(int), htn_treated=htn_tr.astype(int),
        dm_dx=dm_dx.astype(int), dm_treated=dm_tr.astype(int),
    ))
    return df[list(STANDARD_VARS)].astype(float)


def _from_standard(target: str, s: pd.Series, unit: str | None) -> pd.Series:
    """Inverse of units.to_standard (used only to *create* raw data)."""
    if unit is None or target not in CONVERSIONS:
        return s
    if target == "hba1c" and unit == "mmol/mol":
        return (s - 2.15) * 10.929
    f = CONVERSIONS[target][unit]
    return s / f  # type: ignore[operator]


# one entry per collected variable
# keys: target, col, desc, unit, dec (decimals), codes {std: raw}, gate, missing
_STYLES: dict[str, list[dict]] = {
    "A": [
        dict(target="male", col="h1", desc="Sex (1 = male, 2 = female)", codes={1: 1, 0: 2}),
        dict(target="age", col="h2", desc="Age in years", unit="years", dec=0),
        dict(target="height_cm", col="h3a", desc="Height, cm", unit="cm", dec=0),
        dict(target="weight_kg", col="h3b", desc="Weight, kg", unit="kg", dec=1),
        dict(target="sbp", col="c1s", desc="Systolic blood pressure, mmHg", unit="mmHg", dec=0),
        dict(target="dbp", col="c1d", desc="Diastolic blood pressure, mmHg", unit="mmHg", dec=0),
        dict(target="fpg", col="g7", desc="Fasting plasma glucose, mg/dL", unit="mg/dL", dec=0),
        dict(target="tc", col="l1", desc="Total cholesterol, mg/dL", unit="mg/dL", dec=0),
        dict(target="hdl", col="l2", desc="HDL cholesterol, mg/dL", unit="mg/dL", dec=0),
        dict(target="ldl", col="l3", desc="LDL cholesterol, mg/dL", unit="mg/dL", dec=0),
        dict(target="trg", col="l4", desc="Triglycerides, mg/dL", unit="mg/dL", dec=0),
        dict(target="htn_dx", col="d1", desc="Ever told by a doctor you have raised blood pressure? (1 = yes, 2 = no)",
             codes={1: 1, 0: 2}),
        dict(target="htn_treated", col="d2",
             desc="Currently taking medication for raised blood pressure? Asked only if d1 = 1 (1 = yes, 2 = no)",
             codes={1: 1, 0: 2}, gate="htn_dx"),
        dict(target="dm_dx", col="d3", desc="Ever told by a doctor you have diabetes? (1 = yes, 2 = no)",
             codes={1: 1, 0: 2}),
        dict(target="dm_treated", col="d4",
             desc="Currently taking medication for diabetes? Asked only if d3 = 1 (1 = yes, 2 = no)",
             codes={1: 1, 0: 2}, gate="dm_dx"),
    ],
    "B": [
        dict(target="male", col="Q101", desc="Sex (M = male, F = female)", codes={1: "M", 0: "F"}),
        dict(target="age", col="Q102", desc="Age in years", unit="years", dec=0),
        dict(target="height_cm", col="Q103", desc="Height in inches", unit="in", dec=1),
        dict(target="weight_kg", col="Q104", desc="Weight in pounds (lb)", unit="lb", dec=1),
        dict(target="sbp", col="SYS_AVG", desc="Average systolic BP, mmHg", unit="mmHg", dec=0),
        dict(target="dbp", col="DIA_AVG", desc="Average diastolic BP, mmHg", unit="mmHg", dec=0),
        dict(target="fpg", col="FPG_MMOL", desc="Fasting plasma glucose, mmol/L", unit="mmol/L", dec=2),
        dict(target="hba1c", col="A1C_IFCC", desc="HbA1c, IFCC units (mmol/mol)", unit="mmol/mol", dec=0),
        dict(target="tc", col="CHOL_T", desc="Total cholesterol, mmol/L", unit="mmol/L", dec=2),
        dict(target="hdl", col="CHOL_H", desc="HDL cholesterol, mmol/L", unit="mmol/L", dec=2),
        dict(target="ldl", col="CHOL_L", desc="LDL cholesterol, mmol/L", unit="mmol/L", dec=2),
        dict(target="trg", col="TG", desc="Triglycerides, mmol/L", unit="mmol/L", dec=2),
        dict(target="htn_dx", col="Q201", desc="Told you have high blood pressure? (Y/N)", codes={1: "Y", 0: "N"}),
        dict(target="htn_treated", col="Q202", desc="Taking BP medication now? Only asked if Q201 = Y (Y/N)",
             codes={1: "Y", 0: "N"}, gate="htn_dx"),
        dict(target="dm_dx", col="Q203", desc="Told you have diabetes? (Y/N)", codes={1: "Y", 0: "N"}),
        dict(target="dm_treated", col="Q204", desc="Taking diabetes medication now? Only asked if Q203 = Y (Y/N)",
             codes={1: "Y", 0: "N"}, gate="dm_dx"),
    ],
    "C": [
        dict(target="male", col="sexo", desc="Sex (0 = male, 1 = female)", codes={1: 0, 0: 1}),
        dict(target="age", col="edad", desc="Age in years", unit="years", dec=0),
        dict(target="height_cm", col="talla_m", desc="Height in metres", unit="m", dec=2),
        dict(target="weight_kg", col="peso_kg", desc="Weight in kilograms", unit="kg", dec=1),
        dict(target="sbp", col="pas", desc="Systolic BP, mmHg. 999 = not measured", unit="mmHg", dec=0, missing=999),
        dict(target="dbp", col="pad", desc="Diastolic BP, mmHg. 999 = not measured", unit="mmHg", dec=0, missing=999),
        dict(target="fpg", col="gluc", desc="Fasting glucose, mmol/L. 999 = not measured", unit="mmol/L", dec=2,
             missing=999),
        dict(target="hba1c", col="hba1c_pct", desc="HbA1c, percent", unit="%", dec=1),
        dict(target="tc", col="col_total", desc="Total cholesterol, mmol/L", unit="mmol/L", dec=2),
        dict(target="hdl", col="hdl_mgdl", desc="HDL cholesterol, mg/dL", unit="mg/dL", dec=0),
        dict(target="ldl", col="ldl", desc="LDL cholesterol, mmol/L", unit="mmol/L", dec=2),
        dict(target="trg", col="trig", desc="Triglycerides, mmol/L", unit="mmol/L", dec=2),
        dict(target="htn_dx", col="hta_dx", desc="Diagnosed hypertension (1 = yes, 0 = no)", codes={1: 1, 0: 0}),
        dict(target="htn_treated", col="hta_trat",
             desc="On antihypertensive treatment (1 = yes, 0 = no); blank if not diagnosed",
             codes={1: 1, 0: 0}, gate="htn_dx"),
        dict(target="dm_dx", col="dm_dx", desc="Diagnosed diabetes (1 = yes, 0 = no)", codes={1: 1, 0: 0}),
        dict(target="dm_treated", col="dm_trat",
             desc="On diabetes treatment (1 = yes, 0 = no); blank if not diagnosed",
             codes={1: 1, 0: 0}, gate="dm_dx"),
    ],
}

SURVEY_IDS = list(_STYLES)


def make_survey(style: str = "A", n: int = 400, seed: int = 0) -> Survey:
    if style not in _STYLES:
        raise ValueError(f"style must be one of {SURVEY_IDS}")
    rng = np.random.default_rng(seed + ord(style))
    truth = _latent(n, rng)
    raw = pd.DataFrame({"pid": np.arange(1, n + 1)})
    codebook = {"pid": "Participant identifier"}
    maps: list[VarMapping] = []

    for spec in _STYLES[style]:
        t, col = spec["target"], spec["col"]
        var = STANDARD_VARS[t]
        codes, gate, missing = spec.get("codes"), spec.get("gate"), spec.get("missing")
        unit = spec.get("unit")
        if var.kind == "binary":
            r = truth[t].map(codes).astype(object)
            if gate:
                r = r.where(truth[gate] == 1, np.nan)
            recode = [ValueRecode(raw=str(v), std=int(k)) for k, v in codes.items()]
        else:
            r = _from_standard(t, truth[t], unit).round(spec.get("dec", 2))
            if spec.get("dec") == 0:
                r = r.astype("float64")
            recode = []
            if missing is not None:
                idx = rng.choice(n, size=max(1, int(0.03 * n)), replace=False)
                r.iloc[idx] = missing
                truth.loc[truth.index[idx], t] = np.nan
        raw[col] = r
        codebook[col] = spec["desc"]
        maps.append(VarMapping(
            target=t, source_column=col, source_unit=unit if var.kind == "continuous" else None,
            recode=recode, missing_codes=[str(missing)] if missing is not None else [],
            na_to_zero_if_zero=gate, confidence=1.0, needs_review=t in ALWAYS_REVIEW,
            rationale="ground truth"))

    # distractor columns
    raw["intv"] = rng.integers(100, 140, n)
    codebook["intv"] = "Interviewer identifier"
    raw["wt_samp"] = np.round(rng.uniform(0.5, 3.0, n), 3)
    codebook["wt_samp"] = "Sampling weight"

    collected = {m.target for m in maps}
    for t in STANDARD_VARS:
        if t not in collected:  # not collected in this survey
            truth[t] = np.nan
            maps.append(VarMapping(target=t, source_column=None, source_unit=None, recode=[],
                                   missing_codes=[], na_to_zero_if_zero=None, confidence=1.0,
                                   needs_review=False, rationale="not collected in this survey"))
    # shuffle columns a little so order carries no signal
    cols = list(raw.columns)
    rng.shuffle(cols)
    raw = raw[["pid"] + [c for c in cols if c != "pid"]]
    return Survey(survey_id=style, raw=raw, codebook=codebook,
                  truth_df=truth[list(STANDARD_VARS)],
                  truth_spec=MappingSpec(survey_id=style, mappings=maps))
