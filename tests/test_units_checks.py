import pandas as pd
import pytest

from core.checks import run_checks
from core.mapping import apply_mapping
from evals.synthetic import SURVEY_IDS, make_survey
from core.units import to_standard


def test_unit_conversions():
    assert to_standard("fpg", pd.Series([90.0]), "mg/dL").iloc[0] == pytest.approx(4.995, abs=0.01)
    assert to_standard("tc", pd.Series([193.35]), "mg/dL").iloc[0] == pytest.approx(5.0, abs=0.01)
    assert to_standard("weight_kg", pd.Series([100.0]), "lb").iloc[0] == pytest.approx(45.36, abs=0.01)
    assert to_standard("hba1c", pd.Series([48.0]), "mmol/mol").iloc[0] == pytest.approx(6.54, abs=0.05)


def test_bad_unit_rejected():
    with pytest.raises(ValueError):
        to_standard("sbp", pd.Series([120.0]), "kg")


@pytest.mark.parametrize("style", SURVEY_IDS)
def test_truth_mapping_reproduces_truth(style):
    sv = make_survey(style)
    out, warnings = apply_mapping(sv.raw, sv.truth_spec)
    assert warnings == []
    assert (out.isna() == sv.truth_df.isna()).all().all()
    for col in out:
        rel = (out[col] - sv.truth_df[col]).abs() / sv.truth_df[col].abs().clip(lower=1e-9)
        assert (rel.dropna().max() if rel.notna().any() else 0) < 0.03, col


@pytest.mark.parametrize("style", SURVEY_IDS)
def test_truth_mapping_has_no_validator_errors(style):
    sv = make_survey(style)
    out, _ = apply_mapping(sv.raw, sv.truth_spec)
    assert [i for i in run_checks(out) if i.severity == "error"] == []


def _with(spec, target, **changes):
    spec = spec.model_copy(deep=True)
    for m in spec.mappings:
        if m.target == target:
            for k, v in changes.items():
                setattr(m, k, v)
    return spec


def test_wrong_glucose_unit_is_flagged_with_hint():
    sv = make_survey("A")
    bad = _with(sv.truth_spec, "fpg", source_unit="mmol/L")  # data is really mg/dL
    out, _ = apply_mapping(sv.raw, bad)
    errs = [i for i in run_checks(out) if i.variable == "fpg" and i.severity == "error"]
    assert errs and "mg/dL" in (errs[0].hint or "")


def test_inches_as_cm_is_flagged():
    sv = make_survey("B")
    bad = _with(sv.truth_spec, "height_cm", source_unit="cm")
    out, _ = apply_mapping(sv.raw, bad)
    errs = [i for i in run_checks(out) if i.variable == "height_cm"]
    assert errs and "'in'" in (errs[0].hint or "")


def test_reversed_binary_recode_triggers_logic_check():
    sv = make_survey("A")
    flipped = [r.model_copy(update={"std": 1 - r.std}) for m in sv.truth_spec.mappings
               if m.target == "htn_dx" for r in m.recode]
    bad = _with(sv.truth_spec, "htn_dx", recode=flipped)
    out, _ = apply_mapping(sv.raw, bad)
    checks = {i.check for i in run_checks(out)}
    assert "treated_gt_diagnosed" in checks


def test_missing_recode_values_warned_and_flagged():
    sv = make_survey("A")
    bad = _with(sv.truth_spec, "male", recode=[])  # raw 1/2 pass through unmapped
    out, _ = apply_mapping(sv.raw, bad)
    assert any(i.check == "non_binary" and i.variable == "male" for i in run_checks(out))


def test_skip_pattern_na_to_zero():
    sv = make_survey("A")
    no_gate = _with(sv.truth_spec, "htn_treated", na_to_zero_if_zero=None)
    out, _ = apply_mapping(sv.raw, no_gate)
    assert out["htn_treated"].isna().sum() > 0
    out2, _ = apply_mapping(sv.raw, sv.truth_spec)
    assert out2["htn_treated"].isna().sum() == 0
