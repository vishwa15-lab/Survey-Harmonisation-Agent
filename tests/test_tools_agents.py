import pytest

from agent import tools
from evals.synthetic import make_survey


def test_codebook_search_finds_glucose():
    sv = make_survey("A")
    hits = tools.core_search_codebook(sv.codebook, "fasting glucose")
    assert hits and hits[0]["column"] == "g7"


def test_describe_and_list_columns():
    sv = make_survey("B")
    cols = {c["column"] for c in tools.core_list_columns(sv.raw)}
    assert {"Q103", "FPG_MMOL"} <= cols
    d = tools.core_describe_column(sv.raw, "Q103")
    assert d["median"] > 50  # inches, not cm
    assert "error" in tools.core_describe_column(sv.raw, "nope")


def test_try_mapping_reports_errors():
    sv = make_survey("A")
    ctx = tools.SurveyContext("A", sv.raw, sv.codebook)
    spec = sv.truth_spec.model_copy(deep=True)
    for m in spec.mappings:
        if m.target == "fpg":
            m.source_unit = "mmol/L"
    r = tools.core_try_mapping(ctx, spec)
    assert r["n_errors"] >= 1 and "fpg" in r["report"]


def test_planner_builds_and_requires_model_name(monkeypatch):
    from agent.agents_def import build_planner
    monkeypatch.delenv("HARMONIZER_MODEL_PLANNER", raising=False)
    with pytest.raises(RuntimeError):
        build_planner()
    monkeypatch.setenv("HARMONIZER_MODEL_PLANNER", "test-model")
    a = build_planner()
    assert a.name == "Planner" and len(a.tools) == 5 and a.output_type.__name__ == "MappingSpec"
