import asyncio

import pytest

from evals.evals import score
from agent.pipeline import auto_approve, harmonize, items_for_review, oracle_planner
from evals.synthetic import SURVEY_IDS, make_survey

def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("style", SURVEY_IDS)
def test_oracle_pipeline_scores_perfectly(style):
    sv = make_survey(style)
    res = run(harmonize(sv.raw, sv.codebook, style, oracle_planner(sv.truth_spec), approve=auto_approve))
    sc = score(res.out, sv.truth_df, res.spec, sv.truth_spec)
    assert res.status == "ok" and res.rounds == 1
    assert sc["mapping_acc"] == sc["unit_acc"] == 1.0 and sc["value_acc"] > 0.999


def test_planner_recovers_after_validator_feedback():
    sv = make_survey("A")
    calls = []

    async def flaky(ctx, feedback):
        calls.append(feedback)
        spec = sv.truth_spec.model_copy(deep=True)
        if feedback is None:  # first attempt: wrong glucose unit
            for m in spec.mappings:
                if m.target == "fpg":
                    m.source_unit = "mmol/L"
        return spec

    res = run(harmonize(sv.raw, sv.codebook, "A", flaky, approve=auto_approve))
    assert len(calls) == 2 and "fpg" in calls[1] and "mg/dL" in calls[1]
    assert res.status == "ok" and res.rounds == 2


def test_gives_up_after_max_rounds_and_reports():
    sv = make_survey("A")

    async def always_wrong(ctx, feedback):
        spec = sv.truth_spec.model_copy(deep=True)
        for m in spec.mappings:
            if m.target == "fpg":
                m.source_unit = "mmol/L"
        return spec

    res = run(harmonize(sv.raw, sv.codebook, "A", always_wrong, approve=auto_approve, max_rounds=2))
    assert res.rounds == 2 and res.status == "needs_attention"


def test_human_rejection_drops_mapping():
    sv = make_survey("A")

    def reject_treated(items):
        return {m.target: m.target != "htn_treated" for m in items}

    res = run(harmonize(sv.raw, sv.codebook, "A", oracle_planner(sv.truth_spec.model_copy(deep=True)),
                        approve=reject_treated))
    assert res.out["htn_treated"].isna().all()
    assert res.decisions["htn_treated"] is False


def test_review_set_contains_judgment_calls():
    sv = make_survey("A")
    targets = {m.target for m in items_for_review(sv.truth_spec)}
    assert {"htn_treated", "dm_treated"} <= targets
