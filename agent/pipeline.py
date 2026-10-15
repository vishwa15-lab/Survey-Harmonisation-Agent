"""Orchestration: plan -> apply -> validate -> (revise) -> human approval -> final.

Orchestration is explicit Python (the SDK is Python-first), so the control flow is easy
to read, test and explain. The planner is injectable so tests run without an API key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

import pandas as pd

from core.checks import Issue, format_issues, run_checks
from core.mapping import MappingSpec, VarMapping, apply_mapping
from core.schema import ALWAYS_REVIEW
from agent.tools import SurveyContext

PlannerFn = Callable[[SurveyContext, "str | None"], Awaitable[MappingSpec]]
ApproveFn = Callable[[list[VarMapping]], dict[str, bool]]


@dataclass
class Result:
    spec: MappingSpec
    out: pd.DataFrame
    issues: list[Issue]
    rounds: int
    status: str  # "ok" | "needs_attention"
    decisions: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def make_agent_planner(session_db: str = "data/sessions.db") -> PlannerFn:
    """Planner backed by the OpenAI Agents SDK. Needs OPENAI_API_KEY and a model name."""
    from agents import Runner, SQLiteSession

    from .agents_def import build_planner

    agent = build_planner()
    sessions: dict[str, SQLiteSession] = {}

    async def plan(ctx: SurveyContext, feedback: str | None) -> MappingSpec:
        session = sessions.setdefault(ctx.survey_id, SQLiteSession(f"survey-{ctx.survey_id}", session_db))
        if feedback is None:
            prompt = (f"Survey id: {ctx.survey_id}. Map this survey to the standard schema. "
                      f"It has {len(ctx.df)} rows and {len(ctx.df.columns)} columns.")
        else:
            prompt = ("The validator found problems with your last mapping. Revise it, using try_mapping "
                      "to confirm the errors are gone.\n" + feedback)
        result = await Runner.run(agent, prompt, context=ctx, session=session, max_turns=25)
        spec = result.final_output_as(MappingSpec)
        spec.survey_id = ctx.survey_id
        return spec

    return plan


def oracle_planner(truth: MappingSpec) -> PlannerFn:
    """Returns the ground-truth mapping. Used to sanity-check the pipeline and evals."""
    async def plan(ctx: SurveyContext, feedback: str | None) -> MappingSpec:
        return truth
    return plan


def items_for_review(spec: MappingSpec, threshold: float = 0.7) -> list[VarMapping]:
    return [m for m in spec.mappings
            if m.source_column is not None
            and (m.needs_review or m.confidence < threshold or m.target in ALWAYS_REVIEW)]


def auto_approve(items: list[VarMapping]) -> dict[str, bool]:
    return {m.target: True for m in items}


def cli_approve(items: list[VarMapping]) -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    print("\nHuman review needed for these mappings:")
    for m in items:
        recode = ", ".join(f"{r.raw}->{r.std}" for r in m.recode) or "-"
        print(f"\n  {m.target} <- {m.source_column}  unit={m.source_unit} recode=[{recode}] "
              f"gate={m.na_to_zero_if_zero} conf={m.confidence:.2f}\n  why: {m.rationale}")
        decisions[m.target] = input("  approve? [y/N] ").strip().lower() == "y"
    return decisions


def _problems(out: pd.DataFrame, warnings: list[str]) -> tuple[list[Issue], str]:
    issues = run_checks(out)
    errors = [i for i in issues if i.severity == "error"]
    text = format_issues(errors) + ("\nWarnings: " + "; ".join(warnings) if warnings else "")
    return issues, text if (errors or warnings) else ""


async def harmonize(df: pd.DataFrame, codebook: dict[str, str], survey_id: str,
                    planner: PlannerFn, approve: ApproveFn = cli_approve,
                    max_rounds: int = 3) -> Result:
    ctx = SurveyContext(survey_id=survey_id, df=df, codebook=codebook)
    feedback: str | None = None
    spec: MappingSpec | None = None
    rounds = 0

    for rounds in range(1, max_rounds + 1):
        spec = await planner(ctx, feedback)
        out, warnings = apply_mapping(df, spec)
        issues, feedback_text = _problems(out, warnings)
        if not feedback_text:
            break
        feedback = feedback_text
    assert spec is not None

    # human-in-the-loop gate: rejected mappings are dropped (never silently kept)
    decisions: dict[str, bool] = {}
    items = items_for_review(spec)
    if items:
        decisions = approve(items)
        for m in spec.mappings:
            if m.target in decisions and not decisions[m.target]:
                m.source_column = None
                m.rationale = "rejected by human reviewer"
        out, warnings = apply_mapping(df, spec)
        issues, feedback_text = _problems(out, warnings)

    status = "ok" if not feedback_text else "needs_attention"
    return Result(spec=spec, out=out, issues=issues, rounds=rounds, status=status,
                  decisions=decisions, warnings=warnings)
