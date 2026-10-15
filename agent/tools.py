"""Function tools the Planner agent can call.

Plain functions live in `core_*` so the MCP server and tests can reuse them;
`@function_tool` wrappers add the SDK context plumbing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
from agents import RunContextWrapper, function_tool

from core.checks import Issue, format_issues, run_checks
from core.mapping import MappingSpec, apply_mapping
from core.schema import STANDARD_VARS
from core.units import allowed_units


@dataclass
class SurveyContext:
    survey_id: str
    df: pd.DataFrame
    codebook: dict[str, str]
    last_issues: list[Issue] = field(default_factory=list)


# ---------- plain functions (reusable, testable, MCP-friendly) ----------

def core_standard_variables() -> list[dict]:
    return [
        dict(name=v.name, description=v.description, kind=v.kind, standard_unit=v.unit,
             allowed_source_units=allowed_units(v.name), plausible_range=[v.lo, v.hi])
        for v in STANDARD_VARS.values()
    ]


def core_list_columns(df: pd.DataFrame) -> list[dict]:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append(dict(column=c, dtype=str(s.dtype), n_missing=int(s.isna().sum()),
                         n_unique=int(s.nunique()),
                         examples=[str(x) for x in s.dropna().unique()[:5]]))
    return rows


def core_search_codebook(codebook: dict[str, str], query: str, top: int = 6) -> list[dict]:
    """Keyword search (stand-in for a vector store; swap in embeddings later)."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for col, text in codebook.items():
        toks = set(re.findall(r"[a-z0-9]+", f"{col} {text}".lower()))
        score = len(q & toks)
        if score:
            scored.append((score, col, text))
    scored.sort(reverse=True)
    return [dict(column=c, description=t) for _, c, t in scored[:top]]


def core_describe_column(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return dict(error=f"no such column {column!r}")
    s = df[column]
    out: dict = dict(column=column, dtype=str(s.dtype), n_missing=int(s.isna().sum()))
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        out.update(min=float(num.min()), median=float(num.median()), max=float(num.max()),
                   value_counts_top=num.value_counts().head(8).to_dict() if num.nunique() <= 12 else None)
    else:
        out["value_counts_top"] = s.value_counts().head(8).to_dict()
    return out


def core_try_mapping(ctx: SurveyContext, spec: MappingSpec) -> dict:
    out, warnings = apply_mapping(ctx.df, spec)
    issues = run_checks(out)
    ctx.last_issues = issues
    return dict(warnings=warnings, n_errors=sum(i.severity == "error" for i in issues),
                report=format_issues(issues))


# ---------- SDK tool wrappers ----------

@function_tool
def list_standard_variables() -> list[dict]:
    """List the standard variables to map to, with allowed source units and plausible ranges."""
    return core_standard_variables()


@function_tool
def list_columns(ctx: RunContextWrapper[SurveyContext]) -> list[dict]:
    """List every raw column with dtype, missing count, unique count and example values."""
    return core_list_columns(ctx.context.df)


@function_tool
def search_codebook(ctx: RunContextWrapper[SurveyContext], query: str) -> list[dict]:
    """Search the survey codebook for columns matching a topic, e.g. 'glucose' or 'blood pressure medication'."""
    return core_search_codebook(ctx.context.codebook, query)


@function_tool
def describe_column(ctx: RunContextWrapper[SurveyContext], column: str) -> dict:
    """Summary statistics or top values for one raw column. Use to verify units and coding."""
    return core_describe_column(ctx.context.df, column)


@function_tool
def try_mapping(ctx: RunContextWrapper[SurveyContext], spec: MappingSpec) -> dict:
    """Apply a draft mapping to the data and run the validator. Fix every 'error' before finalizing."""
    return core_try_mapping(ctx.context, spec)


PLANNER_TOOLS = [list_standard_variables, list_columns, search_codebook, describe_column, try_mapping]
