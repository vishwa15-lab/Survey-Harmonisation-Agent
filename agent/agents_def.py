"""Agent definitions (OpenAI Agents SDK).

Model names are read from the environment on purpose: names and prices change,
so nothing is hard-coded. See .env.example.
"""
from __future__ import annotations

import os

from agents import Agent

from core.mapping import MappingSpec
from agent.tools import PLANNER_TOOLS, SurveyContext

PLANNER_INSTRUCTIONS = """\
You harmonize one health survey into a fixed standard schema. You output a MappingSpec.

Workflow:
1. Call list_standard_variables to see targets, allowed source units and plausible ranges.
2. Call list_columns, then search_codebook per variable to find the right raw column.
   Use describe_column to verify units and coding against the actual values.
3. Draft the full MappingSpec (one entry for EVERY standard variable) and call try_mapping.
   Fix every 'error' it reports, then call it again until there are no errors.
4. Return the final MappingSpec.

Rules:
- Use only columns that exist. Never invent a column or a unit.
- source_unit must be one of the allowed_source_units for that target. Read the codebook for the unit;
  do not infer it from values alone when the codebook states it.
- Binary variables need a recode entry for EVERY raw value that can occur (e.g. '1'->1, '2'->0).
  Check the direction of coding carefully: some surveys code 1 = female or 2 = no.
- If a variable was not collected, set source_column to null with confidence 1.0 and say so in the rationale.
- If a question is only asked of a subgroup (e.g. medication asked only if diagnosed), set
  na_to_zero_if_zero to the gating standard variable (e.g. 'htn_dx').
- missing_codes lists raw values meaning 'missing' (e.g. '999') when the codebook says so.
- Set needs_review=true and lower confidence for any real judgment call. Be honest about uncertainty.
- Keep each rationale to one or two sentences and cite the codebook line you used.
"""


def planner_model() -> str:
    name = os.environ.get("HARMONIZER_MODEL_PLANNER", "").strip()
    if not name:
        raise RuntimeError(
            "Set HARMONIZER_MODEL_PLANNER to an OpenAI model name (see .env.example). "
            "It is not defaulted because model names and prices change."
        )
    return name


def build_planner(model: str | None = None) -> Agent[SurveyContext]:
    return Agent[SurveyContext](
        name="Planner",
        instructions=PLANNER_INSTRUCTIONS,
        tools=PLANNER_TOOLS,
        output_type=MappingSpec,
        model=model or planner_model(),
    )
