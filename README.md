# Survey Harmonization Agent

An agent system that maps messy, country-specific health survey files onto a fixed standard
schema (variable names, units, coding), validates the result, and asks a human about judgment
calls. Built on the **OpenAI Agents SDK** (pinned `0.23.1`; the SDK is pre-1.0, so upgrade deliberately).

Uses **synthetic data only**. No real survey data, internal documents or restricted datasets are
included, and none should be committed.

## Design in one picture

```
raw survey + codebook
        |
   [Planner agent]  --tools-->  list_columns, describe_column, search_codebook,
        |                       list_standard_variables, try_mapping
        v
   MappingSpec (declarative JSON, strict schema)
        |
   [apply_mapping]  deterministic executor (units, recodes, missing codes, skip patterns)
        |
   [Validator]      deterministic checks -> issues + hints
        |  errors? --> feedback to Planner (same SQLite session), up to N rounds
        v
   [Human gate]     judgment calls (e.g. "treated" variables) must be approved
        v
   harmonized.csv + mapping.json + issues.json
```

**Key decision:** the model proposes a *declarative* mapping; plain code applies it. The LLM never
executes code on the data, every transformation is auditable and replayable, and the Validator's
quality does not depend on the model.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`HARMONIZER_MODEL_PLANNER` is deliberately not defaulted: model names and prices change, so pick
one from OpenAI's current models page. Start with a cheaper model and compare before moving up.

## Run

```bash
# no API calls: uses the ground-truth mapping to prove the pipeline and scoring work
python -m harmonizer.cli eval --oracle

# real agent run on one synthetic survey (needs OPENAI_API_KEY + model name)
python -m harmonizer.cli run A

# all three surveys, non-interactive approval
python -m harmonizer.cli eval --auto-approve
```

Outputs go to `runs/<survey>/` (`harmonized.csv`, `mapping.json`, `issues.json`) and
`runs/eval_results.json`.

## The three synthetic surveys

| Survey | What makes it hard |
|---|---|
| A | cryptic column names, mg/dL lipids and glucose, 1/2 yes-no coding, HbA1c not collected |
| B | M/F and Y/N letter codes, height in inches, weight in lb, HbA1c in mmol/mol |
| C | reversed sex coding (1 = female), height in metres, 999 as a missing code |

Weight in lb is a deliberate trap: 170 lb is a plausible kg value, so the Validator cannot catch
it. Only reading the codebook can.

## Metrics

`mapping_acc` (right raw column), `unit_acc` (right source unit), `value_acc` (harmonized values
match ground truth: continuous within 3%, binary exact).

## Status: what is and is not verified

| Part | Status |
|---|---|
| Schema, units, mapping executor, Validator, synthetic data, scoring | Unit-tested (25 tests passing) |
| Pipeline loop, feedback retries, human rejection, MCP tools | Tested with injected planners (no API) |
| Agent definition and tool schemas | Built and schema-checked against the SDK |
| **Live LLM runs (Planner calling OpenAI)** | **Not yet run.** Needs your API key; expect to tune the instructions |

## Roadmap

1. Run live on survey A, then B and C; record `eval_results.json` as your baseline.
2. Swap `search_codebook` keyword search for embeddings (Chroma/FAISS) via LlamaIndex.
3. Connect the MCP server (`mcp_server/server.py`) to the Planner with `MCPServerStdio`.
4. Add an LLM-as-judge for rationale quality; compare small vs large planner models.
5. Add NHANES as a real public dataset (check its terms; do not commit raw data).
6. Optional: LangGraph comparison branch; Docker; demo video.

## Layout

```
src/harmonizer/  schema, units, mapping, checks, synthetic, tools, agents_def, pipeline, evals, cli
mcp_server/      optional MCP server wrapping the deterministic tools
tests/           25 tests
```
