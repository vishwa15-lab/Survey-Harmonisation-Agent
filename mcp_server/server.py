"""Optional MCP server exposing the deterministic harmonization tools.

Run:  python mcp_server/server.py      (stdio transport)
Any MCP client (including the OpenAI Agents SDK's MCPServerStdio) can then call these tools,
so the tools are not tied to one framework.
"""
from __future__ import annotations

import pandas as pd
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from core.checks import run_checks
from core.mapping import MappingSpec, apply_mapping
from agent.tools import core_standard_variables
from core.units import allowed_units, to_standard

mcp = _Server("survey-harmonizer")


@mcp.tool()
def standard_variables() -> list[dict]:
    """List standard variables, standard units, allowed source units and plausible ranges."""
    return core_standard_variables()


@mcp.tool()
def convert_units(target: str, unit: str, values: list[float]) -> list[float]:
    """Convert numeric values from `unit` into the standard unit of `target` (e.g. fpg, mg/dL)."""
    return [float(v) for v in to_standard(target, pd.Series(values, dtype="float64"), unit)]


@mcp.tool()
def units_for(target: str) -> list[str]:
    """Allowed source units for a standard variable."""
    return allowed_units(target)


@mcp.tool()
def validate_csv(csv_path: str, mapping_json_path: str) -> list[dict]:
    """Apply a saved MappingSpec (JSON) to a raw CSV and return validator issues."""
    df = pd.read_csv(csv_path)
    spec = MappingSpec.model_validate_json(open(mapping_json_path).read())
    out, warnings = apply_mapping(df, spec)
    issues = [i.model_dump() for i in run_checks(out)]
    return issues + [dict(variable="-", severity="warning", check="apply", message=w) for w in warnings]


if __name__ == "__main__":
    mcp.run()
