import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp_server"))


def test_mcp_tools_registered_and_work():
    import server

    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"standard_variables", "convert_units", "units_for", "validate_csv"} <= names
    assert server.convert_units("fpg", "mg/dL", [90.0])[0] == __import__("pytest").approx(4.995, abs=0.01)
