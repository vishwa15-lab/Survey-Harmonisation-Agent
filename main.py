"""CLI:  python -m harmonizer.cli run A   |   python -m harmonizer.cli eval --oracle"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from core.checks import format_issues
from evals.evals import score
from agent.pipeline import auto_approve, cli_approve, harmonize, make_agent_planner, oracle_planner
from evals.synthetic import SURVEY_IDS, make_survey


def _load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


async def _run_one(style: str, args, planner_factory) -> dict:
    sv = make_survey(style, n=args.n, seed=args.seed)
    planner = oracle_planner(sv.truth_spec) if args.oracle else planner_factory()
    approve = auto_approve if (args.auto_approve or args.oracle) else cli_approve
    res = await harmonize(sv.raw, sv.codebook, sv.survey_id, planner, approve=approve)
    sc = score(res.out, sv.truth_df, res.spec, sv.truth_spec)

    outdir = Path(args.out) / style
    outdir.mkdir(parents=True, exist_ok=True)
    res.out.to_csv(outdir / "harmonized.csv", index=False)
    (outdir / "mapping.json").write_text(res.spec.model_dump_json(indent=2))
    (outdir / "issues.json").write_text(json.dumps([i.model_dump() for i in res.issues], indent=2))
    print(f"\n=== Survey {style}: status={res.status} rounds={res.rounds} "
          f"mapping={sc['mapping_acc']:.0%} units={sc['unit_acc']:.0%} values={sc['value_acc']:.0%}")
    if res.issues:
        print(format_issues(res.issues))
    return dict(survey=style, status=res.status, rounds=res.rounds,
                **{k: sc[k] for k in ("mapping_acc", "unit_acc", "value_acc")})


def main() -> None:
    _load_env()
    ap = argparse.ArgumentParser(prog="harmonizer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "eval"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("survey", choices=SURVEY_IDS)
        p.add_argument("--n", type=int, default=400)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--oracle", action="store_true", help="use ground-truth mapping (no API calls)")
        p.add_argument("--auto-approve", action="store_true", help="skip the human approval prompt")
        p.add_argument("--out", default="runs")
    args = ap.parse_args()

    styles = [args.survey] if args.cmd == "run" else SURVEY_IDS

    async def go() -> list[dict]:
        factory = make_agent_planner if not args.oracle else (lambda: None)
        return [await _run_one(s, args, factory) for s in styles]

    rows = asyncio.run(go())
    if args.cmd == "eval":
        Path(args.out).mkdir(exist_ok=True)
        (Path(args.out) / "eval_results.json").write_text(json.dumps(rows, indent=2))
        print("\nsurvey  status           rounds  mapping  units  values")
        for r in rows:
            print(f"{r['survey']:<7} {r['status']:<16} {r['rounds']:<7} {r['mapping_acc']:<8.0%} "
                  f"{r['unit_acc']:<6.0%} {r['value_acc']:.0%}")


if __name__ == "__main__":
    main()
