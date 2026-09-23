from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _overrides(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if getattr(args, "mode", None):
        out["MODE"] = args.mode
    if getattr(args, "render", None):
        out["RENDER_BACKEND"] = args.render
    if getattr(args, "error_rate", None) is not None:
        out["MOCK_ERROR_RATE"] = float(args.error_rate)
    if getattr(args, "seed", None) is not None:
        out["MOCK_SEED"] = int(args.seed)
    if getattr(args, "runs_dir", None):
        out["RUNS_DIR"] = args.runs_dir
    if getattr(args, "store_dir", None):
        out["STORE_DIR"] = args.store_dir
    if getattr(args, "no_store", False):
        out["STORE_ENABLED"] = False
    if getattr(args, "no_node_review", False):
        out["NODE_REVIEW"] = False
    if getattr(args, "generate", None):
        out["GENERATE_BACKEND"] = args.generate
    return out


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_cli

    hint = None
    if args.hint:
        hint = json.loads(Path(args.hint).read_text(encoding="utf-8"))
    elif args.case:
        from .bench_cases import case_by_id

        hint = case_by_id(args.case)["hint"]
        if not args.prompt:
            args.prompt = case_by_id(args.case)["prompt"]
    if not args.prompt:
        print("a prompt or --case is required", file=sys.stderr)
        return 2
    result = run_cli(args.prompt, args.config, _overrides(args), run_dir=args.out, hint=hint)
    print(json.dumps({
        "run_dir": result["run_dir"],
        "scene_passed": result["scene_passed"],
        "final_passed": result["final_passed"],
        "scene_rounds": result["scene_rounds"],
        "final_rounds": result["final_rounds"],
        "final_png": result["final_png"],
        "draft_png": result["draft_png"],
        "reuse": result["reuse"],
        "cost_units": result["cost"]["units"],
        "cost_by_phase": {k: v["units"] for k, v in result["cost"]["by_phase"].items()},
        "repairs": result["repairs"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import run_bench
    from .config import load_config

    cfg = load_config(args.config, _overrides(args))
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    report = run_bench(cfg, case_ids=case_ids, systems=systems, faults=int(args.faults), out_dir=args.out, seed=int(args.seed or 0), oracle_plan=bool(args.oracle_plan))
    print((Path(report["out_dir"]) / "report.md").read_text(encoding="utf-8"))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import build_report, to_markdown

    data = json.loads((Path(args.bench_dir) / "bench.json").read_text(encoding="utf-8"))
    report = build_report(data["records"])
    text = to_markdown(report)
    (Path(args.bench_dir) / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_cases(args: argparse.Namespace) -> int:
    from .bench_cases import CASES

    for case in CASES:
        print(f"{case['id']:18s} {len(case['hint']['nodes']):2d} nodes  {len(case['checks'])} checks  {case['prompt']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="test0", description="DraftDAG test0: DAG environment without training")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["mock", "api"], default=None)
    parser.add_argument("--render", choices=["auto", "playwright", "pil"], default=None)
    parser.add_argument("--generate", choices=["mock", "remote"], default=None)
    parser.add_argument("--error-rate", dest="error_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--runs-dir", dest="runs_dir", default=None)
    parser.add_argument("--store-dir", dest="store_dir", default=None)
    parser.add_argument("--no-store", dest="no_store", action="store_true")
    parser.add_argument("--no-node-review", dest="no_node_review", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("prompt", nargs="?", default=None)
    p_run.add_argument("--case", default=None)
    p_run.add_argument("--hint", default=None)
    p_run.add_argument("--out", default=None)
    p_run.set_defaults(func=cmd_run)
    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--cases", default=None)
    p_bench.add_argument("--systems", default="dag,baseline")
    p_bench.add_argument("--faults", default=2)
    p_bench.add_argument("--out", default=None)
    p_bench.add_argument("--oracle-plan", dest="oracle_plan", action="store_true")
    p_bench.set_defaults(func=cmd_bench)
    p_report = sub.add_parser("report")
    p_report.add_argument("bench_dir")
    p_report.set_defaults(func=cmd_report)
    p_cases = sub.add_parser("cases")
    p_cases.set_defaults(func=cmd_cases)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
