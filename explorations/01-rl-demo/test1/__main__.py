from __future__ import annotations

import argparse
import json
import sys
import time
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
    if getattr(args, "policy", None):
        out["POLICY_BACKEND"] = args.policy
    if getattr(args, "policy_model", None):
        out["POLICY_MODEL"] = args.policy_model
    if getattr(args, "max_edits", None) is not None:
        out["TRAIN_MAX_EDITS"] = int(args.max_edits)
    if getattr(args, "no_node_review", False):
        out["NODE_REVIEW"] = False
    return out


def _setup(args: argparse.Namespace):
    from test0.backends import build_backends
    from test0.config import load_config
    from test0.store import Store

    from .policy import build_policy

    cfg = load_config(args.config, _overrides(args))
    out = Path(args.out or (Path(str(cfg.RUNS_DIR)) / f"test1-{time.strftime('%Y%m%d-%H%M%S')}"))
    out.mkdir(parents=True, exist_ok=True)
    store = Store(str(out / "store"), enabled=bool(cfg.STORE_ENABLED))
    backends = build_backends(cfg, store=store)
    policy = build_policy(cfg, backends, seed=int(args.seed or 0))
    if getattr(args, "policy_dir", None):
        policy.load(args.policy_dir)
    return cfg, backends, policy, out


def cmd_train(args: argparse.Namespace) -> int:
    from .stages import train

    cfg, backends, policy, out = _setup(args)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    try:
        result = train(int(args.stage), cfg, policy, backends, int(args.iters), int(args.group), str(out), case_ids=cases)
    finally:
        backends.close()
    last = result["history"][-1]["metrics"] if result["history"] else {}
    print(json.dumps({"out_dir": result["out_dir"], "stage": result["stage"], "iterations": len(result["history"]), "last": last}, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .stages import evaluate

    cfg, backends, policy, out = _setup(args)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    try:
        report = evaluate(int(args.stage), cfg, policy, backends, str(out), case_ids=cases, faults=int(args.faults), with_baseline=bool(args.with_baseline))
    finally:
        backends.close()
    print((Path(report["out_dir"]) / "report.md").read_text(encoding="utf-8"))
    return 0


def cmd_curriculum(args: argparse.Namespace) -> int:
    from .stages import curriculum

    cfg, backends, policy, out = _setup(args)
    iters = [int(v) for v in args.iters.split(",")]
    while len(iters) < 3:
        iters.append(iters[-1])
    cases = [c.strip() for c in args.cases.split(",") if c.strip()] if args.cases else None
    eval_cases = [c.strip() for c in args.eval_cases.split(",") if c.strip()] if args.eval_cases else cases
    try:
        summary = curriculum(cfg, backends, policy, iters[:3], int(args.group), str(out), case_ids=cases, eval_cases=eval_cases)
    finally:
        backends.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="test1", description="DraftDAG test1: policy optimisation on top of test0")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["mock", "api"], default=None)
    parser.add_argument("--render", choices=["auto", "playwright", "pil"], default=None)
    parser.add_argument("--policy", choices=["mock", "api", "hf"], default=None)
    parser.add_argument("--policy-model", dest="policy_model", default=None)
    parser.add_argument("--policy-dir", dest="policy_dir", default=None)
    parser.add_argument("--error-rate", dest="error_rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs-dir", dest="runs_dir", default=None)
    parser.add_argument("--store-dir", dest="store_dir", default=None)
    parser.add_argument("--max-edits", dest="max_edits", type=int, default=None)
    parser.add_argument("--no-node-review", dest="no_node_review", action="store_true")
    parser.add_argument("--out", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p_train.add_argument("--iters", type=int, default=8)
    p_train.add_argument("--group", type=int, default=4)
    p_train.add_argument("--cases", default=None)
    p_train.set_defaults(func=cmd_train)
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--stage", type=int, choices=[1, 2, 3], default=3)
    p_eval.add_argument("--cases", default=None)
    p_eval.add_argument("--faults", type=int, default=0)
    p_eval.add_argument("--with-baseline", dest="with_baseline", action="store_true")
    p_eval.set_defaults(func=cmd_eval)
    p_cur = sub.add_parser("curriculum")
    p_cur.add_argument("--iters", default="8,8,8")
    p_cur.add_argument("--group", type=int, default=4)
    p_cur.add_argument("--cases", default=None)
    p_cur.add_argument("--eval-cases", dest="eval_cases", default=None)
    p_cur.set_defaults(func=cmd_curriculum)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
