import argparse
import json
import os
import time

from common import P0, add_vlm_args, ids_filter, load_json, load_set, make_vlm, plan_dir, run_dir, shard

from draftdag.loop import Loop, LoopConfig
from draftdag.schema import Plan, add_generated_nodes
from draftdag.stats import Stats

ap = argparse.ArgumentParser(description="Run the DAG loop (stage 1 drafts, optional stage 2 repair) over a prompt set.")
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True, help="plan directory tag")
ap.add_argument("--run", required=True, help="run name under runs/")
ap.add_argument("--config", default=os.path.join(P0, "configs", "loop.default.json"))
ap.add_argument("--mode", default=None, choices=["adaptive", "draw", "repair", "draw_repair"])
ap.add_argument("--n0", type=int, default=None)
ap.add_argument("--n-max", type=int, default=None)
ap.add_argument("--calls-max", type=float, default=None)
ap.add_argument("--seconds-max", type=float, default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--shard", default=None)
ap.add_argument("--ids-file", default=None, help="only prompts listed in this file (one id per line), e.g. data/tier_picks/<tier>.txt")
ap.add_argument("--executor", default="qi21", choices=["qi21", "mock"])
ap.add_argument("--lora", default=None)
ap.add_argument("--qi-steps", type=int, default=40)
ap.add_argument("--qi-res", type=int, default=1024)
ap.add_argument("--stats", default=None, help="stats json from a previous run (priorities, protection, decision)")
ap.add_argument("--drafts-from", default=None, help="scripts/draw.py run (name or drafts.jsonl): stage 1 takes its drafts in order, so every mode starts from the same drafts")
ap.add_argument("--skip-preflight", action="store_true", help="mock executor only")
add_vlm_args(ap)
args = ap.parse_args()

cfg_d = load_json(args.config, {})
for k in ("mode", "n0", "n_max", "calls_max", "seconds_max", "seed"):
    v = getattr(args, k)
    if v is not None:
        cfg_d[k] = v
if args.stats:
    cfg_d["stats_path"] = args.stats
cfg = LoopConfig(**cfg_d)
out = os.path.join(P0, "runs", args.run)
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "args.json"), "w", encoding="utf-8") as f:
    json.dump(dict(vars(args), config=cfg.to_dict()), f, ensure_ascii=False, indent=1)

rows = shard(ids_filter(load_set(args.set), args.ids_file)[: args.limit], args.shard)
pdir = plan_dir(args.set, args.tag)
plans = []
for r in rows:
    p = os.path.join(pdir, r["id"] + ".json")
    if os.path.exists(p):
        plans.append(add_generated_nodes(Plan.load(p)))
print(f"{len(plans)} plans", flush=True)

if args.executor == "mock":
    from draftdag.conditions import build_conditions
    from draftdag.mock import MockExecutor, MockVLM

    assert plans
    ex = MockExecutor(plans[0], seed=cfg.seed)
    vlm = MockVLM(plans[0], build_conditions(plans[0]), seed=cfg.seed)
else:
    if not args.skip_preflight:
        from draftdag.preflight import require_preflight

        require_preflight()
    if args.no_gate:
        raise SystemExit("the loop never runs without an evaluator gate; run scripts/calibrate_eval.py first")
    vlm = make_vlm(args, require_gate=True)
    from draftdag.executor_qi21 import QI21

    ex = QI21(steps=args.qi_steps, res=args.qi_res)
    if args.lora:
        ex.load_lora(args.lora)

drafts_jsonl = None
if args.drafts_from:
    from draftdag.executor import PreDrawn

    drafts_jsonl = args.drafts_from if args.drafts_from.endswith(".jsonl") else os.path.join(run_dir(args.drafts_from), "drafts.jsonl")
    ex = PreDrawn(ex, drafts_jsonl, size=cfg.size)

loop = Loop(ex, vlm, cfg, out, stats=Stats.load(args.stats))
t0 = time.time()
for i, plan in enumerate(plans):
    if args.executor == "mock":
        from draftdag.conditions import build_conditions
        from draftdag.mock import MockExecutor, MockVLM

        loop.ex = MockExecutor(plan, seed=cfg.seed)
        if drafts_jsonl:
            loop.ex = PreDrawn(loop.ex, drafts_jsonl, size=cfg.size)
        loop.vlm = MockVLM(plan, build_conditions(plan), seed=cfg.seed)
    s = loop.run_prompt(plan)
    print(i, s["pid"], "all_pass", s["all_pass"], "pass", f"{s['n_pass']}/{s['n_cond']}", "rounds", s["rounds"], "calls", round(s["calls"], 1), "elapsed", round(time.time() - t0), flush=True)
print("DONE", flush=True)
