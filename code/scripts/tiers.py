"""Assign planned prompts to complexity tiers by their condition count (planner nodes plus generated count and pair nodes,
without quality), across sets. Writes data/tiers.json: per tier the (set, pid, tag, conds, split) rows. Example:
  python scripts/tiers.py --sets xl_seed:qwen3.7-plus-v5 cb_mi:qwen3.7-plus-v5 ...
--measure planner tiers by the planner's own requirement nodes only (no background, no generated count and pair nodes);
every row carries both numbers (conds, conds_planner) either way."""
import argparse
import json
import os

from common import P0, load_set, plan_dir

from draftdag.schema import Plan, add_generated_nodes

TIERS = (("<=15", 0, 15), ("16-25", 16, 25), ("26-35", 26, 35), ("36-50", 36, 50), (">50", 51, 10 ** 6))
ap = argparse.ArgumentParser()
ap.add_argument("--sets", nargs="+", required=True, help="set:tag pairs")
ap.add_argument("--out", default=os.path.join(P0, "data", "tiers.json"))
ap.add_argument("--measure", choices=("all", "planner"), default="all")
args = ap.parse_args()
rows = []
for spec in args.sets:
    s, tag = spec.split(":")
    pdir = plan_dir(s, tag)
    for r in load_set(s):
        p = os.path.join(pdir, r["id"] + ".json")
        if not os.path.exists(p):
            continue
        base = Plan.load(p)
        cp = sum(1 for v in base.order if base.typ(v) != "background")
        plan = add_generated_nodes(base)
        c = sum(1 for v in plan.order if plan.typ(v) != "quality")
        rows.append(dict(set=s, pid=r["id"], tag=tag, conds=c, conds_planner=cp, split=r.get("split")))
out = {}
for name, lo, hi in TIERS:
    key = "conds" if args.measure == "all" else "conds_planner"
    out[name] = [r for r in rows if lo <= r[key] <= hi]
summary = {name: dict(n=len(v), dev=sum(1 for r in v if r["split"] == "dev"), test=sum(1 for r in v if r["split"] == "test"), by_set={s: sum(1 for r in v if r["set"] == s) for s in sorted({r["set"] for r in v})}) for name, v in out.items()}
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(dict(summary=summary, measure=args.measure, tiers=out), f, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
