import argparse
import json
import os

from common import P0, ids_filter, load_set, plan_dir, shard

from redag.planner import DEFAULT_MODEL, load_or_plan
from redag.schema import Plan, add_generated_nodes

ap = argparse.ArgumentParser(description="Make DAG plans for a prompt set (planner v4).")
ap.add_argument("--set", required=True)
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--tag", default=None, help="plan directory tag, default <model>-v4")
ap.add_argument("--canvas", type=int, default=1024)
ap.add_argument("--max-nodes", type=int, default=64, help="planner-written nodes per prompt; LongBench-T2I prompts may need more")
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--shard", default=None)
ap.add_argument("--ids-file", default=None, help="only these prompt ids (one per line), e.g. the prompts a set was extended by")
args = ap.parse_args()

tag = args.tag or f"{args.model}-v4"
pdir = plan_dir(args.set, tag)
rows = shard(ids_filter(load_set(args.set), args.ids_file)[: args.limit], args.shard)
stats = dict(ok=0, failed=0, types={}, edges=0, depth=[], nodes=[], conds=[])
for r in rows:
    plan, errs = load_or_plan(r["prompt"], r["id"], pdir, W=args.canvas, H=args.canvas, model=args.model, max_nodes=args.max_nodes)
    if plan is None:
        stats["failed"] += 1
        print("FAILED", r["id"], errs[:3], flush=True)
        continue
    stats["ok"] += 1
    full = add_generated_nodes(Plan(plan.to_dict()))
    for v in plan.order:
        t = plan.typ(v)
        stats["types"][t] = stats["types"].get(t, 0) + 1
    stats["edges"] += len(plan.edges)
    stats["depth"].append(max(full.levels().values()))
    stats["nodes"].append(len(plan.order) - 1)
    stats["conds"].append(sum(1 for v in full.order if full.typ(v) != "quality"))
    print(r["id"], "nodes", len(plan.order) - 1, "edges", len(plan.edges), flush=True)
n = max(1, stats["ok"])
summary = dict(set=args.set, tag=tag, ok=stats["ok"], failed=stats["failed"], types=stats["types"], mean_nodes=sum(stats["nodes"]) / n, mean_edges=stats["edges"] / n, mean_depth=sum(stats["depth"]) / n, cond_tiers={k: sum(1 for c in stats["conds"] if lo <= c <= hi) for k, (lo, hi) in (("<=15", (0, 15)), ("16-25", (16, 25)), ("26-35", (26, 35)), ("36-50", (36, 50)), (">50", (51, 10 ** 6)))}, mean_conds=sum(stats["conds"]) / n, frac_non_entity=(sum(v for k, v in stats["types"].items() if k in ("attribute", "state", "action", "relation")) / max(1, sum(v for k, v in stats["types"].items() if k != "background"))))
os.makedirs(pdir, exist_ok=True)
part = "" if not (args.shard or args.ids_file) else "_" + "_".join(x for x in (os.path.basename(args.ids_file or "").rsplit(".", 1)[0], (args.shard or "").replace("/", "of")) if x)
with open(os.path.join(pdir, f"_summary{part}.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False))
