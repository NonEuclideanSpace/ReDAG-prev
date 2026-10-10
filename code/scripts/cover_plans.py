"""Extra planner pass on an existing plan tag, written to a new tag.
  --mode cover (default): ask the planner for nodes covering stated facts the first pass left out (coverage_pass).
  --mode facts: the planner lists the description's facts without seeing the plan, matches them to the plan's
                requirements and writes nodes only for the unmatched ones (facts_pass); for long prose prompts.
Examples:
  python scripts/cover_plans.py --set xl_seed --tag qwen3.7-plus-v4f --out-tag qwen3.7-plus-v5 --workers 8
  python scripts/cover_plans.py --set lb_t2i --tag qwen3.7-plus-v5 --out-tag qwen3.7-plus-v6 --mode facts --max-nodes 96 --workers 8
  --ids lb_0001,lb_0002 restricts the pass to those prompts (a pilot); --ids-from <audit gate json> takes its sample."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from common import P0, load_set, plan_dir, shard

from redag.planner import DEFAULT_MODEL, coverage_pass, facts_pass

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out-tag", required=True)
ap.add_argument("--mode", default="cover", choices=["cover", "facts"])
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--max-nodes", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--shard", default=None)
ap.add_argument("--ids", default=None, help="comma-separated prompt ids to process (pilot)")
ap.add_argument("--ids-from", default=None, help="gate json whose 'per' sample gives the ids")
args = ap.parse_args()

src, dst = plan_dir(args.set, args.tag), plan_dir(args.set, args.out_tag)
os.makedirs(dst, exist_ok=True)
rows = shard(load_set(args.set)[: args.limit], args.shard)
keep = None
if args.ids:
    keep = set(args.ids.split(","))
if args.ids_from:
    keep = (keep or set()) | {p["pid"] for p in json.load(open(args.ids_from, encoding="utf-8"))["per"]}
if keep is not None:
    rows = [r for r in rows if r["id"] in keep]
version = "v6" if args.mode == "facts" else "v5"


def one(r):
    sp, dp = os.path.join(src, r["id"] + ".json"), os.path.join(dst, r["id"] + ".json")
    if os.path.exists(dp):
        return r["id"], "exists", 0, 0, {}
    if not os.path.exists(sp):
        return r["id"], "missing", 0, 0, {}
    d = json.load(open(sp, encoding="utf-8"))
    if args.mode == "facts":
        out, info = facts_pass(d, model=args.model, max_nodes=args.max_nodes)
        added, dropped = info.get("added", 0), info.get("dropped", 0)
    else:
        out, added, dropped = coverage_pass(d, model=args.model, max_nodes=args.max_nodes)
        info = {}
    out["meta"] = dict(out.get("meta", {}), planner=f"{args.model}:{version}", base_tag=args.tag)
    with open(dp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if info:
        with open(os.path.join(dst, "_facts_pass.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(pid=r["id"], **info), ensure_ascii=False) + "\n")
    print(r["id"], "added", added, "dropped", dropped, {k: v for k, v in info.items() if k in ("n_facts", "n_uncovered", "error")}, flush=True)
    return r["id"], "ok", added, dropped, info


with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
    res = list(ex.map(one, rows))
n_ok = sum(1 for _, st, _, _, _ in res if st == "ok")
errors = {}
for _, st, _, _, info in res:
    if info.get("error"):
        errors[info["error"]] = errors.get(info["error"], 0) + 1
print(json.dumps(dict(set=args.set, out_tag=args.out_tag, mode=args.mode, done=n_ok, exists=sum(1 for _, st, _, _, _ in res if st == "exists"), missing=[pid for pid, st, _, _, _ in res if st == "missing"], added=sum(a for _, _, a, _, _ in res), dropped=sum(dd for _, _, _, dd, _ in res), mean_added=(sum(a for _, _, a, _, _ in res) / max(1, n_ok)), errors=errors)))
