"""One-fact-per-node pass (atomize_pass) over existing plans, written to a new tag; the input tag of each prompt comes
from data/tiers.json (or --tag), so one call covers all sets. Logs every prompt's changes to <out dir>/_atomize.jsonl.
  python scripts/atomize_plans.py --out-tag qwen3.7-plus-v8 --workers 12                     # all sets, tags from tiers.json
  python scripts/atomize_plans.py --out-tag qwen3.7-plus-v8pilot --ids c_ma_016,lb_0374 --workers 2   # pilot
v8 (10-09 night): rewrites checked for lost words and redundancy, deletions judged on the node's own words, one repair
round (planner.atomize_pass); v7 outputs are void (the bundled node stayed beside its parts, deletions trusted the reason)."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from common import P0, plan_dir

from redag.planner import DEFAULT_MODEL, atomize_pass

ap = argparse.ArgumentParser()
ap.add_argument("--out-tag", required=True)
ap.add_argument("--tiers", default=os.path.join(P0, "data", "tiers.json"), help="source of (set, pid, tag) for every prompt")
ap.add_argument("--sets", default=None, help="comma list: only these sets")
ap.add_argument("--ids", default=None, help="comma list: only these prompt ids")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--max-nodes", type=int, default=128)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--no-repair", action="store_true", help="skip the repair round (ablation only)")
ap.add_argument("--version", default="v8", help="written to meta.planner as <model>:<version>")
args = ap.parse_args()

rows = [r for v in json.load(open(args.tiers, encoding="utf-8"))["tiers"].values() for r in v]
if args.sets:
    keep = set(args.sets.split(","))
    rows = [r for r in rows if r["set"] in keep]
if args.ids:
    keep = set(args.ids.split(","))
    rows = [r for r in rows if r["pid"] in keep]
rows = sorted(rows, key=lambda r: (r["set"], r["pid"]))
print("prompts:", len(rows), flush=True)


def one(r):
    src, dst = plan_dir(r["set"], r["tag"]), plan_dir(r["set"], args.out_tag)
    os.makedirs(dst, exist_ok=True)
    sp, dp = os.path.join(src, r["pid"] + ".json"), os.path.join(dst, r["pid"] + ".json")
    if os.path.exists(dp):
        return r["pid"], "exists", {}
    if not os.path.exists(sp):
        return r["pid"], "missing", {}
    d = json.load(open(sp, encoding="utf-8"))
    try:
        out, info = atomize_pass(d, model=args.model, max_nodes=args.max_nodes, repair=not args.no_repair)
    except Exception as ex:  # api failure: keep the input plan, record the error, the prompt can be re-run
        out, info = d, dict(error="api: " + str(ex)[:200])
    if (info.get("error") or "").startswith("api"):
        return r["pid"], "api_error", info
    out = json.loads(json.dumps(out))
    out["meta"] = dict(out.get("meta", {}), planner=f"{args.model}:{args.version}", base_tag=r["tag"])
    with open(dp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(dst, "_atomize.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(pid=r["pid"], set=r["set"], base_tag=r["tag"], n_before=len(d["nodes"]), n_after=len(out["nodes"]), **info), ensure_ascii=False) + "\n")
    print(r["pid"], len(d["nodes"]), "->", len(out["nodes"]), {k: info.get(k) for k in ("rewrites", "rewrites_kept", "new_nodes", "added", "set_count", "error")}, "deleted", len(info.get("deleted", [])),
          "repair", (info.get("repair") or {}).get("kept"), "/", len((info.get("repair") or {}).get("asked", [])), flush=True)
    return r["pid"], "ok", info


with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
    res = list(ex.map(one, rows))
st = {}
for _, s, _ in res:
    st[s] = st.get(s, 0) + 1
ok = [i for _, s, i in res if s == "ok"]
print(json.dumps(dict(out_tag=args.out_tag, status=st, rewrites=sum(i.get("rewrites", 0) for i in ok), rewrites_kept=sum(i.get("rewrites_kept", 0) for i in ok),
                      new_nodes=sum(i.get("new_nodes", 0) for i in ok), added=sum(i.get("added", 0) for i in ok), set_count=sum(i.get("set_count", 0) for i in ok),
                      deleted=sum(len(i.get("deleted", [])) for i in ok), reverted=sum(len(i.get("reverted", [])) for i in ok), ungrounded=sum(len(i.get("ungrounded", [])) for i in ok),
                      errors=sum(1 for i in ok if i.get("error")), api_errors=[pid for pid, s, _ in res if s == "api_error"],
                      outcomes=dict(__import__("collections").Counter(g.get("outcome") for i in ok for g in i.get("groups", []))),
                      repair_asked=sum(len((i.get("repair") or {}).get("asked", [])) for i in ok), repair_kept=sum((i.get("repair") or {}).get("kept", 0) for i in ok),
                      repair_errors=sum(1 for i in ok if (i.get("repair") or {}).get("error")))))
