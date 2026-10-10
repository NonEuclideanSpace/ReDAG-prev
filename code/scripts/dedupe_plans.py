"""Write a plan tag with its twin nodes removed (redag.dedupe: twin child questions, and the entity merges a reviewed
config lists), without any model call. Every prompt of --tiers gets a plan under --out-tag: a copy of the --src-tag plan
when nothing was removed. Per-plan records go to <set>/<out-tag>/_dedupe.jsonl; the plan's meta gets "dedupe".
  python scripts/dedupe_plans.py --src-tag qwen3.7-plus-v84 --out-tag qwen3.7-plus-v85 --merges configs/dedupe_merges_v85.json"""
import argparse
import json
import os

from common import P0, plan_dir

from redag.dedupe import dedupe_plan

ap = argparse.ArgumentParser()
ap.add_argument("--src-tag", required=True)
ap.add_argument("--out-tag", required=True)
ap.add_argument("--merges", default=None, help="json: pid -> list of groups [id, id, ..., reason]")
ap.add_argument("--tiers", default=os.path.join(P0, "data", "tiers_v84.json"))
ap.add_argument("--sets", default=None)
ap.add_argument("--max-nodes", type=int, default=128)
args = ap.parse_args()

merges = json.load(open(args.merges, encoding="utf-8")) if args.merges else {}
rows = [r for v in json.load(open(args.tiers, encoding="utf-8"))["tiers"].values() for r in v]
if args.sets:
    rows = [r for r in rows if r["set"] in set(args.sets.split(","))]
tot = dict(prompts=0, changed=0, merged=0, refused=0, dropped=0, errors=0, missing=0, nodes_before=0, nodes_after=0)
used = set()
for r in rows:
    src = os.path.join(plan_dir(r["set"], args.src_tag), r["pid"] + ".json")
    if not os.path.exists(src):
        tot["missing"] += 1
        continue
    d = json.load(open(src, encoding="utf-8"))
    groups = [g[:-1] for g in merges.get(r["pid"], [])]  # the last item of a group is its reason
    if groups:
        used.add(r["pid"])
    out, info = dedupe_plan(d, groups, max_nodes=args.max_nodes)
    tot["prompts"] += 1
    tot["nodes_before"] += info["n_before"]
    tot["nodes_after"] += info["n_after"]
    tot["merged"] += len(info["merged"])
    tot["refused"] += len(info["refused"])
    tot["dropped"] += len(info["dropped"])
    if info["error"]:
        tot["errors"] += 1
        print(r["pid"], "ERROR", info["error"], flush=True)
    changed = out is not d
    if changed:
        tot["changed"] += 1
        out = json.loads(json.dumps(out))
        out["meta"] = dict(out.get("meta", {}), planner=str((d.get("meta") or {}).get("planner", "")) + "+dedupe", dedupe_of=args.src_tag,
                           dedupe=dict(merged=[(m["drop"], m["keep"]) for m in info["merged"]], dropped=[(x["id"], x["twin"]) for x in info["dropped"]]))
    dst = plan_dir(r["set"], args.out_tag)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, r["pid"] + ".json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(dst, "_dedupe.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(pid=r["pid"], set=r["set"], src_tag=args.src_tag, changed=changed, **info), ensure_ascii=False) + "\n")
    if changed or info["refused"]:
        print(r["pid"], f"{info['n_before']}->{info['n_after']}", "merged", [(m["drop"], m["keep"]) for m in info["merged"]], "dropped", len(info["dropped"]),
              "refused", [x.get("ids", x.get("group")) for x in info["refused"]], flush=True)
unused = sorted(set(merges) - {"_about"} - used)
if unused:
    print("merge groups for prompts not in --tiers:", unused)
print(json.dumps(tot))
