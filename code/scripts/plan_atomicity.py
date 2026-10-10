"""One fact per node, measured without any model: for every plan, (1) nodes the simple bundling check flags (planner.
likely_bundled: an entity question longer than seven words or with a list or a second object; a background, attribute,
state or absent question with a list or a joined look), (2) redundant parents: a node such that at least two other nodes
about the same subject each ask only words this node already asks (the bundled node left beside its parts), (3)
duplicate pairs: two nodes about the same subject with the same content words. Compares tags on the same prompts.
  python scripts/plan_atomicity.py --tags qwen3.7-plus-v5,qwen3.7-plus-v8pilot --ids c_ma_001,... --out ../gates/atomicity_v8pilot.json
  python scripts/plan_atomicity.py --tags BASE,qwen3.7-plus-v8      # BASE = each prompt's tag in data/tiers.json"""
import argparse
import json
import os

from common import P0, plan_dir

from draftdag.planner import SPLIT_FREE, _content, _names, _of_ids, likely_bundled

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True, help="comma list; BASE = the prompt's tag in data/tiers.json")
ap.add_argument("--tiers", default=os.path.join(P0, "data", "tiers.json"))
ap.add_argument("--ids", default=None)
ap.add_argument("--sets", default=None)
ap.add_argument("--fallback", default=None, help="tag:fallback_tag, used for a prompt whose plan under tag is missing")
ap.add_argument("--out", default=None)
args = ap.parse_args()

rows = [r for v in json.load(open(args.tiers, encoding="utf-8"))["tiers"].values() for r in v]
if args.sets:
    rows = [r for r in rows if r["set"] in set(args.sets.split(","))]
if args.ids:
    rows = [r for r in rows if r["pid"] in set(args.ids.split(","))]
fb = dict(x.split(":", 1) for x in args.fallback.split(",")) if args.fallback else {}
CHILD = ("attribute", "state", "relation", "action")


def subject(n):
    t = n.get("type")
    if t == "entity":
        return {n["id"]}
    if t in CHILD:
        return set(_of_ids(n))
    return {t}


def measure(d):
    nodes = [n for n in d["nodes"]]
    words = {}
    for n in nodes:
        names = _names(nodes, _of_ids(n)) if n.get("type") in CHILD else set()
        words[n["id"]] = _content(n.get("question", ""), SPLIT_FREE) - names
    parents, dups = [], []
    for n in nodes:
        if n.get("type") not in ("entity",) + CHILD:
            continue
        s, w = subject(n), words[n["id"]]
        if len(w) < 2:
            continue
        subs = [m["id"] for m in nodes if m["id"] != n["id"] and m.get("type") in CHILD and (subject(m) & (s | {n["id"]})) and words[m["id"]] and words[m["id"]] < w]
        if len(subs) >= 2:
            parents.append(dict(id=n["id"], question=n.get("question"), parts=subs[:6]))
    seen = {}
    for n in nodes:
        if n.get("type") not in CHILD:
            continue
        key = (n.get("type"), json.dumps(sorted(_of_ids(n))), json.dumps(sorted(words[n["id"]])))
        if words[n["id"]] and key in seen:
            dups.append((seen[key], n["id"]))
        seen.setdefault(key, n["id"])
    return dict(n=len(nodes), flagged=len(likely_bundled(d)), parents=parents, dups=dups)


res = {}
for tag in args.tags.split(","):
    agg = dict(prompts=0, nodes=0, flagged=0, parents=0, dups=0, missing=0)
    per = {}
    for r in rows:
        t = r["tag"] if tag == "BASE" else tag
        p = os.path.join(plan_dir(r["set"], t), r["pid"] + ".json")
        if not os.path.exists(p) and tag in fb:
            p = os.path.join(plan_dir(r["set"], fb[tag]), r["pid"] + ".json")
        if not os.path.exists(p):
            agg["missing"] += 1
            continue
        m = measure(json.load(open(p, encoding="utf-8")))
        per[r["pid"]] = m
        agg["prompts"] += 1
        agg["nodes"] += m["n"]
        agg["flagged"] += m["flagged"]
        agg["parents"] += len(m["parents"])
        agg["dups"] += len(m["dups"])
    agg["flagged_rate"] = round(agg["flagged"] / max(1, agg["nodes"]), 4)
    agg["parents_per_prompt"] = round(agg["parents"] / max(1, agg["prompts"]), 3)
    res[tag] = dict(summary=agg, per=per)
    print(tag, json.dumps(agg))
if args.out:
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", args.out)
