"""Check a dedupe tag against its source tag, without any model: every plan validates, an unchanged plan is byte-identical
to its source, a changed plan keeps every content word its source asked (frame positions, sound/motion cues, hedges
excepted), keeps every surviving node's fields (only of/type/box may change by a merge), has unique ids and resolving
of references. Prints the totals and every violation.
  python scripts/check_dedupe.py --src-tag qwen3.7-plus-v84 --tag qwen3.7-plus-v85 [--tiers ../data/tiers_v84.json]"""
import argparse
import collections
import json
import os

from common import P0, plan_dir

from redag.planner import LOST_FREE, MOD_FREE, NONVISUAL_WORD, SPLIT_FREE, _content, _frame_sub, _of_ids, planner_errors
from redag.schema import Plan

ap = argparse.ArgumentParser()
ap.add_argument("--src-tag", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--tiers", default=os.path.join(P0, "data", "tiers_v84.json"))
ap.add_argument("--max-nodes", type=int, default=128)
args = ap.parse_args()


def words(d):
    w = set()
    for n in d["nodes"]:
        t, _ = _frame_sub(n.get("question", ""))
        w |= _content(t, SPLIT_FREE)
    return w


rows = [r for v in json.load(open(args.tiers, encoding="utf-8"))["tiers"].values() for r in v]
tot = collections.Counter()
bad = []
for r in rows:
    a = json.load(open(os.path.join(plan_dir(r["set"], args.src_tag), r["pid"] + ".json"), encoding="utf-8"))
    p = os.path.join(plan_dir(r["set"], args.tag), r["pid"] + ".json")
    if not os.path.exists(p):
        tot["missing"] += 1
        continue
    b = json.load(open(p, encoding="utf-8"))
    tot["plans"] += 1
    tot["nodes_src"] += len(a["nodes"])
    tot["nodes"] += len(b["nodes"])
    errs, _w = Plan(b).validate(min_side=48, max_nodes=args.max_nodes)
    if errs:
        bad.append((r["pid"], "validate", errs[:3]))
    new_pe = sorted(set(planner_errors(b)) - set(planner_errors(a)))
    if new_pe:
        bad.append((r["pid"], "planner_errors", new_pe[:2]))
    if not (b.get("meta") or {}).get("dedupe"):
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            bad.append((r["pid"], "unchanged plan differs", None))
        continue
    tot["changed"] += 1
    lost = sorted({w for w in words(a) - words(b) if not NONVISUAL_WORD.fullmatch(w)} - LOST_FREE - MOD_FREE)
    if lost:
        bad.append((r["pid"], "lost words", lost))
    bb = {n["id"]: n for n in b["nodes"]}
    if len(bb) != len(b["nodes"]):
        bad.append((r["pid"], "duplicate ids", None))
    for n in a["nodes"]:
        m = bb.get(n["id"])
        if m is None:
            continue
        for k in n:
            if k not in ("of", "type", "box") and n[k] != m.get(k):
                bad.append((r["pid"], "field changed", (n["id"], k)))
        if m["type"] != n["type"]:
            tot[f"retyped {n['type']}->{m['type']}"] += 1
    for n in b["nodes"]:
        for u in _of_ids(n):
            if u not in bb:
                bad.append((r["pid"], "dangling of", (n["id"], u)))
print(json.dumps(tot))
print("violations:", len(bad))
for x in bad[:50]:
    print(" ", x)
