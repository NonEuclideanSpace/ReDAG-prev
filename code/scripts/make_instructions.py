"""Editing instructions for scripts/edit_bench.py. Two sources: templated rows from a drafts run (scripts/draw.py) and its
plans, no evaluator involved and the ops blind (an "add" on an entity that is already present yields a duplicate, which
the label set needs anyway); or the instructions the loop really wrote (--from-runs: every branch of every round of
runs/<run>/img/<pid>/trace.json, with its canvas). Rows: {pid, canvas, instruction, op, node, ntype, region, source}."""
import argparse
import json
import os
import random

from common import P0, plan_dir, run_dir

from redag.diagnose import KEEP_TAIL
from redag.schema import CHILD_TYPES, MULTI_TYPES, Plan

ap = argparse.ArgumentParser()
ap.add_argument("--drafts-run", default=None, help="scripts/draw.py run: templated rows")
ap.add_argument("--from-runs", nargs="*", default=[], help="loop runs: the instructions the evaluator wrote, one row per branch")
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True, help="jsonl path")
ap.add_argument("--per-draft", type=int, default=2)
ap.add_argument("--drafts-per-prompt", type=int, default=2)
ap.add_argument("--ops", default="add,fix,remove", help="subset of add,fix,remove")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

ops = set(args.ops.split(","))
rng = random.Random(args.seed)
pdir = plan_dir(args.set, args.tag)


def where(plan, box):
    cx = (box[0] + box[2]) / 2 / plan.W
    cy = (box[1] + box[3]) / 2 / plan.H
    h = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    v = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    if h == "center" and v == "middle":
        return "in the center of the image"
    if h == "center":
        return f"in the {v} part of the image"
    if v == "middle":
        return f"on the {h} side of the image"
    return f"in the {v} {h} of the image"


def desc(plan, v):
    return plan.nodes[v]["desc"].strip().rstrip(".")


def rows_for(plan, canvas):
    cands = []
    for v in plan.order:
        t = plan.typ(v)
        if t == "entity" and "add" in ops:
            box = plan.planned_box(v)
            cands.append(dict(node=v, ntype=t, op="add", region=box, instruction=f"Add {desc(plan, v)} {where(plan, box)}, at a realistic size." + KEEP_TAIL))
        elif (t in CHILD_TYPES or t in MULTI_TYPES) and "fix" in ops:
            box = plan.planned_box(v)
            parts = " and ".join(desc(plan, u) for u in plan.participants(v))
            cands.append(dict(node=v, ntype=t, op="fix", region=box, instruction=f"Change {parts} so that this becomes true: {desc(plan, v)}." + KEEP_TAIL))
        elif t == "absent" and "remove" in ops:
            cands.append(dict(node=v, ntype=t, op="remove", region=None, instruction=f"Remove any {desc(plan, v)} from the image and fill its place with the surrounding background." + KEEP_TAIL))
    rng.shuffle(cands)
    return [dict(pid=plan.id, canvas=canvas, source="template", **c) for c in cands[: args.per_draft]]


n = 0
with open(args.out, "w", encoding="utf-8") as f:
    if args.drafts_run:
        rd = run_dir(args.drafts_run)
        for line in open(os.path.join(rd, "drafts.jsonl"), encoding="utf-8"):
            rec = json.loads(line)
            pp = os.path.join(pdir, rec["pid"] + ".json")
            if not os.path.exists(pp):
                continue
            plan = Plan.load(pp)
            drafts = list(rec["drafts"])
            rng.shuffle(drafts)
            for canvas in drafts[: args.drafts_per_prompt]:
                for r in rows_for(plan, canvas):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
    for run in args.from_runs:
        rd = run_dir(run)
        root = os.path.join(rd, "img")
        for pid in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            tp = os.path.join(root, pid, "trace.json")
            if not os.path.exists(tp):
                continue
            tr = json.load(open(tp, encoding="utf-8"))
            for rdd in tr.get("rounds", []):
                for br in rdd.get("branches", []):
                    if not br.get("instruction"):
                        continue
                    row = dict(pid=pid.split("+")[0], canvas=rdd["canvas"], instruction=br["instruction"], op=br["op"], node=br["node"], ntype=br["ntype"], region=br.get("region"), source=f"loop:{os.path.basename(rd.rstrip('/'))}:r{rdd['round']}:{br['target']}", accepted=br.get("accepted"))
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
print(json.dumps(dict(rows=n, out=args.out)))
