import argparse
import json
import os
import random

from common import P0, plan_dir, run_dir

from redag.labels import attach_questions, items_from_run, make_sheet, read_jsonl, write_jsonl
from redag.schema import Plan, add_generated_nodes

ap = argparse.ArgumentParser(description="Build a label set (items + HTML sheet) from one or more runs.")
ap.add_argument("--runs", nargs="*", default=[], help="loop runs (traces)")
ap.add_argument("--drafts-runs", nargs="*", default=[], help="scripts/draw.py runs: every condition of every draft becomes a cond item")
ap.add_argument("--edits-runs", nargs="*", default=[], help="scripts/edit_bench.py runs: edits.jsonl rows become edit items")
ap.add_argument("--set", default=None, help="one set for every item (or give --picks)")
ap.add_argument("--tag", default=None)
ap.add_argument("--picks", default=None, help="tier picks file (pid set tag split per line): the plan of each pid is read from its own set and tag, e.g. data/tier_picks/le15.txt for a pool that spans several sets")
ap.add_argument("--name", required=True, help="labels/<name>/")
ap.add_argument("--kinds", default="cond,edit,diag")
ap.add_argument("--max-cond", type=int, default=1200)
ap.add_argument("--max-edit", type=int, default=400)
ap.add_argument("--max-diag", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fail-boost", type=float, default=3.0, help="oversample conditions the system scored as failing")
args = ap.parse_args()

kinds = tuple(args.kinds.split(","))
items = []
for r in args.runs:
    rd = run_dir(r)
    items += items_from_run(rd, os.path.basename(rd.rstrip("/")), kinds)
where = {}
if args.picks:
    for line in open(args.picks, encoding="utf-8"):
        t = line.strip().split()
        if t and not t[0].startswith("#") and len(t) >= 3:
            where[t[0]] = (t[1], t[2])
elif not (args.set and args.tag):
    raise SystemExit("give --set and --tag, or --picks")
plans = {}


def plan_of(pid):
    pid = pid.split("+")[0]
    if pid not in plans:
        if args.picks:
            if pid not in where:
                return None
            pdir = plan_dir(*where[pid])
        else:
            pdir = plan_dir(args.set, args.tag)
        p = os.path.join(pdir, pid + ".json")
        if os.path.exists(p):
            plans[pid] = add_generated_nodes(Plan.load(p))
    return plans.get(pid)


from redag.conditions import build_conditions  # noqa: E402

for r in args.drafts_runs:
    rd = run_dir(r)
    name = os.path.basename(rd.rstrip("/"))
    for line in open(os.path.join(rd, "drafts.jsonl"), encoding="utf-8"):
        rec = json.loads(line)
        plan = plan_of(rec["pid"])
        if plan is None:
            continue
        for di, img in enumerate(rec["drafts"]):
            img = os.path.normpath(img if os.path.isabs(img) else os.path.join(rd, img))
            for c in build_conditions(plan):
                if c.kind == "quality":
                    continue
                items.append(dict(id=f"{name}:{rec['pid']}:d{di}:{c.cid}", kind="cond", image=img, cid=c.cid, ntype=c.ntype, inv=bool(c.inv), question=c.question, region=plan.planned_box(c.node) if c.kind == "node" else None, image_size=[plan.W, plan.H], pid=rec["pid"], run=name, status=None))
for r in args.edits_runs:
    rd = run_dir(r)
    for line in open(os.path.join(rd, "edits.jsonl"), encoding="utf-8"):
        items.append(json.loads(line))
for it in items:
    it["pid"] = it["pid"].split("+")[0]
    plan_of(it["pid"])
attach_questions(items, plans)
rng = random.Random(args.seed)


def pick(kind, n):
    xs = [it for it in items if it["kind"] == kind]
    if kind == "cond":
        w = [args.fail_boost if it.get("status") == "fail" else 1.0 for it in xs]
        out, seen = [], set()
        while len(out) < min(n, len(xs)):
            i = rng.choices(range(len(xs)), weights=w, k=1)[0]
            if i not in seen:
                seen.add(i)
                out.append(xs[i])
        return out
    rng.shuffle(xs)
    return xs[:n]


chosen = pick("cond", args.max_cond) + pick("edit", args.max_edit) + pick("diag", args.max_diag)
out_dir = os.path.join(P0, "labels", args.name)
os.makedirs(out_dir, exist_ok=True)
write_jsonl(os.path.join(out_dir, "items.jsonl"), chosen)
make_sheet(chosen, os.path.join(out_dir, "sheet.html"), title=args.name)
print(len(chosen), "items ->", out_dir)
