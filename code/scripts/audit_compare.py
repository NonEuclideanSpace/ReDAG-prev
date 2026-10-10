"""Planner audit (planner.audit_plan: questions beyond the description, questions that ask more than one thing, stated
facts no requirement covers) on the same prompts under several plan tags, with an auditing model of another series than
the planner. For before/after comparisons of a planner pass on a pilot; the per-set gate stays scripts/audit_plans.py.
  python scripts/audit_compare.py --tags BASE,qwen3.7-plus-v8pilot2 --ids c_ma_001,lb_0374 --model deepseek-v4-pro --out ../gates/audit_v8pilot2.json
BASE = the prompt's tag in data/tiers.json."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from common import P0, plan_dir

from redag.planner import audit_plan
from redag.schema import Plan

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--tiers", default=os.path.join(P0, "data", "tiers.json"))
ap.add_argument("--model", default="deepseek-v4-pro")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()

want = set(args.ids.split(","))
rows = [r for v in json.load(open(args.tiers, encoding="utf-8"))["tiers"].values() for r in v if r["pid"] in want]
jobs = []
for tag in args.tags.split(","):
    for r in rows:
        t = r["tag"] if tag == "BASE" else tag
        p = os.path.join(plan_dir(r["set"], t), r["pid"] + ".json")
        if os.path.exists(p):
            jobs.append((tag, r["pid"], p))
print("jobs:", len(jobs), flush=True)


def one(job):
    tag, pid, p = job
    plan = Plan.load(p)
    try:
        a = audit_plan(plan, model=args.model)
    except Exception as ex:
        return dict(tag=tag, pid=pid, error=str(ex)[:200])
    a["n_req"] = sum(1 for v in plan.order if plan.typ(v) != "background")
    return dict(tag=tag, pid=pid, **a)


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    res = list(ex.map(one, jobs))
summary = {}
for tag in args.tags.split(","):
    rs = [r for r in res if r["tag"] == tag and not r.get("error")]
    q = sum(r["n_questions"] for r in rs)
    ok = [r for r in rs if not r.get("match_suspect")]
    req = sum(r["n_req"] for r in ok)
    miss = sum(len(r["missing"]) for r in ok)
    summary[tag] = dict(prompts=len(rs), errors=sum(1 for r in res if r["tag"] == tag and r.get("error")), questions=q,
                        beyond_rate=round(sum(r["n_beyond"] for r in rs) / max(1, q), 4), multi_rate=round(sum(r.get("n_multi", 0) for r in rs) / max(1, q), 4),
                        missing_rate=round(miss / max(1, req + miss), 4), match_suspect=[r["pid"] for r in rs if r.get("match_suspect")])
    print(tag, json.dumps(summary[tag]))
json.dump(dict(summary=summary, model=args.model, rows=res), open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", args.out)
