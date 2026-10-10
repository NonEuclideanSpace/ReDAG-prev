import argparse
import json
import os
import random

from common import P0, load_set, plan_dir

from redag.paths import gates_dir
from redag.planner import DEFAULT_MODEL, audit_plan
from redag.schema import Plan

ap = argparse.ArgumentParser(description="Planner gate: questions beyond the prompt, and uncovered requirements, on a sample of plans.")
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--model", default=DEFAULT_MODEL, help="auditing model; use a different model than the planner when possible")
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--beyond-max", type=float, default=0.05)
ap.add_argument("--missing-max", type=float, default=0.05)
ap.add_argument("--multi-max", type=float, default=0.05, help="questions that ask about more than one thing")
ap.add_argument("--workers", type=int, default=8, help="plans audited in parallel")
args = ap.parse_args()

pdir = plan_dir(args.set, args.tag)
rows = [r for r in load_set(args.set) if os.path.exists(os.path.join(pdir, r["id"] + ".json"))]
rng = random.Random(args.seed)
rows = rng.sample(rows, min(args.n, len(rows)))
tot_q = tot_b = tot_req = tot_miss = tot_multi = 0
per = []


def one(r):
    plan = Plan.load(os.path.join(pdir, r["id"] + ".json"))
    a = audit_plan(plan, model=args.model)
    a["n_req"] = sum(1 for v in plan.order if plan.typ(v) != "background")
    return r["id"], a


from concurrent.futures import ThreadPoolExecutor  # noqa: E402

with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
    results = list(ex.map(one, rows))
suspect = []
for pid, a in results:
    tot_q += a["n_questions"]
    tot_b += a["n_beyond"]
    tot_multi += a.get("n_multi", 0)
    n_req = a.pop("n_req")
    if a.get("match_suspect"):
        suspect.append(pid)
    else:
        tot_req += n_req
        tot_miss += len(a["missing"])
    per.append(dict(pid=pid, **a))
    print(pid, "beyond", a["n_beyond"], "multi", a.get("n_multi", 0), "/", a["n_questions"], "missing", a["missing"][:3], flush=True)
beyond_rate = tot_b / max(1, tot_q)
missing_rate = tot_miss / max(1, tot_req + tot_miss)
multi_rate = tot_multi / max(1, tot_q)
gate = dict(set=args.set, tag=args.tag, auditor=args.model, n_plans=len(per), n_match_suspect=len(suspect), match_suspect=suspect, beyond_rate=beyond_rate, missing_rate=missing_rate, multi_rate=multi_rate, pass_beyond=beyond_rate <= args.beyond_max, pass_missing=missing_rate <= args.missing_max, pass_multi=multi_rate <= args.multi_max, per=per)
gate["pass"] = bool(per) and gate["pass_beyond"] and gate["pass_missing"] and gate["pass_multi"]
out = os.path.join(gates_dir(), f"planner_{args.set}_{args.tag}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(gate, f, ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in gate.items() if k != "per"}, ensure_ascii=False))
