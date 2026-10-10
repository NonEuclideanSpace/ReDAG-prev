"""Planner coverage measured against an independent decomposition: the official CoReBench checklists. For every official
question a matcher model (another series than the planner) names the one plan requirement that asks about the same fact,
or null. From the mapping: coverage (official questions some requirement asks about), and bundling (requirements that
cover two or more official questions, i.e. ask several separately checkable facts at once), by official tag and by
requirement type. The checklists stay out of the loop; this only grades the planner.
  python scripts/checklist_coverage.py --sets cb_ma,cb_mi,cb_mr --tag qwen3.7-plus-v5 --out ../gates/coverage_official_v5.json"""
import argparse
import collections
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from common import P0, plan_dir

from redag.conditions import build_conditions
from redag.planner import api_chat, parse_json
from redag.schema import Plan, add_generated_nodes

MATCH = """Scene description:
{prompt}

Official checklist questions (numbered Q1, Q2, ...). Each asks about one fact that the description requires:
{questions}

Requirements that a planner extracted from the same description (numbered R1, R2, ...), each a yes/no question:
{reqs}

For each official question, give the number of the one requirement that asks about the same fact, or null if no requirement asks about it. A requirement asks about a fact when answering the requirement "yes" would confirm that fact, even if the requirement also asks about other things or uses different words; if several requirements do, pick the one closest to the official question. An official question of the form "If X is present, ..." is about the part after the comma. An official question "Is there no X?" or "does it have no X?" is covered by a requirement that checks whether X appears. Reply with one JSON object and nothing else: {{"Q1": <R number or null>, "Q2": <R number or null>, ...}}"""

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="cb_ma,cb_mi,cb_mr")
ap.add_argument("--tag", required=True)
ap.add_argument("--model", default="deepseek-v4-pro")
ap.add_argument("--ids", default=None, help="comma list of prompt ids (trial run)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def requirements(plan):
    """Plan conditions as the evaluator asks them, without the two image-quality checks."""
    p = add_generated_nodes(Plan(plan.to_dict()))
    out = []
    for c in build_conditions(p):
        if c.kind == "quality":
            continue
        out.append(dict(cid=c.cid, ntype=c.ntype, question=c.question))
    return out


jobs = []
for s in args.sets.split(","):
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(P0, "data", "sets", s + ".jsonl"), encoding="utf-8")}
    checks = json.load(open(os.path.join(P0, "data", "checklists", s + ".json"), encoding="utf-8"))
    ids = sorted(rows)
    if args.ids:
        want = set(args.ids.split(","))
        ids = [i for i in ids if i in want]
    if args.limit:
        ids = ids[: args.limit]
    for pid in ids:
        pp = os.path.join(plan_dir(s, args.tag), pid + ".json")
        if not os.path.exists(pp):
            continue
        jobs.append((s, pid, rows[pid]["prompt"], checks[pid], requirements(Plan.load(pp))))
print("prompts:", len(jobs), flush=True)


def one(job):
    s, pid, prompt, official, reqs = job
    qs = "\n".join(f"Q{i+1}. {c['question']}" for i, c in enumerate(official))
    rs = "\n".join(f"R{j+1}. {r['question']}" for j, r in enumerate(reqs))
    text = MATCH.format(prompt=prompt, questions=qs, reqs=rs)
    m, err = None, None
    for attempt in range(2):
        try:
            out = api_chat([{"role": "user", "content": text}], model=args.model, max_tokens=1500, temperature=0.0, tag=f"cov_official:{pid}")
            d = parse_json(out)
            m = {}
            for i in range(len(official)):
                v = d.get(f"Q{i+1}")
                if isinstance(v, str):
                    v = v.strip().lstrip("R")
                    v = int(v) if v.isdigit() else None
                m[i] = (v - 1) if isinstance(v, int) and 1 <= v <= len(reqs) else None
            break
        except Exception as ex:
            err = str(ex)[:200]
    return dict(set=s, pid=pid, n_official=len(official), n_reqs=len(reqs), mapping=None if m is None else {str(k): v for k, v in m.items()},
                official=[dict(q=c["question"], tag=(c.get("tags") or ["?"])[0]) for c in official], reqs=reqs, error=err)


with ThreadPoolExecutor(max_workers=args.workers) as ex:
    res = list(ex.map(one, jobs))

# metrics
tot = collections.Counter()
by_tag = collections.defaultdict(lambda: [0, 0])
by_set = collections.defaultdict(lambda: [0, 0, 0])
shared_by_rtype = collections.Counter()
covered_by_rtype = collections.Counter()
bundle_hist = collections.defaultdict(collections.Counter)
uncovered = []
failed = 0
for r in res:
    if r["mapping"] is None:
        failed += 1
        continue
    cover_n = collections.Counter(v for v in r["mapping"].values() if v is not None)
    for j, req in enumerate(r["reqs"]):
        bundle_hist[req["ntype"]][min(cover_n.get(j, 0), 5)] += 1
    for i, o in enumerate(r["official"]):
        v = r["mapping"][str(i)]
        by_tag[o["tag"]][0] += 1
        by_set[r["set"]][0] += 1
        if v is None:
            uncovered.append((r["pid"], o["tag"], o["q"]))
            continue
        by_tag[o["tag"]][1] += 1
        by_set[r["set"]][1] += 1
        rt = r["reqs"][v]["ntype"]
        covered_by_rtype[rt] += 1
        if cover_n[v] >= 2:
            shared_by_rtype[rt] += 1
            by_set[r["set"]][2] += 1
n_off = sum(v[0] for v in by_set.values())
n_cov = sum(v[1] for v in by_set.values())
n_sh = sum(v[2] for v in by_set.values())
summary = dict(tag=args.tag, model=args.model, prompts=len(res), failed=failed, official=n_off, covered=n_cov, coverage=round(n_cov / max(1, n_off), 4),
               shared=n_sh, shared_rate=round(n_sh / max(1, n_cov), 4),
               by_set={k: dict(official=v[0], coverage=round(v[1] / max(1, v[0]), 4), shared_rate=round(v[2] / max(1, v[1]), 4)) for k, v in sorted(by_set.items())},
               by_tag={k: dict(official=v[0], coverage=round(v[1] / max(1, v[0]), 4)) for k, v in sorted(by_tag.items())},
               covered_by_req_type=dict(covered_by_rtype), shared_by_req_type=dict(shared_by_rtype),
               bundle_size_by_req_type={k: dict(sorted(v.items())) for k, v in sorted(bundle_hist.items())})
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(dict(summary=summary, uncovered=uncovered, rows=res), open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("->", args.out)
