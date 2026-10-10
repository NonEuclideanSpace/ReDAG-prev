"""Drafts only (executor, no evaluator): material for the label set and for the complexity scan before the
evaluator gate exists. Writes runs/<run>/img/<pid>/draft_XX.png and runs/<run>/drafts.jsonl."""
import argparse
import json
import os
import random
import time

from common import P0, ids_filter, load_set, plan_dir, shard

from redag.schema import Plan
from redag.views import save_png

ap = argparse.ArgumentParser()
ap.add_argument("--set", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--run", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--shard", default=None)
ap.add_argument("--ids-file", default=None, help="only prompts listed in this file (one id per line), e.g. data/tier_picks/<tier>.txt")
ap.add_argument("--picks", default=None, help="tier picks file (pid set tag split per line): draw these prompts across their sets in one process (--set/--tag unused)")
ap.add_argument("--executor", default="qi21", choices=["qi21", "mock"])
ap.add_argument("--qi-steps", type=int, default=40)
ap.add_argument("--skip-preflight", action="store_true", help="mock only")
args = ap.parse_args()

out = os.path.join(P0, "runs", args.run)
os.makedirs(out, exist_ok=True)
if args.picks:
    where = {}
    for line in open(args.picks, encoding="utf-8"):
        t = line.strip().split()
        if t and not t[0].startswith("#") and len(t) >= 3:
            where[t[0]] = (t[1], t[2])
    by_id = {}
    for s in sorted({v[0] for v in where.values()}):
        for r in load_set(s):
            if r["id"] in where and where[r["id"]][0] == s:
                by_id[r["id"]] = dict(r, _set=s, _tag=where[r["id"]][1])
    rows = shard([by_id[pid] for pid in where if pid in by_id][: args.limit], args.shard)
elif args.set and args.tag:
    rows = shard(ids_filter(load_set(args.set), args.ids_file)[: args.limit], args.shard)
else:
    raise SystemExit("give --set and --tag, or --picks")
if args.executor == "mock":
    from redag.mock import MockExecutor
else:
    if not args.skip_preflight:
        from redag.preflight import require_preflight

        require_preflight()
    from redag.executor_qi21 import QI21

    ex = QI21(steps=args.qi_steps, res=args.size)


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def claimed_elsewhere(d, max_age=3600):
    """A live claim by another process of this machine, younger than max_age seconds (a crashed run leaves a dead pid)."""
    p = os.path.join(d, "claim.json")
    if not os.path.exists(p):
        return False
    try:
        c = json.load(open(p, encoding="utf-8"))
    except Exception:
        return False
    return c.get("pid") != os.getpid() and time.time() - float(c.get("t", 0)) < max_age and _alive(c.get("pid"))


def claim(d):
    with open(os.path.join(d, "claim.json"), "w", encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(), t=time.time()), f)


def unclaim(d):
    try:
        os.remove(os.path.join(d, "claim.json"))
    except OSError:
        pass


t0 = time.time()
for i, r in enumerate(rows):
    pp = os.path.join(plan_dir(r["_set"], r["_tag"]) if args.picks else plan_dir(args.set, args.tag), r["id"] + ".json")
    if not os.path.exists(pp):
        continue
    plan = Plan.load(pp)
    d = os.path.join(out, "img", r["id"])
    if os.path.exists(os.path.join(d, "drafts.json")):
        continue
    os.makedirs(d, exist_ok=True)
    if claimed_elsewhere(d):  # another draw process on this machine is on this prompt (two queues can overlap)
        continue
    claim(d)
    if args.executor == "mock":
        from redag.schema import add_generated_nodes

        ex = MockExecutor(add_generated_nodes(Plan(plan.to_dict())), seed=args.seed)
    seed = random.Random(f"{args.seed}:{r['id']}").randrange(2 ** 31)
    t1 = time.time()
    ims = ex.draw(plan.prompt, args.n, seed, args.size)
    sec = time.time() - t1
    paths = [save_png(im, os.path.join(d, f"draft_{k:02d}.png")) for k, im in enumerate(ims)]
    rec = dict(pid=r["id"], prompt=plan.prompt, seed=seed, n=len(paths), drafts=paths, sec=round(sec, 1), size=args.size)
    with open(os.path.join(d, "drafts.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "drafts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    unclaim(d)
    print(i, r["id"], len(paths), "drafts", round(sec, 1), "s", "elapsed", round(time.time() - t0), flush=True)
print("DONE", flush=True)
