"""Pick a fixed subset of prompts per complexity tier from data/tiers.json for the GPU runs (draft pool, scan,
system comparison), keeping each prompt's split. Writes data/tier_picks/<tier>.txt (one "pid set tag" per line)
and data/tier_picks/summary.json. The draw/loop scripts take --ids-file with these files.
  python scripts/pick_tiers.py --per-tier 120 --seed 0
  python scripts/pick_tiers.py --per-tier 100 --test-frac 1.0 --min-test 100 --out data/tier_picks_test   # reported runs: test only"""
import argparse
import json
import os
import random

from common import P0

ap = argparse.ArgumentParser()
ap.add_argument("--tiers", default=os.path.join(P0, "data", "tiers.json"))
ap.add_argument("--per-tier", type=int, default=120)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--test-frac", type=float, default=1 / 3, help="share of each pick taken from the test split (1.0 = test only, for the reported comparison)")
ap.add_argument("--min-test", type=int, default=0, help="require at least this many test prompts per tier (0 = no check)")
ap.add_argument("--out", default=os.path.join(P0, "data", "tier_picks"))
args = ap.parse_args()

t = json.load(open(args.tiers, encoding="utf-8"))["tiers"]
os.makedirs(args.out, exist_ok=True)
summary = {}
short = []
for tier, rows in t.items():
    rng = random.Random(f"{args.seed}:{tier}")
    rows = sorted(rows, key=lambda r: (r["set"], r["pid"]))
    rng.shuffle(rows)
    test = [r for r in rows if r.get("split") == "test"]
    dev = [r for r in rows if r.get("split") != "test"]
    n_test = min(len(test), max(int(round(args.per_tier * args.test_frac)), args.per_tier - len(dev), args.min_test))
    pick = test[:n_test] + dev[: max(0, args.per_tier - n_test)]
    pick = sorted(pick, key=lambda r: (r["set"], r["pid"]))
    name = tier.replace("<=", "le").replace(">", "gt")
    with open(os.path.join(args.out, name + ".txt"), "w", encoding="utf-8") as f:
        f.write(f"# tier {tier}: {len(pick)} prompts, test {n_test}, dev {len(pick) - n_test}; seed {args.seed}\n")
        for r in pick:
            f.write(f"{r['pid']} {r['set']} {r['tag']} {r.get('split')}\n")
    by_set = {}
    for r in pick:
        by_set[r["set"]] = by_set.get(r["set"], 0) + 1
    summary[tier] = dict(file=name + ".txt", n=len(pick), test=n_test, dev=len(pick) - n_test, available_test=len(test), by_set=by_set)
    if args.min_test and n_test < args.min_test:
        short.append((tier, n_test))
json.dump(dict(per_tier=args.per_tier, seed=args.seed, tiers=summary), open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False))
if short:
    raise SystemExit(f"tiers with too few test prompts: {short}; extend the small synthetic sets (make_sets.py synthetic) and re-plan")
