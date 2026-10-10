"""Progress of a draft pool against the tier picks: prompts done per tier and set, images, seconds per image.
  python scripts/pool_status.py --run pool_s0 [--drafts <path to drafts.jsonl>]"""
import argparse
import json
import os

from common import P0

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="pool_s0")
ap.add_argument("--drafts", default=None)
ap.add_argument("--picks", default=os.path.join(P0, "data", "tier_picks"))
args = ap.parse_args()
path = args.drafts or os.path.join(P0, "runs", args.run, "drafts.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
done = {r["pid"]: r for r in rows}
summ = json.load(open(os.path.join(args.picks, "summary.json"), encoding="utf-8"))["tiers"]
print(f"{len(rows)} prompts, {sum(r['n'] for r in rows)} drafts, mean {sum(r['sec'] for r in rows) / max(1, len(rows)) / 16:.1f} s per draft")
for tier, s in summ.items():
    pids = [l.split()[0] for l in open(os.path.join(args.picks, s["file"]), encoding="utf-8") if l.strip() and not l.startswith("#")]
    d = [p for p in pids if p in done]
    by = {}
    for p in d:
        by[p.rsplit("_", 1)[0]] = by.get(p.rsplit("_", 1)[0], 0) + 1
    print(f"  tier {tier:6}: {len(d):3}/{len(pids)} prompts  {by}")
