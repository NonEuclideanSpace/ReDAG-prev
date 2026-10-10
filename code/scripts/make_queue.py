"""Write a jobq.py queue file for one machine from the tier picks. N shards are spread over all machines; this
machine gets the shard indices in --mine. Every line is one resumable task (one set, one shard).
  python scripts/make_queue.py --phase A --shards 8 --mine 0,1,2 --seed 0 --out ../queue/$(hostname).txt
  python scripts/make_queue.py --phase B --shards 8 --mine 3,4,5 --seed 0 --out ../queue/$(hostname).txt
Phase A: draft pool (draw.py, 16 drafts) for every set in the picks, then the edit benchmark on the sets of
--edit-sets. Phase B: scoring/scan runs (run_loop --mode draw --drafts-from) and repair runs on the dev picks."""
import argparse
import os

from common import P0

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["A", "B"])
ap.add_argument("--shards", type=int, required=True, help="total generation GPUs over all machines")
ap.add_argument("--mine", required=True, help="comma list of shard indices for this machine")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--picks", default=os.path.join(P0, "data", "tier_picks", "all.txt"))
ap.add_argument("--n", type=int, default=16, help="drafts per prompt")
ap.add_argument("--edit-sets", default="t20,t10,cb_mi,lb_t2i", help="phase A: sets whose drafts get templated edit instructions")
ap.add_argument("--out", required=True)
args = ap.parse_args()

tags = {}
for line in open(args.picks, encoding="utf-8"):
    t = line.strip().split()
    if t and not t[0].startswith("#") and len(t) >= 3:
        tags[t[1]] = t[2]
mine = [int(x) for x in args.mine.split(",")]
rel = os.path.relpath(args.picks, os.path.join(P0, "code"))
lines = [f"# phase {args.phase}, seed {args.seed}, shards {mine} of {args.shards}, picks {rel}"]
if args.phase == "A":
    for s in sorted(tags):
        for i in mine:
            lines.append(f"python scripts/draw.py --set {s} --tag {tags[s]} --run pool_s{args.seed} --n {args.n} --seed {args.seed} --ids-file {rel} --shard {i}/{args.shards}")
    lines.append("BARRIER")  # instructions are written from this machine's own drafts, so wait for its pool shards
    for s in args.edit_sets.split(","):
        if s in tags:
            lines.append(f"python scripts/make_instructions.py --drafts-run pool_s{args.seed} --set {s} --tag {tags[s]} --out ../runs/pool_s{args.seed}/instr_{s}.jsonl --seed {args.seed}")
    lines.append("BARRIER")
    for s in args.edit_sets.split(","):
        if s in tags:
            for k in range(len(mine)):  # local split: the instruction file only has this machine's prompts
                lines.append(f"python scripts/edit_bench.py --instructions ../runs/pool_s{args.seed}/instr_{s}.jsonl --run edits_s{args.seed} --G 4 --seed {args.seed} --shard {k}/{len(mine)}")
else:
    for s in sorted(tags):
        for i in mine:
            lines.append(f"python scripts/run_loop.py --set {s} --tag {tags[s]} --run scan_s{args.seed} --mode draw --drafts-from pool_s{args.seed} --seed {args.seed} --ids-file {rel} --shard {i}/{args.shards}")
    for s in sorted(tags):
        for i in mine:
            lines.append(f"python scripts/run_loop.py --set {s} --tag {tags[s]} --run repair_dev_s{args.seed} --mode repair --seed {args.seed} --ids-file {rel} --shard {i}/{args.shards}")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines) - 1, "tasks ->", args.out)
