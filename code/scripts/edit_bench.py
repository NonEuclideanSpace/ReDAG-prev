"""Editor benchmark (component 5) and acceptance material (component 6): candidates from a file of instructions,
region-committed, no evaluator involved. Input jsonl rows: {pid, canvas, instruction, op, node, ntype, region}.
Output runs/<run>/img/<pid>/<node>_raw{i}.png, _c{i}.png and runs/<run>/edits.jsonl (label-set edit items)."""
import argparse
import json
import os
import random
import time

from common import P0, shard

from redag.commit import region_commit
from redag.views import load_image, save_png, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--instructions", required=True, help="jsonl with pid, canvas, instruction, op, node, ntype, region")
ap.add_argument("--run", required=True)
ap.add_argument("--G", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", default=None)
ap.add_argument("--executor", default="qi21", choices=["qi21", "mock"])
ap.add_argument("--plan", default=None, help="plan json (mock executor only)")
ap.add_argument("--qi-steps", type=int, default=40)
ap.add_argument("--qi-res", type=int, default=1024)
ap.add_argument("--skip-preflight", action="store_true", help="mock only")
args = ap.parse_args()

rows = shard([json.loads(l) for l in open(args.instructions, encoding="utf-8") if l.strip()], args.shard)
out = os.path.join(P0, "runs", args.run)
os.makedirs(out, exist_ok=True)
if args.executor == "mock":
    from redag.mock import MockExecutor
    from redag.schema import Plan, add_generated_nodes

    ex = MockExecutor(add_generated_nodes(Plan.load(args.plan)), seed=args.seed)
else:
    if not args.skip_preflight:
        from redag.preflight import require_preflight

        require_preflight()
    from redag.executor_qi21 import QI21

    ex = QI21(steps=args.qi_steps, res=args.qi_res)
rng = random.Random(args.seed)
t0 = time.time()
for i, r in enumerate(rows):
    d = os.path.join(out, "img", r["pid"])
    os.makedirs(d, exist_ok=True)
    key = f"{r['node']}_{i}"
    if os.path.exists(os.path.join(d, key + ".json")):
        continue
    canvas = to_rgb(load_image(r["canvas"]))
    seed = rng.randrange(2 ** 31)
    t1 = time.time()
    cands = ex.edit(canvas, r["instruction"], args.G, seed)
    sec = time.time() - t1
    items = []
    for k, im in enumerate(cands):
        raw = save_png(im, os.path.join(d, f"{key}_raw{k}.png"))
        comp, info = region_commit(canvas, im, scope=[r["region"]] if r.get("region") else [])
        comp = ex.commit_hook(canvas, im, comp, info)
        cp = save_png(comp, os.path.join(d, f"{key}_c{k}.png"))
        it = dict(id=f"{args.run}:{r['pid']}:{key}:c{k}", kind="edit", before=r["canvas"], after=cp, raw=raw, op=r["op"], node=r["node"], ntype=r.get("ntype"), instruction=r["instruction"], region=r.get("region"), pid=r["pid"], run=args.run, share=info["share"], commit_boxes=info["boxes"], seed=seed, sec=round(sec / args.G, 1))
        items.append(it)
        with open(os.path.join(out, "edits.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(os.path.join(d, key + ".json"), "w", encoding="utf-8") as f:
        json.dump(dict(row=r, items=items), f, ensure_ascii=False, indent=1)
    print(i, r["pid"], r["op"], r["node"], f"{sec:.1f}s", "elapsed", round(time.time() - t0), flush=True)
print("DONE", flush=True)
