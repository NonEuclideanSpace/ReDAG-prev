"""Second, independent labelling of a label set by an API model of another family (component 0). Reads
labels/<name>/items.jsonl, writes labels/<name>/labels_<model>.jsonl in the same format as the human file.
Never the evaluator under test; never looks at any score. Defaults: OpenAI API, gpt-6.1-sol, key OPENAI_API_KEY from the
environment or P0/secrets.env. Example:
  python scripts/label_with_model.py --name seed_v4 --limit 10
"""
import argparse
import json
import os

from common import P0, add_vlm_args, make_vlm

from draftdag.labels import label_items_with_model, read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="labels/<name>/")
ap.add_argument("--kinds", default="cond,edit,diag")
ap.add_argument("--limit", type=int, default=10 ** 9)
ap.add_argument("--side", type=int, default=1024)
add_vlm_args(ap)
ap.set_defaults(vlm_url="https://api.openai.com/v1", vlm_model="gpt-6.1-sol", vlm_mode="graded", vlm_key_env="OPENAI_API_KEY", vlm_workers=4)
args = ap.parse_args()

vlm = make_vlm(args, require_gate=False)
d = os.path.join(P0, "labels", args.name)
items = [it for it in read_jsonl(os.path.join(d, "items.jsonl")) if it["kind"] in args.kinds.split(",")][: args.limit]
out = os.path.join(d, f"labels_{vlm.tag()}.jsonl")
done = {r["id"] for r in read_jsonl(out)}
todo = [it for it in items if it["id"] not in done]
print(f"{len(items)} items, {len(done)} already labelled, {len(todo)} to do", flush=True)
rows = read_jsonl(out)
B = 32
for i in range(0, len(todo), B):
    rows += label_items_with_model(todo[i:i + B], vlm, side=args.side)
    write_jsonl(out, rows)
    print(f"{min(i + B, len(todo))}/{len(todo)}", vlm.stats(), flush=True)
missing = sum(1 for r in rows if r.get("label") is None and r.get("fixed") is None)
print(json.dumps(dict(out=out, n=len(rows), unparsed=missing)))
