"""Build the reference label file for the evaluator gate from several label files in priority order: the first file
that labels an item wins (human sheets first, then the model labels), and an item labelled by two model files that
disagree is left out unless --keep-disagreements. Reports what came from where.
  python scripts/merge_ref.py --name seed_v4 --human labels_human.jsonl labels_human_count.jsonl labels_human_dispute.jsonl \
      --models labels_gpt-6.1-sol.jsonl labels_claude.jsonl --out labels_ref.jsonl"""
import argparse
import collections
import os

from common import P0

from redag.labels import read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="labels/<name>/")
ap.add_argument("--human", nargs="*", default=[], help="human label files, highest priority first")
ap.add_argument("--models", nargs="*", default=[], help="model label files; used only where all of them agree")
ap.add_argument("--keep-disagreements", action="store_true", help="where the model files disagree keep the first file's label instead of dropping the item")
ap.add_argument("--out", default="labels_ref.jsonl")
args = ap.parse_args()

d = os.path.join(P0, "labels", args.name)
items = read_jsonl(os.path.join(d, "items.jsonl"))
ref, source = {}, {}
for fn in args.human:
    p = fn if os.path.isabs(fn) else os.path.join(d, fn)
    for r in read_jsonl(p):
        if r.get("label") in ("y", "u", "n") and r["id"] not in ref:
            ref[r["id"]] = dict(id=r["id"], label=r["label"], note=r.get("note"))
            source[r["id"]] = os.path.basename(fn)
model_labels = []
for fn in args.models:
    p = fn if os.path.isabs(fn) else os.path.join(d, fn)
    model_labels.append({r["id"]: r for r in read_jsonl(p) if r.get("label") in ("y", "u", "n")})
dropped = []
for it in items:
    i = it["id"]
    if i in ref:
        continue
    labs = [m[i]["label"] for m in model_labels if i in m]
    if not labs:
        continue
    if len(set(labs)) == 1 or args.keep_disagreements:
        ref[i] = dict(id=i, label=labs[0], note=None)
        source[i] = "models-agree" if len(set(labs)) == 1 else "models-first"
    else:
        dropped.append(i)
rows = [ref[it["id"]] for it in items if it["id"] in ref]
write_jsonl(os.path.join(d, args.out), rows)
by_src = collections.Counter(source.values())
by_lab = collections.Counter(r["label"] for r in rows)
print(f"{len(rows)} of {len(items)} items -> {os.path.join(d, args.out)}; labels {dict(by_lab)}; sources {dict(by_src)}; dropped (models disagree, no human) {len(dropped)}")
