"""A labelling sheet for a random subset of one label set (the human audit: 100 node questions plus 30 edit pairs).
Writes labels/<name>/<out>; the page exports labels_human.jsonl. Example:
  python scripts/make_sheet.py --name seed_v4 --n-cond 100 --n-edit 30 --out sheet_human.html"""
import argparse
import os
import random

from common import P0

from redag.labels import make_sheet, read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--n-cond", type=int, default=100)
ap.add_argument("--n-edit", type=int, default=30)
ap.add_argument("--n-diag", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="sheet_human.html")
ap.add_argument("--title", default="人工抽查")
ap.add_argument("--embed", action="store_true", help="inline the images so the page can be sent on its own")
ap.add_argument("--side", type=int, default=768)
ap.add_argument("--ids", default=None, help="file of item ids (one per line): make the sheet for exactly these instead of a random subset")
ap.add_argument("--exclude-ids", default=None, help="file of item ids to leave out (e.g. the ids of an earlier sheet)")
args = ap.parse_args()

d = os.path.join(P0, "labels", args.name)
items = read_jsonl(os.path.join(d, "items.jsonl"))
rng = random.Random(args.seed)


def read_ids(path):
    return {l.strip().split()[0] for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")}


if args.exclude_ids:
    drop = read_ids(args.exclude_ids)
    items = [it for it in items if it["id"] not in drop]
pick = []
if args.ids:
    keep = read_ids(args.ids)
    pick = [it for it in items if it["id"] in keep]
else:
    for kind, n in (("cond", args.n_cond), ("edit", args.n_edit), ("diag", args.n_diag)):
        xs = [it for it in items if it["kind"] == kind]
        rng.shuffle(xs)
        pick += xs[:n]
out = os.path.join(d, args.out)
make_sheet(pick, out, title=args.title, embed=args.embed, side=args.side)
write_jsonl(os.path.join(d, os.path.splitext(args.out)[0] + "_items.jsonl"), pick)
print(len(pick), "items ->", out)
