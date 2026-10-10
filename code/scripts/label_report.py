"""Agreement between two label files of one label set, plus the label mix; the component-0 gate line is the
human-vs-model agreement on the audited subset. Example:
  python scripts/label_report.py --name t20_seed --a labels_human.jsonl --b labels_claude.jsonl"""
import argparse
import json
import os
from collections import Counter

from common import P0

from redag.labels import agreement, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
args = ap.parse_args()

d = os.path.join(P0, "labels", args.name)
items = {it["id"]: it for it in read_jsonl(os.path.join(d, "items.jsonl"))}
la = read_jsonl(os.path.join(d, args.a))
lb = read_jsonl(os.path.join(d, args.b))
out = {}
for kind in ("cond", "edit", "diag"):
    ids = {i for i, it in items.items() if it["kind"] == kind}
    a = [r for r in la if r["id"] in ids]
    b = [r for r in lb if r["id"] in ids]
    if not a or not b:
        continue
    if kind == "diag":
        da, db = {r["id"]: r.get("label") for r in a}, {r["id"]: r.get("label") for r in b}
        common = [i for i in da if i in db]
        out[kind] = dict(n=len(common), exact=sum(1 for i in common if da[i] == db[i]) / max(1, len(common)))
    else:
        out[kind] = agreement(a, b, kind)
    out[kind]["mix_a"] = dict(Counter(r.get("label") if kind != "edit" else f"{r.get('fixed')}{r.get('damaged')}{r.get('duplicate')}" for r in a))
    out[kind]["mix_b"] = dict(Counter(r.get("label") if kind != "edit" else f"{r.get('fixed')}{r.get('damaged')}{r.get('duplicate')}" for r in b))
    if kind == "cond":
        by_type = {}
        da, db = {r["id"]: r.get("label") for r in a}, {r["id"]: r.get("label") for r in b}
        for i in da:
            if i in db:
                t = items[i].get("ntype") or "?"
                by_type.setdefault(t, [0, 0])
                by_type[t][1] += 1
                by_type[t][0] += int(da[i] == db[i])
        out[kind]["by_type"] = {t: dict(n=v[1], exact=v[0] / v[1]) for t, v in by_type.items()}
print(json.dumps(out, ensure_ascii=False, indent=1))
with open(os.path.join(d, f"agreement_{os.path.splitext(args.a)[0]}_vs_{os.path.splitext(args.b)[0]}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
