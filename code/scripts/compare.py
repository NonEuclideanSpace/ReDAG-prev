import argparse
import json
import os

from common import P0

from draftdag.report import compare, contact_sheet, run_metrics

ap = argparse.ArgumentParser(description="Paired comparison of two runs on their common prompts, plus contact sheets for eyeballing.")
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--sheet", type=int, default=16, help="how many final images per run to put on a contact sheet")
args = ap.parse_args()

ra = args.a if os.path.isabs(args.a) else os.path.join(P0, "runs", args.a)
rb = args.b if os.path.isabs(args.b) else os.path.join(P0, "runs", args.b)
res = compare(ra, rb)
print(json.dumps(res, ensure_ascii=False, indent=1))
for name, rd in (("a", ra), ("b", rb)):
    m = run_metrics(rd)
    pids = sorted(m)[: args.sheet]
    out = os.path.join(rd, "sheet_finals.jpg")
    contact_sheet([m[p]["final"] for p in pids], out, labels=[f"{p} {m[p]['node_pass']:.2f}" if m[p]["node_pass"] is not None else p for p in pids])
    print(name, "sheet:", out)
with open(os.path.join(ra, f"compare_vs_{os.path.basename(rb.rstrip('/'))}.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
