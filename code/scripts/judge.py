import argparse
import json
import os

from common import P0, add_vlm_args, load_set, make_vlm, plan_dir

from redag.report import judge_run
from redag.schema import Plan, add_generated_nodes
from redag.verify import VerifyConfig

ap = argparse.ArgumentParser(description="Score a run's final images with an independent judge (different model family); never used inside the loop.")
ap.add_argument("--run", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--which", default="final", choices=["final", "draft0"])
add_vlm_args(ap)
args = ap.parse_args()

vlm = make_vlm(args, require_gate=False)
if vlm.coord_mode is None:
    vlm.calibrate_coords()
pdir = plan_dir(args.set, args.tag)
plans = {}
for r in load_set(args.set):
    p = os.path.join(pdir, r["id"] + ".json")
    if os.path.exists(p):
        plans[r["id"]] = add_generated_nodes(Plan.load(p))
rd = args.run if os.path.isabs(args.run) else os.path.join(P0, "runs", args.run)
rows = judge_run(rd, plans, vlm, vcfg=VerifyConfig(), which=args.which)
if rows:
    print(json.dumps(dict(run=args.run, judge=vlm.model, n=len(rows), node_pass=sum(r["node_pass"] for r in rows) / len(rows), all_pass=sum(r["all_pass"] for r in rows) / len(rows)), ensure_ascii=False))
