import argparse
import json
import os

from common import P0, add_vlm_args, make_vlm, plan_dir

from redag.calibrate import GATE, run_calibration

ap = argparse.ArgumentParser(description="Evaluator gate: score the label set with the production verification path and compare with reference labels.")
ap.add_argument("--items", required=True, help="labels/<set>/items.jsonl")
ap.add_argument("--ref", required=True, help="reference labels jsonl (human-adjudicated)")
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--full-side", type=int, default=896)
ap.add_argument("--crop-min", type=int, default=448)
ap.add_argument("--crop-max", type=int, default=1024)
ap.add_argument("--no-two-stage", action="store_true")
ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--count-mode", default="number", choices=["number", "yesno"], help="count conditions: number question + one-box-per-instance (default) or the old inverted yes/no")
ap.add_argument("--rescue-min", type=float, default=None, help="a stage-1 fail passes only if the second view scores at least this (default: --thr)")
ap.add_argument("--confirm-band", default=None, help="lo,hi: stage-1 passes with a score in [lo,hi) are confirmed on the second view")
ap.add_argument("--three-way", action="store_true", help="stage-1 questions offer yes / partly / no; partly counts against the condition")
ap.add_argument("--out", default=None, help="gate file; default gates/evaluator_<model>.json (the one the loop reads)")
add_vlm_args(ap)
args = ap.parse_args()

vlm = make_vlm(args, require_gate=False)
vcfg = dict(full_side=args.full_side, crop_min=args.crop_min, crop_max=args.crop_max, two_stage=not args.no_two_stage, thr=args.thr, count_mode=args.count_mode, rescue_min=args.rescue_min, confirm_band=[float(x) for x in args.confirm_band.split(",")] if args.confirm_band else None, three_way=args.three_way)
gate = run_calibration(vlm, args.items, args.ref, plan_dir(args.set, args.tag), vcfg_kw=vcfg, out_path=args.out)
print(json.dumps({k: gate[k] for k in ("model", "pass", "checks", "summary", "n_scored", "coord_mode", "sec")}, ensure_ascii=False, indent=1))
print("by type:", json.dumps(gate["metrics"]["by_type"], ensure_ascii=False))
print("by size:", json.dumps(gate["metrics"]["by_size"], ensure_ascii=False))
