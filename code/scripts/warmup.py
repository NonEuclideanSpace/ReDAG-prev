import argparse
import os

from common import P0, plan_dir

from redag.executor_qi21 import load_pipe
from redag.policy_qi21 import QI21Policy
from redag.warmup import Warmup, accepted_from_runs

ap = argparse.ArgumentParser(description="Regression warm-up on accepted edits (before GRPO only).")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=768)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--omega", type=float, default=0.5)
ap.add_argument("--min-samples", type=int, default=100)
args = ap.parse_args()

out = os.path.join(P0, "runs", args.out)
os.makedirs(out, exist_ok=True)
samples = accepted_from_runs([r if os.path.isabs(r) else os.path.join(P0, "runs", r) for r in args.runs], plan_dir(args.set, args.tag), os.path.join(out, "samples.jsonl"))
print("accepted samples", len(samples), flush=True)
if len(samples) < args.min_samples:
    raise SystemExit("too few accepted edits for a warm-up")
pipe, device = load_pipe()
pol = QI21Policy(pipe, device, res=args.res)
Warmup(pol, samples, out, lr=args.lr, batch=args.batch, max_steps=args.steps, omega=args.omega).fit()
print("DONE", flush=True)
