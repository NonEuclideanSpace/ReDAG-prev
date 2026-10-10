"""Run one shard of a command per allowed GPU of this machine (configs/machine.json, disabled GPUs excluded).
The command template gets {shard} = "i/n" and {gpu}; each process sees only its GPU via CUDA_VISIBLE_DEVICES.
Example:
  python scripts/launch.py --cmd "python scripts/run_loop.py --set t20 --tag qwen3.7-plus-v4 --run t20_repair --mode repair --shard {shard}"
"""
import argparse
import os
import subprocess
import time

from common import P0

from draftdag.preflight import allowed_gpus, load_machine, require_preflight

ap = argparse.ArgumentParser()
ap.add_argument("--cmd", required=True)
ap.add_argument("--gpus", default=None, help="comma list to restrict further")
ap.add_argument("--name", default=None)
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()

gpus, disabled, host = allowed_gpus(load_machine())
if args.gpus:
    want = [int(x) for x in args.gpus.split(",")]
    bad = [g for g in want if g in disabled]
    if bad:
        raise SystemExit(f"GPU(s) {bad} are disabled in configs/machine.json for {host}")
    gpus = [g for g in want if g in gpus]
if not gpus:
    raise SystemExit("no allowed GPUs")
require_preflight()
name = args.name or time.strftime("launch_%m%d_%H%M%S")
os.makedirs(os.path.join(P0, "logs"), exist_ok=True)
procs = []
for i, g in enumerate(gpus):
    cmd = args.cmd.format(shard=f"{i}/{len(gpus)}", gpu=g)
    log = os.path.join(P0, "logs", f"{name}_gpu{g}.log")
    print(f"[gpu {g}] {cmd} > {log}", flush=True)
    if args.dry:
        continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), P0_DEVICE="cuda:0")
    procs.append(subprocess.Popen(cmd, shell=True, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=os.path.join(P0, "code")))
for p in procs:
    p.wait()
print("ALL_DONE", [p.returncode for p in procs], flush=True)
