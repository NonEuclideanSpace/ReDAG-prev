"""Write this host's entry into configs/machine.json (the file preflight.py, launch.py and jobq.py read).
GPUs default to every visible CUDA device; pass --gpus / --disabled to restrict. The file keeps other hosts' entries.
Examples:
  python scripts/machine_init.py --note "rented node, 5 GPUs, GPU0 serves the evaluator" --disabled 0
  python scripts/machine_init.py --gpus 1 2 3 --disabled 0 --note "lab server: never GPU 0, stop before 07:00"
Then: python scripts/preflight.py --render"""
import argparse
import json
import os
import socket

from common import P0

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", nargs="*", type=int, default=None, help="usable GPU indices; default: all visible")
ap.add_argument("--disabled", nargs="*", type=int, default=[], help="GPUs never to use (bad card, evaluator card, GPU 0 on the lab server)")
ap.add_argument("--note", default="")
ap.add_argument("--host", default=None, help="override hostname (default socket.gethostname())")
ap.add_argument("--path", default=os.path.join(P0, "configs", "machine.json"))
args = ap.parse_args()

host = args.host or socket.gethostname()
gpus = args.gpus
if gpus is None:
    try:
        import torch

        gpus = list(range(torch.cuda.device_count()))
    except Exception:
        gpus = []
names = []
try:
    import torch

    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except Exception:
    pass
machine = {"hosts": {}}
if os.path.exists(args.path):
    machine = json.load(open(args.path, encoding="utf-8"))
elif os.path.exists(os.path.join(P0, "configs", "machine.example.json")):
    machine = json.load(open(os.path.join(P0, "configs", "machine.example.json"), encoding="utf-8"))
machine.setdefault("hosts", {})[host] = {"gpus": [g for g in gpus if g not in set(args.disabled)], "disabled": sorted(set(args.disabled)), "note": args.note, "names": names}
os.makedirs(os.path.dirname(args.path), exist_ok=True)
with open(args.path, "w", encoding="utf-8") as f:
    json.dump(machine, f, ensure_ascii=False, indent=1)
print(json.dumps({host: machine["hosts"][host]}, ensure_ascii=False))
