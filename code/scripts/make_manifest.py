import argparse
import json
import os

from common import P0

from draftdag.preflight import sha256_file

ap = argparse.ArgumentParser(description="Record sizes and sha256 of the model files under models/<dir> into configs/models.json (used by preflight and by the final deliverable check).")
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--out", default=os.path.join(P0, "configs", "models.json"))
ap.add_argument("--no-hash", action="store_true")
args = ap.parse_args()

man = json.load(open(args.out, encoding="utf-8")) if os.path.exists(args.out) else {"models": []}
have = {m["dir"]: m for m in man["models"]}
for d in args.dirs:
    root = os.path.join(P0, "models", d)
    files = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            if fn.endswith((".safetensors", ".json", ".txt", ".model", ".bin")):
                rel = os.path.relpath(p, root)
                files.append(dict(path=rel, size=os.path.getsize(p), sha256=None if args.no_hash else sha256_file(p)))
                print(d, rel, files[-1]["size"], flush=True)
    have[d] = dict(dir=d, files=files)
man["models"] = list(have.values())
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(man, f, indent=1)
print("->", args.out)
