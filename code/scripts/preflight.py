import argparse
import json
import os

from common import P0

from draftdag.preflight import run_preflight

ap = argparse.ArgumentParser(description="Machine preflight: GPU allowlist, arithmetic consistency, optional render test, model checksums.")
ap.add_argument("--machine", default=os.path.join(P0, "configs", "machine.json"))
ap.add_argument("--render", action="store_true", help="also draw one 512px image per GPU with Qwen-Image-2.1 and check for black output")
ap.add_argument("--manifest", default=os.path.join(P0, "configs", "models.json"))
args = ap.parse_args()


def factory():
    from draftdag.executor_qi21 import QI21

    return QI21(steps=8, res=512)


out = run_preflight(args.machine, do_render=args.render, executor_factory=factory if args.render else None, manifest=args.manifest if os.path.exists(args.manifest) else None)
print(json.dumps(out, ensure_ascii=False, indent=1))
raise SystemExit(0 if out["pass"] else 1)
