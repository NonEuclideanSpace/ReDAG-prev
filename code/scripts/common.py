import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from draftdag.paths import P0  # noqa: E402


def add_vlm_args(ap):
    ap.add_argument("--vlm-url", default=os.environ.get("VLM_URL", "http://127.0.0.1:18080/v1"))
    ap.add_argument("--vlm-model", default=os.environ.get("VLM_MODEL", "Qwen3.8-27B"))
    ap.add_argument("--vlm-mode", default=os.environ.get("VLM_MODE", "logprob"), choices=["logprob", "graded"])
    ap.add_argument("--vlm-key-env", default="VLM_API_KEY")
    ap.add_argument("--vlm-workers", type=int, default=8)
    ap.add_argument("--vlm-cache", default=None)
    ap.add_argument("--no-gate", action="store_true", help="only for calibration and labelling runs; the loop never accepts this")


def secret(name):
    """An API key from the environment, else from P0/secrets.env (NAME=value lines); never printed."""
    if os.environ.get(name):
        return os.environ[name]
    p = os.environ.get("P0_SECRETS", os.path.join(P0, "secrets.env"))
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def make_vlm(args, require_gate=True):
    from draftdag.vlm import VLM

    v = VLM(args.vlm_url, args.vlm_model, api_key=secret(args.vlm_key_env), mode=args.vlm_mode, workers=args.vlm_workers, cache_path=args.vlm_cache, require_gate=require_gate and not args.no_gate)
    if require_gate and not args.no_gate:
        v.check_gate()
    elif v.coord_mode is None:
        gp = v.gate_path()
        if os.path.exists(gp):
            v.coord_mode = json.load(open(gp)).get("coord_mode")
    return v


def load_set(name):
    path = name if name.endswith(".jsonl") else os.path.join(P0, "data", "sets", name + ".jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plan_dir(set_name, planner_tag):
    return os.path.join(P0, "data", "plans", set_name, planner_tag)


def load_json(path, default=None):
    if path and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return default


def run_dir(name):
    """A run given as an absolute path, a path relative to P0 (e.g. labels/seed-1008/drafts_one_q4), or a name under runs/."""
    if os.path.isabs(name):
        return name
    if os.path.isdir(os.path.join(P0, name)):
        return os.path.join(P0, name)
    return os.path.join(P0, "runs", name)


def shard(rows, spec):
    if not spec:
        return rows
    i, n = [int(x) for x in spec.split("/")]
    return rows[i::n]


def ids_filter(rows, path):
    """Keep only rows whose id is listed in path (one id per line, # comments); None keeps all."""
    if not path:
        return rows
    keep = set()
    for line in open(path, encoding="utf-8"):
        t = line.strip()
        if t and not t.startswith("#"):
            keep.add(t.split()[0])
    return [r for r in rows if r["id"] in keep]
