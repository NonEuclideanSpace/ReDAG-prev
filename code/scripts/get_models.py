"""Fetch model weights into models/<name>/ and record them in configs/models.json (what preflight checks).
Sources, tried in this order unless --source is given: an rsync source (--from, e.g. the lab server where the
weights already are), ModelScope (domestic, fast on AutoDL), Hugging Face through the hf-mirror endpoint.
  python scripts/get_models.py --names Qwen-Image-2.1 Qwen3.8-27B-FP8 --from user@host:/path/to/models
  python scripts/get_models.py --names Qwen3.8-27B-FP8 --source modelscope
  python scripts/get_models.py --names Qwen-Image-2.1 --link /path/to/Qwen-Image-2.1   # weights already on this machine: symlink, no copy
Model ids live in configs/models.sources.json (edit there if a repo id differs). After a fetch the sizes and
sha256 of every weight file are written with scripts/make_manifest.py (pass --no-hash to skip hashing)."""
import argparse
import json
import os
import subprocess
import sys

from common import P0

SOURCES_PATH = os.path.join(P0, "configs", "models.sources.json")
DEFAULT_SOURCES = {
    "Qwen-Image-2.1": {"modelscope": "Qwen/Qwen-Image-2.1", "hf": "Qwen/Qwen-Image-2.1", "lab": None, "role": "executor (text-to-image and editing)"},
    "Qwen3.8-27B-FP8": {"modelscope": "Qwen/Qwen3.8-27B-FP8", "hf": "Qwen/Qwen3.8-27B-FP8", "lab": None, "role": "evaluator, FP8 for one 48 GB card (vLLM)"},
    "Qwen3.8-27B": {"modelscope": "Qwen/Qwen3.8-27B", "hf": "Qwen/Qwen3.8-27B", "lab": None, "role": "evaluator, bf16 for two cards tensor parallel"},
    "FLUX.2-klein": {"modelscope": "black-forest-labs/FLUX.2-klein", "hf": "black-forest-labs/FLUX.2-klein", "lab": None, "role": "second executor base"},
    "Qwen3-VL-8B-Instruct": {"modelscope": "Qwen/Qwen3-VL-8B-Instruct", "hf": "Qwen/Qwen3-VL-8B-Instruct", "lab": None, "role": "fallback evaluator (previous stage)"},
}

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True)
ap.add_argument("--source", default=None, choices=[None, "rsync", "modelscope", "hf"])
ap.add_argument("--from", dest="from_host", default=None, help="rsync source: user@host:/dir (model dir name appended) or user@host (uses the lab path from models.sources.json)")
ap.add_argument("--link", default=None, help="symlink models/<name> to this existing directory (single name)")
ap.add_argument("--no-hash", action="store_true")
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()

sources = DEFAULT_SOURCES
if os.path.exists(SOURCES_PATH):
    sources = dict(DEFAULT_SOURCES, **json.load(open(SOURCES_PATH, encoding="utf-8")))
else:
    os.makedirs(os.path.dirname(SOURCES_PATH), exist_ok=True)
    json.dump(DEFAULT_SOURCES, open(SOURCES_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
models_root = os.path.join(P0, "models")
os.makedirs(models_root, exist_ok=True)


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    if args.dry:
        return 0
    return subprocess.call(cmd)


def fetch_rsync(name, dest):
    if ":" in args.from_host:  # user@host:/dir -> /dir/<name>
        src = args.from_host.rstrip("/") + "/" + name + "/"
    else:  # user@host -> the lab path recorded for this model
        lab = sources[name].get("lab")
        if not lab:
            print(f"no lab path for {name}; trying the next source", file=sys.stderr)
            return False
        src = args.from_host + ":" + lab.rstrip("/") + "/"
    return run(["rsync", "-a", "--partial", "--info=progress2", src, dest + "/"]) == 0


def fetch_modelscope(name, dest):
    rid = sources[name]["modelscope"]
    if args.dry:
        print("+ modelscope snapshot_download", rid, "->", dest)
        return True
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("pip install modelscope first", file=sys.stderr)
        return False
    snapshot_download(rid, local_dir=dest)
    return True


def fetch_hf(name, dest):
    rid = sources[name]["hf"]
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_OFFLINE"] = "0"
    if args.dry:
        print("+ huggingface_hub snapshot_download", rid, "->", dest, "via", os.environ["HF_ENDPOINT"])
        return True
    from huggingface_hub import snapshot_download

    snapshot_download(rid, local_dir=dest, max_workers=8)
    return True


done = []
for name in args.names:
    if name not in sources:
        raise SystemExit(f"unknown model {name}; add it to {SOURCES_PATH}")
    dest = os.path.join(models_root, name)
    if args.link:
        if os.path.lexists(dest):
            os.remove(dest) if os.path.islink(dest) else None
        os.symlink(args.link, dest)
        print("linked", dest, "->", args.link)
        done.append(name)
        continue
    os.makedirs(dest, exist_ok=True)
    order = [args.source] if args.source else (["rsync"] if args.from_host else []) + ["modelscope", "hf"]
    ok = False
    for s in order:
        try:
            ok = {"rsync": fetch_rsync, "modelscope": fetch_modelscope, "hf": fetch_hf}[s](name, dest)
        except Exception as ex:
            print(f"{s} failed for {name}: {ex}", file=sys.stderr)
            ok = False
        if ok:
            print("fetched", name, "via", s)
            break
    if not ok:
        raise SystemExit(f"could not fetch {name}")
    done.append(name)
if done and not args.dry:
    subprocess.call([sys.executable, os.path.join(P0, "code", "scripts", "make_manifest.py"), "--dirs", *done] + (["--no-hash"] if args.no_hash else []))
