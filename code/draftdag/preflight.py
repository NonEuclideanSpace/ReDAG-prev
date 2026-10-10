"""Machine preflight: GPU allowlist, per-GPU arithmetic and read-back consistency, optional one-image render test with
black-image detection, model file checksums. A GPU listed in `disabled` is never used; the loop refuses to start
unless preflight.json for this host is fresh and passes."""
import hashlib
import json
import os
import platform
import socket
import time

import numpy as np

from .paths import P0, gates_dir


def load_machine(path=None):
    path = path or os.path.join(P0, "configs", "machine.json")
    if not os.path.exists(path):
        raise RuntimeError(f"machine config missing: {path} (copy configs/machine.example.json and edit)")
    return json.load(open(path, encoding="utf-8"))


def allowed_gpus(machine):
    host = socket.gethostname()
    m = machine.get("hosts", {}).get(host) or machine.get("hosts", {}).get("default") or {}
    gpus = m.get("gpus", [])
    disabled = set(m.get("disabled", []))
    return [g for g in gpus if g not in disabled], sorted(disabled), host


def gpu_arith_test(idx, n=2048, reps=3):
    import torch

    dev = torch.device(f"cuda:{idx}")
    torch.cuda.set_device(dev)
    g = torch.Generator(device=dev).manual_seed(123)
    a = torch.randn((n, n), generator=g, device=dev)
    b = torch.randn((n, n), generator=g, device=dev)
    ref = (a @ b).float().cpu()
    ok = True
    for _ in range(reps):
        c = (a @ b).float().cpu()
        if not torch.equal(c, ref):
            ok = False
    x = torch.arange(0, 1 << 20, device=dev, dtype=torch.int64)
    rb = x.cpu()
    if not torch.equal(rb, torch.arange(0, 1 << 20, dtype=torch.int64)):
        ok = False
    nan = bool(torch.isnan(ref).any())
    return dict(gpu=idx, consistent=ok, nan=nan, name=torch.cuda.get_device_name(dev), mem_gb=round(torch.cuda.get_device_properties(dev).total_memory / 2 ** 30, 1))


def is_black(im, thr=2.0):
    a = np.asarray(im.convert("L"), dtype=np.float32)
    return float(a.mean()) < thr or float(a.std()) < 1.0


def render_test(idx, executor_factory):
    """Draw one tiny image on the GPU and check it is not black or constant."""
    os.environ["P0_DEVICE"] = f"cuda:{idx}"
    ex = executor_factory()
    ims = ex.draw("a red apple on a wooden table, photo", 1, 7, size=512)
    return dict(gpu=idx, black=is_black(ims[0]), size=list(ims[0].size))


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_models(manifest_path):
    if not os.path.exists(manifest_path):
        return dict(checked=0, missing=["manifest"], bad=[])
    man = json.load(open(manifest_path, encoding="utf-8"))
    missing, bad, n = [], [], 0
    for m in man.get("models", []):
        for f in m.get("files", []):
            p = os.path.join(P0, "models", m["dir"], f["path"])
            if not os.path.exists(p):
                missing.append(p)
                continue
            n += 1
            if f.get("size") and os.path.getsize(p) != f["size"]:
                bad.append(p)
            elif f.get("sha256") and f.get("verify", True) and sha256_file(p) != f["sha256"]:
                bad.append(p)
    return dict(checked=n, missing=missing, bad=bad)


def run_preflight(machine_path=None, do_render=False, executor_factory=None, manifest=None):
    machine = load_machine(machine_path)
    gpus, disabled, host = allowed_gpus(machine)
    out = dict(host=host, platform=platform.platform(), time=time.strftime("%F %T"), gpus=gpus, disabled=disabled, arith=[], render=[], models=None, pass_=True)
    try:
        import torch

        for g in gpus:
            r = gpu_arith_test(g)
            out["arith"].append(r)
            if not r["consistent"] or r["nan"]:
                out["pass_"] = False
        if do_render and executor_factory is not None:
            for g in gpus:
                r = render_test(g, executor_factory)
                out["render"].append(r)
                if r["black"]:
                    out["pass_"] = False
    except Exception as ex:
        out["error"] = str(ex)
        out["pass_"] = False
    if manifest:
        out["models"] = check_models(manifest)
        if out["models"]["missing"] or out["models"]["bad"]:
            out["pass_"] = False
    out["pass"] = out.pop("pass_")
    path = os.path.join(gates_dir(), f"preflight_{host}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def require_preflight(max_age_h=24):
    host = socket.gethostname()
    path = os.path.join(gates_dir(), f"preflight_{host}.json")
    if not os.path.exists(path):
        raise RuntimeError(f"no preflight for {host}; run scripts/preflight.py")
    d = json.load(open(path, encoding="utf-8"))
    age = (time.time() - os.path.getmtime(path)) / 3600.0
    if not d.get("pass") or age > max_age_h:
        raise RuntimeError(f"preflight for {host} failed or stale ({age:.1f} h): {path}")
    return d
