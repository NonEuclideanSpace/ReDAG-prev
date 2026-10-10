import json
import os

import numpy as np
from PIL import Image, ImageDraw

from .conditions import build_conditions
from .verify import Verifier, VerifyConfig
from .views import load_image, to_rgb


def run_metrics(run_dir):
    """Per-prompt metrics from trace.json: node pass rate, all-pass, calls, seconds, drafts, rounds.
    External baselines only need runs/<run>/img/<pid>/final.png and a trace.json whose "final" has "image"
    (optionally "budget": {"calls", "elapsed"}); the loop's own fields are read when present."""
    out = {}
    root = os.path.join(run_dir, "img")
    if not os.path.isdir(root):
        return out
    for pid in sorted(os.listdir(root)):
        tp = os.path.join(root, pid, "trace.json")
        if not os.path.exists(tp):
            continue
        tr = json.load(open(tp, encoding="utf-8"))
        fin = tr.get("final") or {}
        image = fin.get("image") or os.path.join(root, pid, "final.png")
        if not os.path.isabs(image):
            image = os.path.join(root, pid, image)
        if not os.path.exists(image):
            continue
        col = fin.get("column")
        node_pass, denom, all_pass = None, 0, fin.get("all_pass")
        if col:
            sts = col["status"]
            nt = col.get("ntype", {})
            node_cids = [c for c in sts if nt.get(c) not in ("quality",) and not c.startswith("q:")]
            scored = [c for c in node_cids if sts[c] in ("pass", "fail")]
            unasked = [c for c in node_cids if sts[c] == "unasked"]
            n_pass = sum(1 for c in scored if sts[c] == "pass")
            denom = len(scored) + len(unasked)
            node_pass = n_pass / denom if denom else 0.0
        budget = fin.get("budget") or {}
        st1 = tr.get("stage1") or {}
        out[pid] = dict(node_pass=node_pass, all_pass=None if all_pass is None else bool(all_pass), calls=budget.get("calls"), elapsed=budget.get("elapsed"), n_drafts=st1.get("n_drafts"), rounds=len(tr.get("rounds") or []), decision=st1.get("decision"), final=image, n_cond=denom)
    return out


def paired_bootstrap(a, b, n=4000, seed=0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    if len(d) == 0:
        return dict(mean=None, lo=None, hi=None, n=0)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    return dict(mean=float(d.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)), n=int(len(d)))


def compare(run_a, run_b, metrics=("node_pass", "all_pass", "calls", "elapsed")):
    ma, mb = run_metrics(run_a), run_metrics(run_b)
    common = sorted(set(ma) & set(mb))
    out = dict(n=len(common))
    for k in metrics:
        ok = [p for p in common if ma[p].get(k) is not None and mb[p].get(k) is not None]
        out[k] = paired_bootstrap([float(ma[p][k]) for p in ok], [float(mb[p][k]) for p in ok])
        out[k]["a"] = float(np.mean([float(ma[p][k]) for p in ok])) if ok else None
        out[k]["b"] = float(np.mean([float(mb[p][k]) for p in ok])) if ok else None
    return out


def contact_sheet(paths, out, cols=4, side=320, labels=None):
    """Grid of images for mandatory visual checks before any number is reported."""
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return None
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * side, rows * (side + 18)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(paths):
        im = to_rgb(load_image(p))
        im.thumbnail((side, side))
        x, y = (i % cols) * side, (i // cols) * (side + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 2, y + 2), (labels[i] if labels else os.path.basename(p))[:48], fill=(0, 0, 0))
    sheet.save(out, quality=90)
    return out


def judge_run(run_dir, plans, judge_vlm, out_path=None, vcfg=None, which="final"):
    """Score a run's final images with an independent judge model on the plan questions (gated, two-stage), one row per prompt."""
    vcfg = vcfg or VerifyConfig()
    rows = []
    root = os.path.join(run_dir, "img")
    for pid in sorted(os.listdir(root)):
        tp = os.path.join(root, pid, "trace.json")
        if not os.path.exists(tp) or pid not in plans:
            continue
        tr = json.load(open(tp, encoding="utf-8"))
        img = tr["final"]["image"] if which == "final" else tr["stage1"]["drafts"][0]
        plan = plans[pid]
        conds = build_conditions(plan)
        col = Verifier(judge_vlm, plan, conds, vcfg).verify(img, tag="judge")
        node_cids = [c.cid for c in conds if c.kind != "quality"]
        n_pass = sum(1 for c in node_cids if col.passes(c))
        rows.append(dict(pid=pid, image=img, node_pass=n_pass / len(node_cids), all_pass=col.all_pass(conds), status=col.statuses(), judge=judge_vlm.model))
    out_path = out_path or os.path.join(run_dir, f"judge_{judge_vlm.tag() if hasattr(judge_vlm, 'tag') else 'x'}_{which}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows
