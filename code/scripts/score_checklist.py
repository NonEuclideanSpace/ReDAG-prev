"""Final scoring on a public benchmark with its official checklist (T2I-CoReBench questions in data/checklists/),
answered by an independent judge. Never used inside the loop. One row per prompt: fraction of official questions
answered as the checklist expects (questions tagged as "absent"/negative count a "no" as correct)."""
import argparse
import json
import os

from common import P0, add_vlm_args, make_vlm

from redag.report import run_metrics
from redag.views import load_image, resize_min_side, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--checklist", required=True, help="data/checklists/<name>.json")
ap.add_argument("--which", default="final", choices=["final"])
ap.add_argument("--side", type=int, default=1024)
add_vlm_args(ap)
args = ap.parse_args()

vlm = make_vlm(args, require_gate=False)
checks = json.load(open(args.checklist, encoding="utf-8"))
rd = args.run if os.path.isabs(args.run) else os.path.join(P0, "runs", args.run)
m = run_metrics(rd)
out_path = os.path.join(rd, f"checklist_{vlm.tag()}.jsonl")
tot, n = 0.0, 0
with open(out_path, "w", encoding="utf-8") as f:
    for pid, v in sorted(m.items()):
        cl = checks.get(pid)
        if not cl:
            continue
        im = to_rgb(load_image(v["final"]))
        im = resize_min_side(im, args.side) if min(im.size) > args.side else im
        items = [dict(image=im, question=q["question"], tpl="a", inv=False) for q in cl]
        res = vlm.yesno(items)
        correct = []
        for q, (p, _) in zip(cl, res):
            neg = any("neg" in t or "absent" in t for t in q.get("tags", []))
            ok = (p < 0.5) if neg else (p >= 0.5)
            correct.append(dict(q=q["question"], p=round(p, 3), ok=bool(ok), tags=q.get("tags", [])))
        score = sum(c["ok"] for c in correct) / len(correct)
        tot += score
        n += 1
        f.write(json.dumps(dict(pid=pid, image=v["final"], score=score, items=correct), ensure_ascii=False) + "\n")
print(json.dumps(dict(run=args.run, judge=vlm.model, n=n, mean_score=tot / n if n else None, out=out_path), ensure_ascii=False))
