"""Probe alternative count questions on the count items of a label set (evaluator fix, kickoff rule: change the view
and the wording before the model). For every count item four variants are asked on the native image and compared
with each reference (human / gpt / claude): agreement, and the gate-style numbers against --ref.
  A  baseline  "Are there N+1 or more X?"  yes/no logprob (what the Verifier asks today)
  B  number    "How many X are in this image? Answer with a single number."  greedy integer + first-digit logprobs
  C  detect    "Locate every X ..." JSON list of boxes, count the boxes
  D  strict    B with a "count only distinct, clearly visible X" clause
  python scripts/count_probe.py --items ../labels/seed_v4/items.jsonl --set xl_seed --tag qwen3.7-plus-v4d --out ../gates/count_probe.json"""
import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from common import P0, add_vlm_args, make_vlm, plan_dir

from draftdag.calibrate import load_plans
from draftdag.labels import effective_label, read_jsonl
from draftdag.vlm import BBOX_RE, _norm_tok
from draftdag.views import load_image, to_rgb

ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True)
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--refs", default="human:labels_human.jsonl,gpt:labels_gpt-6.1-sol.jsonl,claude:labels_claude.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
add_vlm_args(ap)
args = ap.parse_args()

vlm = make_vlm(args, require_gate=False)
items = [it for it in read_jsonl(args.items) if it["kind"] == "cond" and it.get("ntype") == "count"]
if args.limit:
    items = items[: args.limit]
plans = load_plans(plan_dir(args.set, args.tag), sorted({it["pid"] for it in items}))
lab_dir = os.path.dirname(args.items)
refs = {}
for spec in args.refs.split(","):
    name, fn = spec.split(":")
    p = os.path.join(lab_dir, fn)
    if os.path.exists(p):
        by_id = {it["id"]: it for it in read_jsonl(args.items)}
        refs[name] = {r["id"]: effective_label(by_id.get(r["id"], {}), r.get("label")) for r in read_jsonl(p)}
print("count items:", len(items), "refs:", {k: len(v) for k, v in refs.items()}, flush=True)

NUMW = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "none": 0, "no": 0}
cache = {}


def ask(image, text, max_tokens, logprobs):
    body = dict(model=vlm.model, messages=vlm._msg([image], text), max_tokens=max_tokens, temperature=0.0, **vlm._extra())
    if logprobs:
        body.update(logprobs=True, top_logprobs=20)
    d = vlm._post(body)
    ch = d["choices"][0]
    txt = ch["message"]["content"] or ""
    lp = None
    if logprobs:
        try:
            lp = ch["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            lp = None
    return txt, lp


def parse_int(txt):
    m = re.search(r"\d+", txt)
    if m:
        return int(m.group(0))
    for w, v in NUMW.items():
        if re.search(r"\b" + w + r"\b", txt.lower()):
            return v
    return None


def digit_dist(lp):
    """P over the first digit from the top-20 first-token logprobs (Qwen tokenizes digits singly)."""
    if not lp:
        return {}
    d = {}
    for e in lp:
        t = _norm_tok(e["token"])
        if t.isdigit() and len(t) == 1:
            d[int(t)] = d.get(int(t), 0.0) + math.exp(float(e["logprob"]))
        elif t in NUMW:
            d[NUMW[t]] = d.get(NUMW[t], 0.0) + math.exp(float(e["logprob"]))
    z = sum(d.values())
    return {k: v / z for k, v in d.items()} if z > 0 else {}


def yes_prob(lp):
    ly = ln = None
    for e in lp or []:
        t = _norm_tok(e["token"])
        if t in ("yes", "yes.", "y") and ly is None:
            ly = float(e["logprob"])
        elif t in ("no", "no.", "n") and ln is None:
            ln = float(e["logprob"])
    if ly is None and ln is None:
        return None
    ly = -30.0 if ly is None else ly
    ln = -30.0 if ln is None else ln
    return 1.0 / (1.0 + math.exp(-(ly - ln)))


def one(it):
    node = plans[it["pid"]].nodes[it["cid"][2:]]
    n, plural, cls = int(node["n"]), node["plural"], node["cls"]
    im = to_rgb(load_image(it["image"]))
    out = dict(id=it["id"], n=n, cls=cls, plural=plural)
    # A baseline
    txt, lp = ask(im, node["question"] + " Answer yes or no.", 2, True)
    py = yes_prob(lp)
    out["A"] = dict(p_yes=py, dup=(py is not None and py >= 0.5), raw=txt[:20])
    # B number
    qb = f"How many {plural} are in this image? Answer with a single number."
    txt, lp = ask(im, qb, 6, True)
    k = parse_int(txt)
    dd = digit_dist(lp)
    p_le = sum(v for d, v in dd.items() if d <= n)
    out["B"] = dict(k=k, raw=txt[:30], p_le_n=p_le, dup=(k is not None and k > n), dist={str(a): round(b, 3) for a, b in sorted(dd.items())})
    # D strict
    qd = f"How many {plural} are clearly visible in this image? Count only distinct {plural}; do not count things of another kind that merely look similar. Answer with a single number."
    txt, lp = ask(im, qd, 6, True)
    k = parse_int(txt)
    dd = digit_dist(lp)
    out["D"] = dict(k=k, raw=txt[:30], p_le_n=sum(v for d, v in dd.items() if d <= n), dup=(k is not None and k > n))
    # C detect
    qc = f'Locate every {cls} in the image. Output a JSON list with one {{"bbox_2d": [x1, y1, x2, y2]}} per {cls} you can see; output [] if there is none.'
    txt, _ = ask(im, qc, 400, False)
    boxes = BBOX_RE.findall(txt)
    out["C"] = dict(k=len(boxes), dup=(len(boxes) > n), raw=txt[:80].replace("\n", " "))
    return out


t0 = time.time()
with ThreadPoolExecutor(max_workers=args.vlm_workers) as ex:
    rows = list(ex.map(one, items))
sec = time.time() - t0
print(f"asked {len(rows)} items x 4 in {sec:.0f}s", flush=True)

summary = {}
for var in ("A", "B", "C", "D"):
    s = {}
    for name, ref in refs.items():
        agree = tot = 0
        d = dict(y=[0, 0], n=[0, 0], u=[0, 0])
        for r in rows:
            lab = ref.get(r["id"])
            if lab not in d:
                continue
            dup = bool(r[var]["dup"])
            d[lab][0] += 1
            d[lab][1] += int(dup)
            if lab in ("y", "n"):
                tot += 1
                agree += int(dup == (lab == "n"))
        s[name] = dict(agree=round(agree / tot, 3) if tot else None, n_yn=tot, false_fail=round(d["y"][1] / d["y"][0], 3) if d["y"][0] else None, detect_n=round(d["n"][1] / d["n"][0], 3) if d["n"][0] else None, detect_u=round(d["u"][1] / d["u"][0], 3) if d["u"][0] else None, n_y=d["y"][0], n_n=d["n"][0], n_u=d["u"][0])
    s["dup_rate"] = round(sum(bool(r[var]["dup"]) for r in rows) / len(rows), 3)
    summary[var] = s
    print(var, json.dumps(s), flush=True)
json.dump(dict(summary=summary, rows=rows, sec=round(sec, 1), model=vlm.model), open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("->", args.out)
