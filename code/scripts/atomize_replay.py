"""Re-apply the acceptance rules of the current planner.atomize_pass to the planner answers stored in an atomize run's
_atomize.jsonl (raw = first answer, repair.raw = repair answer), without any API call. A prompt whose first-round
problems (the repair message) come out exactly as stored got the same repair question, so its stored repair answer is
the one the planner gave to it and the replayed plan is what the current rules make of that run: written to --out-tag.
A prompt whose first-round problems differ (or that needed a parse retry) would have been asked a different repair
question: listed in <out dir>/_replay_rerun.txt for a fresh run with scripts/atomize_plans.py.
  python scripts/atomize_replay.py --src-tag qwen3.7-plus-v8 --out-tag qwen3.7-plus-v81 [--sets cb_ma,cb_mi]"""
import argparse
import glob
import json
import os

from common import P0, plan_dir

from redag.planner import atomize_pass

ap = argparse.ArgumentParser()
ap.add_argument("--src-tag", required=True)
ap.add_argument("--out-tag", required=True)
ap.add_argument("--sets", default=None)
args = ap.parse_args()

stats = dict(same=0, rerun=0, unchanged_plan=0, changed_plan=0, missing=0)
for lp in sorted(glob.glob(os.path.join(P0, "data", "plans", "*", args.src_tag, "_atomize.jsonl"))):
    s = lp.split(os.sep)[-3]
    if args.sets and s not in args.sets.split(","):
        continue
    dst = plan_dir(s, args.out_tag)
    os.makedirs(dst, exist_ok=True)
    rerun = []
    done = set()
    if os.path.exists(os.path.join(dst, "_atomize.jsonl")):
        done = {json.loads(l)["pid"] for l in open(os.path.join(dst, "_atomize.jsonl"), encoding="utf-8")}
    for line in open(lp, encoding="utf-8"):
        q = json.loads(line)
        pid = q["pid"]
        if pid in done:
            continue
        bp = os.path.join(plan_dir(s, q["base_tag"]), pid + ".json")
        if not os.path.exists(bp) or "raw" not in q or not isinstance(q.get("raw"), dict):
            stats["missing"] += 1
            rerun.append(pid)
            continue
        d = json.load(open(bp, encoding="utf-8"))
        raw = q["raw"]
        rraw = (q.get("repair") or {}).get("raw")
        asked = []

        def fake(messages, model=None, tag="", **kw):
            if tag.startswith("atomize_repair"):
                asked.append(messages[-1]["content"])
                if rraw is None:
                    raise RuntimeError("no stored repair answer")
                return json.dumps(rraw)
            if tag.startswith("atomize_retry"):
                raise RuntimeError("retry not replayable")
            return json.dumps(raw)

        out, info = atomize_pass(d, chat=fake)
        same = (not q.get("parse_retry")) and info.get("problems", {}) == (q.get("problems") or {}) and not info.get("error", "") == "parse"
        if not same:
            stats["rerun"] += 1
            rerun.append(pid)
            continue
        stats["same"] += 1
        out = json.loads(json.dumps(out))
        old_out = json.load(open(os.path.join(plan_dir(s, args.src_tag), pid + ".json"), encoding="utf-8"))
        meta = dict(out.get("meta", {}), planner=(old_out.get("meta") or {}).get("planner", "") + "+replay", base_tag=q["base_tag"], replay_of=args.src_tag)
        out["meta"] = meta
        old_core = dict(old_out, meta=None)
        new_core = dict(out, meta=None)
        stats["unchanged_plan" if json.dumps(old_core, sort_keys=True) == json.dumps(new_core, sort_keys=True) else "changed_plan"] += 1
        with open(os.path.join(dst, pid + ".json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        info["replay_of"] = args.src_tag
        with open(os.path.join(dst, "_atomize.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(pid=pid, set=s, base_tag=q["base_tag"], n_before=len(d["nodes"]), n_after=len(out["nodes"]), **info), ensure_ascii=False) + "\n")
    with open(os.path.join(dst, "_replay_rerun.txt"), "a", encoding="utf-8") as f:
        for pid in rerun:
            f.write(pid + "\n")
    print(s, "rerun", len(rerun), flush=True)
print(json.dumps(stats))
