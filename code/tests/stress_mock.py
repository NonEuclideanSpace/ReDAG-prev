import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import lab_plan

from redag.conditions import build_conditions
from redag.loop import Loop, LoopConfig
from redag.mock import MockExecutor, MockVLM, Truth


def truly_satisfied(plan, conds, truth):
    n = 0
    for c in conds:
        t = c.ntype
        if t == "entity":
            n += bool(truth.present.get(c.node))
        elif t in ("attribute", "state"):
            n += bool(truth.present.get(c.participants[0]) and truth.ok.get(c.node))
        elif t in ("action", "relation"):
            n += all(truth.present.get(u) for u in c.participants) and bool(truth.ok.get(c.node))
        elif t == "absent":
            n += c.node not in truth.absent_present
        elif c.kind == "count":
            n += truth.extras.get(c.cls, 0) == 0
        elif c.kind == "pair":
            n += c.node not in truth.merged and all(truth.present.get(u) for u in c.participants)
    return n


def run(seed, perfect):
    plan = lab_plan()
    conds = build_conditions(plan)
    out = tempfile.mkdtemp(prefix="dagstress_")
    try:
        ex = MockExecutor(plan, p_fix=0.6, p_collateral=0.35, p_dup=0.4, seed=seed, e_present=0.7, a_ok=0.6, r_ok=0.5, p_extra=0.3, p_merge=0.3, p_absent=0.3)
        vlm = MockVLM(plan, conds, f=0.0 if perfect else 0.06, miss=0.0 if perfect else 0.1, f2=0.0 if perfect else 0.03, seed=seed)
        cfg = LoopConfig(mode="repair", n0=2, G=3, max_rounds=6, max_tries=2, max_parallel=3, seed=seed)
        t0 = time.time()
        loop = Loop(ex, vlm, cfg, out)
        summ = loop.run_prompt(plan)
        el = time.time() - t0
        tr = json.load(open(os.path.join(out, "img", plan.id, "trace.json")))
        ops = {}
        viol = []
        q_per_cand = []
        from PIL import Image

        for rd in tr["rounds"]:
            tcanvas = Truth.from_json(Image.open(rd["canvas"]).info["truth"])
            before = truly_satisfied(plan, conds, tcanvas)
            for br in rd["branches"]:
                ops[br["op"]] = ops.get(br["op"], 0) + 1
                if perfect and br["op"] == "add" and tcanvas.present.get(br["node"]):
                    viol.append(("add-on-present", rd["round"], br["node"]))
                if perfect and br["op"] == "fix" and br["ntype"] == "entity" and not tcanvas.present.get(br["node"]):
                    pass
                for cj in br["candidates"]:
                    q_per_cand.append(len(br["affected"]))
            if rd["merged"]:
                p = os.path.join(os.path.dirname(rd["canvas"]), f"canvas_r{rd['round']:02d}.png")
                after = truly_satisfied(plan, conds, Truth.from_json(Image.open(p).info["truth"]))
                if perfect and after < before:
                    viol.append(("merge-decreased-truth", rd["round"], before, after))
        tf = Truth.from_json(Image.open(tr["final"]["image"]).info["truth"])
        return dict(seed=seed, sec=round(el, 1), rounds=len(tr["rounds"]), ops=ops, viol=viol, start_fail=len(conds) - truly_satisfied(plan, conds, Truth.from_json(Image.open(tr["stage1"]["drafts"][tr["stage1"]["pick"]]).info["truth"])) - 2, final_fail=len(conds) - truly_satisfied(plan, conds, tf) - 2, reported_all_pass=summ["all_pass"], mean_affected=round(sum(q_per_cand) / max(1, len(q_per_cand)), 1), n_cond=len(conds))
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    perfect = "--noisy" not in sys.argv
    tot_v = 0
    for s in range(n):
        r = run(100 + s, perfect)
        tot_v += len(r["viol"])
        print(json.dumps(r))
    print("violations:", tot_v)
