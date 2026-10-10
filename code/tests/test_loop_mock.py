import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import lab_plan

from draftdag.conditions import build_conditions
from draftdag.loop import Loop, LoopConfig
from draftdag.mock import MockExecutor, MockVLM, Truth
from draftdag.verify import Verifier, VerifyConfig


def _truth_of(path):
    from PIL import Image

    im = Image.open(path)
    return Truth.from_json(im.info["truth"])


def test_verify_matches_truth_without_noise():
    plan = lab_plan()
    conds = build_conditions(plan)
    ex = MockExecutor(plan, seed=1)
    vlm = MockVLM(plan, conds, f=0.0, miss=0.0, f2=0.0, seed=1)
    ims = ex.draw(plan.prompt, 3, 7)
    ver = Verifier(vlm, plan, conds, VerifyConfig())
    cols = ver.verify_batch(ims)
    for im, col in zip(ims, cols):
        tr = Truth.from_json(im.info["truth"])
        for c in conds:
            st = col.status(c.cid)
            if c.ntype == "entity":
                assert st == ("pass" if tr.present[c.node] else "fail"), (c.cid, st)
            if c.ntype in ("attribute", "state"):
                par = c.participants[0]
                if not tr.present[par]:
                    assert st == "unasked", (c.cid, st)
                else:
                    assert st == ("pass" if tr.ok[c.node] else "fail")
            if c.kind == "count":
                assert st == ("fail" if tr.extras.get(c.cls, 0) > 0 else "pass")
                assert col.recs[c.cid].v1.get("method") == "number" and col.recs[c.cid].v1.get("k") is not None, col.recs[c.cid].v1
                if st == "fail":
                    assert col.recs[c.cid].v2.get("method") == "detect", col.recs[c.cid].v2
        for v in plan.entities():
            if tr.present[v]:
                assert v in col.boxes, v


def test_loop_repairs_and_respects_rules():
    plan = lab_plan()
    conds = build_conditions(plan)
    out = tempfile.mkdtemp(prefix="dagloop_")
    try:
        ex = MockExecutor(plan, p_fix=0.7, p_collateral=0.3, p_dup=0.3, seed=3, e_present=0.7, a_ok=0.6, r_ok=0.5, p_extra=0.2, p_merge=0.2, p_absent=0.3)
        vlm = MockVLM(plan, conds, f=0.05, miss=0.05, f2=0.02, seed=3)
        cfg = LoopConfig(mode="repair", n0=2, G=3, max_rounds=6, max_tries=2, max_parallel=3, seed=5)
        loop = Loop(ex, vlm, cfg, out)
        summ = loop.run_prompt(plan)
        tr = json.load(open(os.path.join(out, "img", plan.id, "trace.json")))
        assert tr["stage1"]["n_drafts"] == 2
        start_col = tr["stage1"]["columns"][tr["stage1"]["pick"]]
        n_fail0 = sum(1 for cid, st in start_col["status"].items() if st == "fail")
        assert tr["rounds"], "no rounds were run although drafts had failures"
        ops_seen = set()
        for rd in tr["rounds"]:
            assert rd["branches"], "a round without branches"
            keys = [b["node"] for b in rd["branches"]]
            assert len(keys) == len(set(keys))
            for br in rd["branches"]:
                ops_seen.add(br["op"])
                assert br["instruction"].startswith(f"[{br['op']}:")
                assert br["instruction"].endswith("Keep everything else in the image unchanged.")
                if br["op"] == "add":
                    canvas_truth = _truth_of(rd["canvas"])
                    assert not canvas_truth.present.get(br["node"]), "add was issued for an entity already present in the truth"
                for cj in br["candidates"]:
                    r = cj["rules"]
                    assert set(r) == {"r1_established", "r2_local_gain", "r3_no_damage", "r4_no_duplicate", "r5_audit"}
                    if cj["accept"]:
                        assert all(r[k]["ok"] for k in r)
                    assert cj["commit"]["share"] <= 1.0
            if rd["merged"]:
                assert rd["n_fail"] <= rd.get("n_fail_before", 99)
        final = tr["final"]
        assert final["n_cond"] > 0
        assert summ["all_pass"] == final["all_pass"]
        rows = [json.loads(l) for l in open(os.path.join(out, "table.jsonl"))]
        assert rows and all("cid" in r for r in rows)
        done = json.load(open(os.path.join(out, "img", plan.id, "done.json")))
        assert done["pid"] == plan.id
        summ2 = loop.run_prompt(plan)
        assert summ2["pid"] == plan.id
        print("ops seen:", sorted(ops_seen), "rounds:", len(tr["rounds"]), "fail0:", n_fail0, "final fail:", final["n_cond"] - final["n_pass"], "all_pass:", final["all_pass"])
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_parallel_branches_merge_disjoint():
    plan = lab_plan()
    conds = build_conditions(plan)
    out = tempfile.mkdtemp(prefix="dagloop_")
    try:
        ex = MockExecutor(plan, p_fix=1.0, p_collateral=0.0, p_dup=0.0, seed=11, e_present=0.5, a_ok=0.5, r_ok=0.5, p_extra=0.0, p_merge=0.0, p_absent=0.0)
        vlm = MockVLM(plan, conds, f=0.0, miss=0.0, f2=0.0, seed=11)
        cfg = LoopConfig(mode="repair", n0=1, G=2, max_rounds=8, max_tries=2, max_parallel=4, seed=2)
        loop = Loop(ex, vlm, cfg, out)
        summ = loop.run_prompt(plan)
        tr = json.load(open(os.path.join(out, "img", plan.id, "trace.json")))
        multi = [rd for rd in tr["rounds"] if len(rd["branches"]) > 1]
        assert multi, "expected at least one round with parallel branches"
        for rd in tr["rounds"]:
            if len(rd["merged"]) > 1:
                t = _truth_of(os.path.join(os.path.dirname(rd["canvas"]), f"canvas_r{rd['round']:02d}.png"))
                for br in rd["branches"]:
                    if br["target"] in rd["merged"]:
                        if br["op"] == "add":
                            assert t.present[br["node"]]
                        elif br["op"] == "fix" and br["ntype"] in ("attribute", "state", "action", "relation"):
                            assert t.ok[br["node"]]
        assert summ["all_pass"], "with a perfect editor and evaluator every condition should end up satisfied"
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    test_verify_matches_truth_without_noise()
    test_loop_repairs_and_respects_rules()
    test_parallel_branches_merge_disjoint()
    print("loop tests ok")
