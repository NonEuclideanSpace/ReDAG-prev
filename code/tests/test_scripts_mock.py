"""Runs the command-line scripts end to end with the mock executor inside a temporary P0 tree:
plan (with a stubbed planner), run_loop (mock), status, make_labels, compare. Catches import and argument errors
that unit tests of the library would miss."""
import glob
import json
import time
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from fixtures import lab_plan

from redag.schema import strip_generated


def run(cmd, env, cwd):
    r = subprocess.run([sys.executable] + cmd, env=env, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError("script failed: " + " ".join(cmd))
    return r.stdout


def test_scripts():
    root = tempfile.mkdtemp(prefix="dagp0_")
    try:
        os.makedirs(os.path.join(root, "data", "sets"))
        os.makedirs(os.path.join(root, "configs"))
        shutil.copy(os.path.join(CODE, "configs", "loop.default.json"), os.path.join(root, "configs", "loop.default.json"))
        plan = lab_plan()
        pdir = os.path.join(root, "data", "plans", "lab", "stub")
        os.makedirs(pdir)
        for i in range(3):
            d = strip_generated(plan).to_dict()
            d["id"] = f"lab_{i:04d}"
            with open(os.path.join(pdir, d["id"] + ".json"), "w") as f:
                json.dump(d, f)
        with open(os.path.join(root, "data", "sets", "lab.jsonl"), "w") as f:
            for i in range(3):
                f.write(json.dumps(dict(id=f"lab_{i:04d}", prompt="chemistry lab scene")) + "\n")
        env = dict(os.environ, P0=root, PYTHONPATH=CODE)
        sc = os.path.join(CODE, "scripts")
        out = run([os.path.join(sc, "run_loop.py"), "--set", "lab", "--tag", "stub", "--run", "mockA", "--executor", "mock", "--mode", "repair", "--n0", "2", "--skip-preflight", "--seed", "1"], env, CODE)
        assert "DONE" in out
        out = run([os.path.join(sc, "run_loop.py"), "--set", "lab", "--tag", "stub", "--run", "mockB", "--executor", "mock", "--mode", "draw", "--n0", "2", "--n-max", "4", "--skip-preflight", "--seed", "2"], env, CODE)
        assert "DONE" in out
        out = run([os.path.join(sc, "status.py"), "--run", "mockA"], env, CODE)
        assert "node_pass" in out
        assert os.path.exists(os.path.join(root, "runs", "mockA", "stats.json"))
        out = run([os.path.join(sc, "make_labels.py"), "--runs", "mockA", "--set", "lab", "--tag", "stub", "--name", "lab_seed", "--max-cond", "40", "--max-edit", "10", "--max-diag", "10"], env, CODE)
        assert os.path.exists(os.path.join(root, "labels", "lab_seed", "sheet.html"))
        items = [json.loads(l) for l in open(os.path.join(root, "labels", "lab_seed", "items.jsonl"))]
        assert any(it["kind"] == "cond" and it.get("question") for it in items)
        out = run([os.path.join(sc, "draw.py"), "--set", "lab", "--tag", "stub", "--run", "drafts1", "--n", "3", "--executor", "mock", "--skip-preflight"], env, CODE)
        assert "DONE" in out
        assert not glob.glob(os.path.join(root, "runs", "drafts1", "img", "*", "claim.json")), "claims must be removed after drawing"
        # a live claim by another process is respected: that prompt is skipped, the rest is drawn
        d2 = os.path.join(root, "runs", "drafts2", "img", "lab_0000")
        os.makedirs(d2)
        json.dump(dict(pid=os.getpid(), t=time.time()), open(os.path.join(d2, "claim.json"), "w"))
        out = run([os.path.join(sc, "draw.py"), "--set", "lab", "--tag", "stub", "--run", "drafts2", "--n", "2", "--executor", "mock", "--skip-preflight"], env, CODE)
        assert "DONE" in out and not os.path.exists(os.path.join(d2, "drafts.json")) and os.path.exists(os.path.join(root, "runs", "drafts2", "img", "lab_0001", "drafts.json")), out[-300:]
        # --picks: prompts of several sets (here one) drawn in one process, plans resolved per prompt
        pk2 = os.path.join(root, "picks_draw.txt")
        with open(pk2, "w") as f:
            f.write("lab_0001 lab stub test\nlab_0002 lab stub test\nnot_a_pid lab stub test\n")
        out = run([os.path.join(sc, "draw.py"), "--picks", pk2, "--run", "drafts3", "--n", "2", "--executor", "mock", "--skip-preflight"], env, CODE)
        assert "DONE" in out and sorted(os.listdir(os.path.join(root, "runs", "drafts3", "img"))) == ["lab_0001", "lab_0002"], out[-300:]
        instr = os.path.join(root, "instr.jsonl")
        with open(instr, "w") as f:
            f.write(json.dumps(dict(pid="lab_0000", canvas=os.path.join(root, "runs", "drafts1", "img", "lab_0000", "draft_00.png"), instruction="[add:researcher_c] add the researcher at the fume hood. Keep everything else in the image unchanged.", op="add", node="researcher_c", ntype="entity", region=[80, 160, 300, 660])) + "\n")
        out = run([os.path.join(sc, "edit_bench.py"), "--instructions", instr, "--run", "edits1", "--G", "2", "--executor", "mock", "--plan", os.path.join(pdir, "lab_0000.json"), "--skip-preflight"], env, CODE)
        assert "DONE" in out and os.path.exists(os.path.join(root, "runs", "edits1", "edits.jsonl"))
        instr2 = os.path.join(root, "instr2.jsonl")
        out = run([os.path.join(sc, "make_instructions.py"), "--drafts-run", "drafts1", "--set", "lab", "--tag", "stub", "--out", instr2, "--per-draft", "3", "--drafts-per-prompt", "1"], env, CODE)
        rows2 = [json.loads(l) for l in open(instr2)]
        assert rows2 and {r["op"] for r in rows2} <= {"add", "fix", "remove"} and all(r["instruction"].endswith("unchanged.") for r in rows2)
        assert all(r["region"] is None or (len(r["region"]) == 4) for r in rows2)
        out = run([os.path.join(sc, "edit_bench.py"), "--instructions", instr2, "--run", "edits2", "--G", "2", "--executor", "mock", "--plan", os.path.join(pdir, "lab_0000.json"), "--skip-preflight", "--shard", "0/2"], env, CODE)
        assert "DONE" in out and os.path.exists(os.path.join(root, "runs", "edits2", "edits.jsonl"))
        instr3 = os.path.join(root, "instr3.jsonl")
        out = run([os.path.join(sc, "make_instructions.py"), "--from-runs", "mockA", "--set", "lab", "--tag", "stub", "--out", instr3], env, CODE)
        rows3 = [json.loads(l) for l in open(instr3)]
        assert rows3 and all(r["source"].startswith("loop:mockA") and os.path.exists(r["canvas"]) and r["instruction"] for r in rows3)
        out = run([os.path.join(sc, "make_labels.py"), "--drafts-runs", "drafts1", "--edits-runs", "edits1", "--set", "lab", "--tag", "stub", "--name", "lab_seed2", "--max-cond", "30", "--max-edit", "5"], env, CODE)
        out = run([os.path.join(sc, "make_sheet.py"), "--name", "lab_seed2", "--n-cond", "5", "--n-edit", "2", "--embed"], env, CODE)
        assert "data:image/jpeg;base64," in open(os.path.join(root, "labels", "lab_seed2", "sheet_human.html")).read()
        items2 = [json.loads(l) for l in open(os.path.join(root, "labels", "lab_seed2", "items.jsonl"))]
        assert any(it["kind"] == "cond" and it["question"] for it in items2) and any(it["kind"] == "edit" for it in items2)
        # the same from a picks file (pid set tag split): plans resolved per prompt, unknown prompts skipped
        pk = os.path.join(root, "picks.txt")
        with open(pk, "w") as f:
            f.write("# picks\nlab_0000 lab stub dev\nlab_0001 lab stub test\n")
        out = run([os.path.join(sc, "make_labels.py"), "--drafts-runs", "drafts1", "--picks", pk, "--name", "lab_seed3", "--max-cond", "30"], env, CODE)
        items3 = [json.loads(l) for l in open(os.path.join(root, "labels", "lab_seed3", "items.jsonl"))]
        assert items3 and all(it["pid"] in ("lab_0000", "lab_0001") for it in items3), out[-300:]
        # merge_ref: human first, then model files only where they agree
        ld = os.path.join(root, "labels", "lab_seed2")
        ids = [it["id"] for it in items2 if it["kind"] == "cond"][:4]
        with open(os.path.join(ld, "h.jsonl"), "w") as f:
            f.write(json.dumps(dict(id=ids[0], label="n")) + "\n")
        with open(os.path.join(ld, "m1.jsonl"), "w") as f:
            f.write("".join(json.dumps(dict(id=i, label="y")) + "\n" for i in ids))
        with open(os.path.join(ld, "m2.jsonl"), "w") as f:
            f.write("".join(json.dumps(dict(id=i, label="y" if k < 3 else "u")) + "\n" for k, i in enumerate(ids)))
        out = run([os.path.join(sc, "merge_ref.py"), "--name", "lab_seed2", "--human", "h.jsonl", "--models", "m1.jsonl", "m2.jsonl", "--out", "ref.jsonl"], env, CODE)
        ref = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(os.path.join(ld, "ref.jsonl"))}
        assert ref == {ids[0]: "n", ids[1]: "y", ids[2]: "y"}, (ref, out[-300:])
        out = run([os.path.join(sc, "compare.py"), "--a", "mockA", "--b", "mockB"], env, CODE)
        assert "node_pass" in out
        ext = os.path.join(root, "runs", "extbase", "img")
        for pid in ("lab_0000", "lab_0001"):
            os.makedirs(os.path.join(ext, pid), exist_ok=True)
            shutil.copy(os.path.join(root, "runs", "drafts1", "img", pid, "draft_00.png"), os.path.join(ext, pid, "final.png"))
            with open(os.path.join(ext, pid, "trace.json"), "w") as f:
                json.dump(dict(final=dict(image="final.png", budget=dict(calls=4, elapsed=80.0))), f)
        out = run([os.path.join(sc, "compare.py"), "--a", "mockA", "--b", "extbase"], env, CODE)
        res = json.loads(out[: out.index("a sheet:")]) if "a sheet:" in out else None
        assert res and res["calls"]["n"] == 2 and res["node_pass"]["n"] == 0, "external baseline runs need only final.png and a minimal trace"
        out = run([os.path.join(sc, "status.py"), "--run", "extbase"], env, CODE)
        assert '"node_pass": null' in out and '"calls": 4.0' in out
        out = run([os.path.join(sc, "run_loop.py"), "--set", "lab", "--tag", "stub", "--run", "mockD", "--executor", "mock", "--mode", "draw", "--n0", "2", "--n-max", "2", "--drafts-from", "drafts1", "--skip-preflight"], env, CODE)
        assert "DONE" in out
        tr = json.load(open(os.path.join(root, "runs", "mockD", "img", "lab_0000", "trace.json")))
        assert tr["final"]["executor"]["name"] == "predrawn" and tr["final"]["executor"]["served"] == 2
        from PIL import Image

        a = Image.open(os.path.join(root, "runs", "drafts1", "img", "lab_0001", "draft_00.png"))
        b = Image.open(os.path.join(root, "runs", "mockD", "img", "lab_0001", "draft_00.png"))
        assert a.tobytes() == b.tobytes() and a.info.get("truth") == b.info.get("truth"), "pre-drawn draft must be served unchanged and keyed by pid"
        out = run([os.path.join(sc, "run_loop.py"), "--set", "lab", "--tag", "stub", "--run", "mockC", "--executor", "mock", "--mode", "adaptive", "--n0", "2", "--n-max", "4", "--stats", os.path.join(root, "runs", "mockA", "stats.json"), "--skip-preflight"], env, CODE)
        assert "DONE" in out
        print(out.strip().splitlines()[-2])
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_scripts()
    print("script tests ok")
