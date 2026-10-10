"""GPU-free checks of the machine helpers: jobq.py (claim, done/failed, kill at deadline, restart skips done lines),
machine_init.py (writes the host entry) and pick_tiers.py (fixed per-tier picks that keep the split)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)


def run(cmd, env):
    return subprocess.run([sys.executable] + cmd, env=env, cwd=CODE, capture_output=True, text=True)


def test_queue():
    root = tempfile.mkdtemp()
    try:
        env = dict(os.environ, P0=root, PYTHONPATH=CODE)
        os.makedirs(os.path.join(root, "queue"))
        q = os.path.join(root, "queue", "t.txt")
        with open(q, "w") as f:
            f.write("# comment\npython -c \"import os; print(os.environ['CUDA_VISIBLE_DEVICES'])\"\nfalse\nsleep 30\n")
        out = run(["scripts/jobq.py", "--queue", q, "--gpus", "3,4", "--poll", "0.5", "--kill-at", "+4"], env)
        assert out.returncode == 0, out.stderr
        st = {}
        for line in open(q + ".state.jsonl"):
            r = json.loads(line)
            st[r["line"]] = r["status"]
        assert st == {1: "done", 2: "failed", 3: "killed"}, st
        logs = [fn for fn in os.listdir(os.path.join(root, "logs")) if fn.startswith("queue_t_L1_")]
        assert logs and open(os.path.join(root, "logs", logs[0])).read().strip().endswith(("3", "4"))
        # restart: the done line is skipped, failed and killed lines run again
        with open(q, "a") as f:
            f.write("python -c \"print('x')\"\n")
        out = run(["scripts/jobq.py", "--queue", q, "--gpus", "0,1", "--poll", "0.5", "--kill-at", "+3"], env)
        assert out.returncode == 0, out.stderr
        started = [json.loads(l) for l in open(q + ".state.jsonl")]
        second = [r for r in started if r["status"] == "running"][3:]
        assert sorted(r["line"] for r in second) == [2, 3, 4], [r["line"] for r in second]

        out = run(["scripts/machine_init.py", "--host", "h1", "--gpus", "0", "1", "2", "--disabled", "0", "--note", "n"], env)
        assert out.returncode == 0, out.stderr
        m = json.load(open(os.path.join(root, "configs", "machine.json")))
        assert m["hosts"]["h1"]["gpus"] == [1, 2] and m["hosts"]["h1"]["disabled"] == [0]

        os.makedirs(os.path.join(root, "data"))
        tiers = {"tiers": {"<=15": [dict(set="s", pid=f"p{i}", tag="t", conds=10, split="test" if i % 3 == 0 else "dev") for i in range(30)], ">50": [dict(set="s", pid=f"q{i}", tag="t", conds=60, split="test" if i < 5 else "dev") for i in range(40)]}}
        json.dump(tiers, open(os.path.join(root, "data", "tiers.json"), "w"))
        out = run(["scripts/pick_tiers.py", "--per-tier", "12"], env)
        assert out.returncode == 0, out.stderr
        s = json.load(open(os.path.join(root, "data", "tier_picks", "summary.json")))["tiers"]
        assert s["<=15"]["n"] == 12 and s["<=15"]["test"] == 4 and s[">50"]["n"] == 12 and s[">50"]["test"] == 4, s
        out = run(["scripts/pick_tiers.py", "--per-tier", "8", "--test-frac", "1.0", "--out", os.path.join(root, "data", "tp2")], env)
        s2 = json.load(open(os.path.join(root, "data", "tp2", "summary.json")))["tiers"]
        assert s2["<=15"]["test"] == 8 and s2["<=15"]["dev"] == 0 and s2[">50"]["test"] == 5 and s2[">50"]["dev"] == 3, s2
        lines = [l.split() for l in open(os.path.join(root, "data", "tier_picks", "gt50.txt")) if not l.startswith("#")]
        assert len(lines) == 12 and all(len(x) == 4 for x in lines)
        out = run(["scripts/pick_tiers.py", "--per-tier", "12", "--min-test", "6"], env)
        assert out.returncode != 0 and "too few test" in (out.stdout + out.stderr)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_queue()
    print("queue/machine/pick tests ok")
