import argparse
import json
import os

from common import P0

from redag.report import run_metrics
from redag.stats import summarise_traces

ap = argparse.ArgumentParser(description="Summarise a run: progress, metrics, acceptance by op, interference, decision mix. Writes stats.json usable by --stats.")
ap.add_argument("--run", required=True)
args = ap.parse_args()

rd = args.run if os.path.isabs(args.run) else os.path.join(P0, "runs", args.run)
m = run_metrics(rd)
traces = [os.path.join(rd, "img", p, "trace.json") for p in sorted(os.listdir(os.path.join(rd, "img")))] if os.path.isdir(os.path.join(rd, "img")) else []
st = summarise_traces(traces)
st.save(os.path.join(rd, "stats.json"))
n = len(m)


def _mean(k):
    xs = [v[k] for v in m.values() if v.get(k) is not None]
    return sum(float(x) for x in xs) / len(xs) if xs else None


summary = dict(run=args.run, prompts=n, node_pass=_mean("node_pass"), all_pass=_mean("all_pass"), calls=_mean("calls"), elapsed=_mean("elapsed"), decisions={d: sum(1 for v in m.values() if v["decision"] == d) for d in set(v["decision"] for v in m.values())} if n else {}, p_op=st.p_op, h_op=st.h_op, e_type=st.e_type, c_e=st.c_e, attempts=st.n)
print(json.dumps(summary, ensure_ascii=False, indent=1))
with open(os.path.join(rd, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
