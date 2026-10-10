import json
import os

from .schedule import repair_wins


class Stats:
    """Scheduling statistics summarised from traces: per-op acceptance rates, interference by op and node type,
    establishment rates per node type. Used for priorities, protection and the repair-vs-redraw decision."""

    def __init__(self, d=None):
        d = d or {}
        self.p_op = d.get("p_op", {})
        self.h_op = d.get("h_op", {})
        self.e_type = d.get("e_type", {})
        self.c_e = float(d.get("c_e", 2.0))
        self.n = d.get("n", {})

    @staticmethod
    def load(path):
        if path and os.path.exists(path):
            return Stats(json.load(open(path, encoding="utf-8")))
        return Stats()

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(p_op=self.p_op, h_op=self.h_op, e_type=self.e_type, c_e=self.c_e, n=self.n), f, indent=1)

    def p_of(self, op, default=0.2):
        return float(self.p_op.get(op, default))

    def interference_of(self, op, ntype, default=0.0):
        return float(self.h_op.get(f"{op}:{ntype}", self.h_op.get(op, default)))

    def decide(self, q_hat, m_hat, ops, m_f=0.0):
        p = sum(self.p_of(o) for o in ops) / max(1, len(ops)) if ops else self.p_of("add")
        return repair_wins(q_hat, m_hat, p, self.c_e, m_f)


def summarise_traces(trace_paths):
    """Build Stats from run traces: acceptance per op, collateral damage per op and target type, draft pass rates per type."""
    acc, att = {}, {}
    harm, harm_n = {}, {}
    e_pass, e_n = {}, {}
    sec_draft, n_draft, sec_edit, n_edit = 0.0, 0, 0.0, 0
    for p in trace_paths:
        try:
            tr = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        s1 = tr.get("stage1", {})
        for col in s1.get("columns", []):
            for cid, st in col.get("status", {}).items():
                t = col.get("ntype", {}).get(cid)
                if t is None:
                    continue
                e_n[t] = e_n.get(t, 0) + 1
                e_pass[t] = e_pass.get(t, 0) + (1 if st == "pass" else 0)
        sec_draft += float(s1.get("sec_draw", 0.0))
        n_draft += int(s1.get("n_drafts", 0))
        for rd in tr.get("rounds", []):
            for br in rd.get("branches", []):
                op, nt = br.get("op"), br.get("ntype")
                att[op] = att.get(op, 0) + 1
                acc[op] = acc.get(op, 0) + (1 if br.get("accepted") else 0)
                sec_edit += float(br.get("sec_edit", 0.0)) + float(br.get("sec_eval", 0.0))
                n_edit += 1
                for cj in br.get("candidates", []):
                    w = cj.get("rules", {}).get("r3_no_damage", {}).get("worst", 0.0)
                    k = f"{op}:{nt}"
                    harm_n[k] = harm_n.get(k, 0) + 1
                    harm[k] = harm.get(k, 0.0) + max(0.0, -float(w))
    st = Stats()
    st.p_op = {o: acc[o] / att[o] for o in att if att[o]}
    st.h_op = {k: harm[k] / harm_n[k] for k in harm if harm_n[k]}
    st.e_type = {t: e_pass[t] / e_n[t] for t in e_n if e_n[t]}
    if n_draft and n_edit:
        st.c_e = max(0.5, (sec_edit / n_edit) / max(1e-6, sec_draft / n_draft))
    st.n = dict(attempts=att, drafts=n_draft, edits=n_edit)
    return st
