import time

from .schema import box_intersects, dilate_box

OP_PRIORITY = {"add": 0, "split": 1, "remove": 2, "fix": 3}


def frontier(plan, conds, col, targets, tried, max_tries):
    """Targets whose support parents pass and that still have attempts left."""
    out = []
    for t in targets:
        if tried.get(t["key"], 0) >= max_tries:
            continue
        v = t["node"]
        if plan.typ(v) == "entity" and any(not col.passes("n:" + u) for u in plan.support_parents(v) if u != "bg"):
            continue
        out.append(t)
    return out


def conflicts(plan, a, b, grow=0.25, stats=None):
    if a["node"] == b["node"]:
        return True
    pa, pb = set(a.get("participants", [])), set(b.get("participants", []))
    if pa & pb:
        return True
    if plan.has_path(a["node"], b["node"]):
        return True
    for u in pa:
        for w in pb:
            if plan.interferes(u, w) or plan.has_path(u, w):
                return True
    ra, rb = a.get("region"), b.get("region")
    if ra is None or rb is None:
        return True
    return box_intersects(dilate_box(ra, grow, plan.W, plan.H), dilate_box(rb, grow, plan.W, plan.H))


def priority_key(plan, t, stats=None):
    h = 0.0
    if stats is not None:
        h = -float(stats.interference_of(t.get("op", "fix"), t.get("ntype", "")))
    r = t.get("region")
    area = -((r[2] - r[0]) * (r[3] - r[1])) if r else 0.0
    return (h, OP_PRIORITY.get(t.get("op", "fix"), 9), area, plan.order.index(t["node"]) if t["node"] in plan.order else 999)


def independent_groups(plan, targets, max_parallel=4, grow=0.25, stats=None):
    """Greedy grouping of targets into sets that can be edited in parallel from the same canvas."""
    order = sorted(targets, key=lambda t: priority_key(plan, t, stats))
    groups = []
    for t in order:
        placed = False
        for g in groups:
            if len(g) < max_parallel and not any(conflicts(plan, t, u, grow, stats) for u in g):
                g.append(t)
                placed = True
                break
        if not placed:
            groups.append([t])
    return groups


def repair_wins(q, m, p, c_e=2.0, m_f=0.0):
    """Repair-vs-redraw criterion: p / (c_e (m + m_f)) > q / (1 - q)."""
    if q >= 1.0:
        return False
    if q <= 0.0:
        return True
    lhs = p / max(1e-9, c_e * (m + m_f))
    return lhs > q / (1.0 - q)


class Budget:
    def __init__(self, calls_max=None, seconds_max=None, c_edit=1.0):
        self.calls_max = calls_max
        self.seconds_max = seconds_max
        self.c_edit = float(c_edit)
        self.calls = 0.0
        self.drafts = 0
        self.edits = 0
        self.t0 = time.time()
        self.sec_gen = 0.0
        self.sec_eval = 0.0

    def add_drafts(self, n, sec=0.0):
        self.drafts += n
        self.calls += n
        self.sec_gen += sec

    def add_edits(self, n, sec=0.0):
        self.edits += n
        self.calls += n * self.c_edit
        self.sec_gen += sec

    def add_eval(self, sec):
        self.sec_eval += sec

    def elapsed(self):
        return time.time() - self.t0

    def exhausted(self):
        if self.calls_max is not None and self.calls >= self.calls_max:
            return True
        if self.seconds_max is not None and self.elapsed() >= self.seconds_max:
            return True
        return False

    def room(self, n_calls):
        return self.calls_max is None or self.calls + n_calls <= self.calls_max + 1e-9

    def to_dict(self):
        return dict(calls=self.calls, drafts=self.drafts, edits=self.edits, elapsed=round(self.elapsed(), 1), sec_gen=round(self.sec_gen, 1), sec_eval=round(self.sec_eval, 1), calls_max=self.calls_max, seconds_max=self.seconds_max)
