import json


class Rec:
    __slots__ = ("cid", "s1", "z1", "v1", "s2", "z2", "v2", "s", "status", "region")

    def __init__(self, cid):
        self.cid = cid
        self.s1 = self.z1 = self.s2 = self.z2 = None
        self.v1 = self.v2 = None
        self.s = None
        self.status = "unasked"
        self.region = None

    def to_dict(self):
        return dict(cid=self.cid, s1=self.s1, z1=self.z1, v1=self.v1, s2=self.s2, z2=self.z2, v2=self.v2, s=self.s, status=self.status, region=self.region)

    @staticmethod
    def from_dict(d):
        r = Rec(d["cid"])
        for k in ("s1", "z1", "v1", "s2", "z2", "v2", "s", "status", "region"):
            setattr(r, k, d.get(k))
        return r

    def copy_as_carried(self):
        r = Rec(self.cid)
        r.s1, r.z1, r.v1, r.s2, r.z2, r.v2, r.s, r.region = self.s1, self.z1, self.v1, self.s2, self.z2, self.v2, self.s, self.region
        r.status = "carried-" + self.status if not self.status.startswith("carried-") else self.status
        return r


def base_status(st):
    return st[8:] if st and st.startswith("carried-") else st


class Column:
    """One verified image: a condition id -> Rec map plus the entity boxes located on that image."""

    def __init__(self, recs=None, boxes=None, image=None, tag=""):
        self.recs = dict(recs or {})
        self.boxes = dict(boxes or {})
        self.unlocated = set()
        self.image = image
        self.tag = tag
        self.cost = dict(questions=0, grounds=0)

    def status(self, cid):
        r = self.recs.get(cid)
        return base_status(r.status) if r else "unasked"

    def statuses(self):
        return {cid: base_status(r.status) for cid, r in self.recs.items()}

    def score(self, cid, default=0.0):
        r = self.recs.get(cid)
        return default if r is None or r.s is None else float(r.s)

    def passes(self, cid):
        return self.status(cid) == "pass"

    def fails(self, cid):
        return self.status(cid) == "fail"

    def failed(self, conds, kinds=None):
        return [c for c in conds if self.fails(c.cid) and (kinds is None or c.kind in kinds)]

    def all_pass(self, conds, kinds=("node", "count", "pair")):
        return all(self.status(c.cid) == "pass" for c in conds if c.kind in kinds)

    def phi(self, conds, lam_q=0.5):
        tot, q, nq = 0.0, 0.0, 0
        for c in conds:
            st = self.status(c.cid)
            if c.kind == "quality":
                if st in ("pass", "fail"):
                    q += self.score(c.cid)
                    nq += 1
                continue
            if st in ("pass", "fail"):
                tot += c.psi * self.score(c.cid)
        return tot + (lam_q * q / nq if nq else 0.0)

    def n_pass(self, conds, kinds=("node",)):
        return sum(1 for c in conds if c.kind in kinds and self.status(c.cid) == "pass")

    def n_scored(self, conds, kinds=("node",)):
        return sum(1 for c in conds if c.kind in kinds and self.status(c.cid) in ("pass", "fail"))

    def to_dict(self):
        return dict(tag=self.tag, image=self.image, boxes=self.boxes, unlocated=sorted(self.unlocated), cost=self.cost, recs={k: r.to_dict() for k, r in self.recs.items()})

    @staticmethod
    def from_dict(d):
        c = Column({k: Rec.from_dict(v) for k, v in d.get("recs", {}).items()}, d.get("boxes", {}), d.get("image"), d.get("tag", ""))
        c.unlocated = set(d.get("unlocated", []))
        c.cost = d.get("cost", c.cost)
        return c

    def rows(self, pid, round_idx, cand):
        for cid, r in self.recs.items():
            yield dict(pid=pid, round=round_idx, cand=cand, tag=self.tag, image=self.image, **r.to_dict())


def write_rows(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
