import json
import re

PLANNER_TYPES = ("background", "entity", "attribute", "state", "action", "relation", "absent")
GENERATED_TYPES = ("count", "pair", "quality")
TYPES = PLANNER_TYPES + GENERATED_TYPES
CHILD_TYPES = ("attribute", "state")
MULTI_TYPES = ("action", "relation")
EDGE_KINDS = ("compose", "support", "interfere")
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
Q_VIS = "Is this image too dark, too blurry or too washed out to clearly see its content?"
Q_ART = "Does this image contain obvious visual artifacts, such as distorted faces or bodies, extra or missing limbs, or smeared and half-rendered regions?"
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")
COORD_RE = re.compile(r"\b\d{2,4}\s*[,x×]\s*(?!(19|20)\d\d\b)\d{2,4}\b|\b\d+\s*(px|pixels?)\b|\bbbox|\bcoordinate", re.I)


def box_ok(b, W, H, min_side=0):
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in b]
    except (TypeError, ValueError):
        return False
    return 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H and (x1 - x0) >= min_side and (y1 - y0) >= min_side


def union_box(boxes):
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = box_area(a) + box_area(b) - inter
    return inter / ua if ua > 0 else 0.0


def box_intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def dilate_box(b, frac, W, H):
    gx, gy = frac * (b[2] - b[0]), frac * (b[3] - b[1])
    return [max(0.0, b[0] - gx), max(0.0, b[1] - gy), min(float(W), b[2] + gx), min(float(H), b[3] + gy)]


class Plan:
    def __init__(self, d):
        self.d = d
        self.id = d.get("id") or d.get("prompt_id")
        self.prompt = d.get("prompt", "")
        self.W, self.H = [int(v) for v in d.get("canvas", [1024, 1024])]
        self.nodes = {}
        self.order = []
        for n in d.get("nodes", []):
            self.nodes[n["id"]] = dict(n)
            self.order.append(n["id"])
        self.edges = [dict(e) for e in d.get("edges", [])]
        self.meta = d.get("meta", {})

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            return Plan(json.load(f))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    def to_dict(self):
        return dict(self.d, id=self.id, prompt=self.prompt, canvas=[self.W, self.H], nodes=[self.nodes[v] for v in self.order], edges=self.edges, meta=self.meta)

    def typ(self, v):
        return self.nodes[v]["type"]

    def entities(self):
        return [v for v in self.order if self.typ(v) == "entity"]

    def of(self, v):
        n = self.nodes[v]
        o = n.get("of")
        if o is None:
            return []
        return [o] if isinstance(o, str) else list(o)

    def participants(self, v):
        t = self.typ(v)
        if t == "entity":
            return [v]
        if t in CHILD_TYPES or t in MULTI_TYPES:
            return self.of(v)
        if t == "pair":
            return list(self.nodes[v]["pair"])
        return []

    def compose_parents(self, v):
        t = self.typ(v)
        if t in CHILD_TYPES or t in MULTI_TYPES or t == "pair":
            return self.participants(v)
        return []

    def support_parents(self, v):
        return [e["from"] for e in self.edges if e.get("kind", "support") == "support" and e["to"] == v]

    def parents(self, v):
        out = []
        for u in self.compose_parents(v) + self.support_parents(v):
            if u not in out and u != v:
                out.append(u)
        return out

    def children(self, v):
        out = []
        for w in self.order:
            if v in self.compose_parents(w) and w not in out:
                out.append(w)
        for e in self.edges:
            if e.get("kind", "support") == "support" and e["from"] == v and e["to"] not in out:
                out.append(e["to"])
        return out

    def descendants(self, v):
        seen, stack = [], list(self.children(v))
        while stack:
            w = stack.pop()
            if w in seen or w == v:
                continue
            seen.append(w)
            stack.extend(self.children(w))
        return seen

    def ancestors(self, v):
        seen, stack = [], list(self.parents(v))
        while stack:
            w = stack.pop()
            if w in seen or w == v:
                continue
            seen.append(w)
            stack.extend(self.parents(w))
        return seen

    def has_path(self, a, b):
        return b in self.descendants(a) or a in self.descendants(b)

    def interfere_pairs(self):
        out = []
        for e in self.edges:
            if e.get("kind") == "interfere":
                out.append((e["from"], e["to"]))
        return out

    def interferes(self, a, b):
        for u, w in self.interfere_pairs():
            if (u, w) == (a, b) or (u, w) == (b, a):
                return True
        return False

    def same_class(self, v):
        c = self.nodes[v].get("class")
        if not c or self.typ(v) != "entity":
            return []
        return [u for u in self.entities() if u != v and self.nodes[u].get("class") == c]

    def class_count(self, c):
        return int(sum(int(self.nodes[u].get("count", 1) or 1) for u in self.entities() if self.nodes[u].get("class") == c))

    def classes(self):
        out = []
        for u in self.entities():
            c = self.nodes[u].get("class")
            if c and c not in out:
                out.append(c)
        return out

    def class_count_exact(self, c):
        """False when the description gives no number for some entity of this class ("several tomatoes"): the class then
        has a lower bound only, and no "exactly N" count node is generated for it."""
        return all(self.nodes[u].get("count_exact", True) is not False for u in self.entities() if self.nodes[u].get("class") == c)

    def levels(self):
        lev, state = {}, {}

        def visit(v):
            if v in lev:
                return lev[v]
            if state.get(v) == 1:
                raise ValueError(f"edges form a cycle at {v}")
            state[v] = 1
            ps = self.parents(v)
            l = 1 + max([visit(u) for u in ps] + [0])
            state[v] = 2
            lev[v] = l
            return l

        for v in self.order:
            visit(v)
        return lev

    def planned_box(self, v):
        n = self.nodes[v]
        t = n["type"]
        if t in ("background", "absent", "count", "quality"):
            return None
        if t == "entity":
            return list(n["box"])
        if t in CHILD_TYPES:
            return list(self.nodes[self.of(v)[0]]["box"])
        if t in MULTI_TYPES or t == "pair":
            return union_box([self.nodes[u]["box"] for u in self.participants(v)])
        return None

    def validate(self, min_side=48, max_nodes=64):
        errs, warns = [], []
        W, H = self.W, self.H
        ids = self.order
        if len(ids) != len(set(ids)):
            errs.append("duplicate node ids")
        if not ids or ids[0] != "bg":
            errs.append('first node must have id "bg"')
        if len(ids) > max_nodes + 1:
            errs.append(f"too many nodes ({len(ids) - 1} > {max_nodes})")
        ents = set(self.entities())
        for v in ids:
            n = self.nodes[v]
            t = n.get("type")
            if not ID_RE.match(str(v)):
                errs.append(f"node {v}: id must be snake_case ascii")
            if t not in PLANNER_TYPES and t not in GENERATED_TYPES:
                errs.append(f"node {v}: bad type {t}")
                continue
            if v == "bg" and t != "background":
                errs.append("bg must have type background")
            if v != "bg" and t == "background":
                errs.append(f"node {v}: only bg may be background")
            for k in ("desc", "question"):
                if not str(n.get(k, "")).strip():
                    errs.append(f"node {v}: missing {k}")
            q = str(n.get("question", "")).strip()
            if q and not q.endswith("?"):
                errs.append(f"node {v}: question must end with ?")
            if COORD_RE.search(q) or COORD_RE.search(str(n.get("instr", ""))):
                errs.append(f"node {v}: no pixel coordinates in question or instr")
            if t in ("entity", "background"):
                if not box_ok(n.get("box"), W, H, 0 if t == "background" else min_side):
                    errs.append(f"node {v}: box {n.get('box')} invalid for canvas {W}x{H} (min side {min_side})")
            if t == "entity":
                if not str(n.get("class", "")).strip():
                    errs.append(f"node {v}: entity needs class")
                try:
                    c = int(n.get("count", 1) or 1)
                    if c < 1 or c > 12:
                        errs.append(f"node {v}: count out of range")
                except (TypeError, ValueError):
                    errs.append(f"node {v}: count must be int")
                if "count_exact" in n and not isinstance(n["count_exact"], bool):
                    errs.append(f"node {v}: count_exact must be true or false")
                if len(q.split()) > 16:
                    warns.append(f"node {v}: entity question is long ({len(q.split())} words); should only ask presence + identifying look")
                if re.search(r"\b(while|and (is|are) (holding|wearing|looking|pointing|sitting|standing)|, (holding|wearing|looking|pointing))\b", q, re.I):
                    warns.append(f"node {v}: entity question seems to bundle an action or attribute")
            if t in CHILD_TYPES:
                o = self.of(v)
                if len(o) != 1 or o[0] not in ents:
                    errs.append(f"node {v}: {t} needs of = one entity id")
            if t in MULTI_TYPES:
                o = self.of(v)
                if len(o) < 2 or len(set(o)) != len(o) or any(u not in ents for u in o):
                    errs.append(f"node {v}: {t} needs of = two or more distinct entity ids")
            if t == "absent" and n.get("of"):
                errs.append(f"node {v}: absent has no of")
        seen = set()
        for e in self.edges:
            k = e.get("kind", "support")
            if k not in EDGE_KINDS:
                errs.append(f"edge {e}: bad kind")
                continue
            a, b = e.get("from"), e.get("to")
            if a not in self.nodes or b not in self.nodes or a == b:
                errs.append(f"edge {a}->{b}: unknown node or self loop")
                continue
            if k == "support" and (self.typ(b) != "entity" or self.typ(a) not in ("entity", "background")):
                errs.append(f"edge {a}->{b}: support edges go from entity or bg to entity")
            if k == "interfere" and (self.typ(a) != "entity" or self.typ(b) != "entity"):
                errs.append(f"edge {a}->{b}: interfere edges connect entities")
            key = (k, a, b) if k != "interfere" else (k,) + tuple(sorted((a, b)))
            if key in seen:
                errs.append(f"edge {a}->{b}: duplicate")
            seen.add(key)
        if not errs:
            try:
                self.levels()
            except ValueError as ex:
                errs.append(str(ex))
        return errs, warns


def add_generated_nodes(plan):
    for c in plan.classes():
        nid = "cnt_" + re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
        if nid in plan.nodes or not plan.class_count_exact(c):
            continue
        n = plan.class_count(c)
        plural = None
        for u in plan.entities():
            if plan.nodes[u].get("class") == c and plan.nodes[u].get("plural"):
                plural = plan.nodes[u]["plural"]
                break
        plural = plural or (c if c.endswith("s") else c + "s")
        k = NUM.get(n + 1, str(n + 1))
        plan.nodes[nid] = dict(id=nid, type="count", cls=c, n=n, plural=plural, desc=f"exactly {NUM.get(n, str(n))} {c if n == 1 else plural}", question=f"Are there {k} or more {plural} in this image?", question_min=f"Are there at least {NUM.get(n, str(n))} {plural if n > 1 else c} in this image?", inv=1)
        plan.order.append(nid)
    ents = plan.entities()
    for i, a in enumerate(ents):
        for b in ents[i + 1:]:
            if plan.nodes[a].get("class") and plan.nodes[a].get("class") == plan.nodes[b].get("class") and int(plan.nodes[a].get("count", 1) or 1) == 1 and int(plan.nodes[b].get("count", 1) or 1) == 1:
                pid = f"pair_{a}__{b}"
                if pid in plan.nodes:
                    continue
                da, db = plan.nodes[a]["desc"].strip().rstrip("."), plan.nodes[b]["desc"].strip().rstrip(".")
                plan.nodes[pid] = dict(id=pid, type="pair", pair=[a, b], desc=f"{da} and {db} are two different individuals", question=f"Are the following two different individuals, not the same one: (1) {da}; (2) {db}?")
                plan.order.append(pid)
                if not plan.interferes(a, b):
                    plan.edges.append(dict({"from": a, "to": b, "kind": "interfere", "why": "same class"}))
    for qid, q in (("q_vis", Q_VIS), ("q_art", Q_ART)):
        if qid not in plan.nodes:
            plan.nodes[qid] = dict(id=qid, type="quality", desc="image quality", question=q, inv=1)
            plan.order.append(qid)
    return plan


def strip_generated(plan):
    keep = [v for v in plan.order if plan.typ(v) in PLANNER_TYPES]
    d = plan.to_dict()
    d["nodes"] = [plan.nodes[v] for v in keep]
    d["edges"] = [e for e in plan.edges if e.get("kind") != "interfere" or e.get("why") != "same class"]
    return Plan(d)
