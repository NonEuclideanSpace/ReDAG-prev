from .schema import CHILD_TYPES, MULTI_TYPES, box_intersects, dilate_box, union_box

VIEW_PLAN = {
    "background": (("full", 0.0), ("full", 0.0)),
    "entity": (("full", 0.0), ("crop", 0.35)),
    "attribute": (("crop", 0.25), ("full", 0.0)),
    "state": (("crop", 0.25), ("full", 0.0)),
    "action": (("crop", 0.25), ("boxed", 0.0)),
    "relation": (("crop", 0.25), ("boxed", 0.0)),
    "pair": (("crop", 0.25), ("native", 0.0)),
    "absent": (("native", 0.0), ("native", 0.0)),
    "count": (("native", 0.0), ("native", 0.0)),
    "quality": (("full", 0.0), None),
}
TEMPLATE_PLAN = {"native": ("a", "b"), "full": ("a", "b"), "crop": ("a", "b"), "boxed": ("a", "b")}


class Condition:
    __slots__ = ("cid", "kind", "node", "ntype", "inv", "question", "question_alt", "participants", "parents", "cls", "psi")

    def __init__(self, cid, kind, node, ntype, inv, question, participants, parents, cls=None, question_alt=None, psi=1.0):
        self.cid, self.kind, self.node, self.ntype, self.inv = cid, kind, node, ntype, bool(inv)
        self.question, self.question_alt = question, question_alt
        self.participants, self.parents, self.cls, self.psi = list(participants), list(parents), cls, float(psi)

    def to_dict(self):
        return dict(cid=self.cid, kind=self.kind, node=self.node, ntype=self.ntype, inv=self.inv, question=self.question, participants=self.participants, parents=self.parents, cls=self.cls, psi=self.psi)


def build_conditions(plan):
    out = []
    for v in plan.order:
        n = plan.nodes[v]
        t = n["type"]
        if t == "count":
            out.append(Condition("c:" + v, "count", v, t, True, n["question"], [], [], cls=n.get("cls"), question_alt=n.get("question_min")))
        elif t == "pair":
            out.append(Condition("p:" + v, "pair", v, t, False, n["question"], n["pair"], n["pair"]))
        elif t == "quality":
            out.append(Condition("q:" + v, "quality", v, t, True, n["question"], [], [], psi=0.5))
        elif t == "absent":
            out.append(Condition("n:" + v, "node", v, t, True, n["question"], [], []))
        else:
            out.append(Condition("n:" + v, "node", v, t, bool(n.get("inv", 0)), n["question"], plan.participants(v), plan.compose_parents(v)))
    return out


def cond_index(conds):
    return {c.cid: c for c in conds}


def region_of(plan, c, boxes, hint=None):
    """Region a condition is judged on: actual entity boxes when known, planned boxes otherwise. None means whole image."""
    t = c.ntype
    if t in ("background", "absent", "count", "quality"):
        return None
    if t == "entity":
        if hint is not None and c.node in hint:
            return list(hint[c.node])
        if c.node in boxes:
            return list(boxes[c.node])
        return list(plan.nodes[c.node]["box"])
    if t in CHILD_TYPES:
        u = c.participants[0]
        return list(boxes.get(u, plan.nodes[u]["box"]))
    if t in MULTI_TYPES or t == "pair":
        return union_box([boxes.get(u, plan.nodes[u]["box"]) for u in c.participants])
    return None


def region_touches(plan, c, boxes, mask_boxes, grow=0.1):
    """Whether a condition's region intersects any of the changed boxes. Whole-image conditions always do."""
    r = region_of(plan, c, boxes)
    if r is None:
        return True
    r = dilate_box(r, grow, plan.W, plan.H)
    return any(box_intersects(r, b) for b in mask_boxes)


def affected_conditions(plan, conds, boxes, mask_boxes, edited_nodes, unlocated=()):
    """Conditions to re-verify after a commit: region intersects the change, descendants and interference neighbours of edited nodes, counts, quality, pairs that involve them, and anything whose region is unknown."""
    aff = set()
    ed = set(edited_nodes)
    nb = set()
    for v in ed:
        nb |= set(plan.descendants(v))
        for a, b in plan.interfere_pairs():
            if a == v:
                nb.add(b)
                nb |= set(plan.descendants(b))
            elif b == v:
                nb.add(a)
                nb |= set(plan.descendants(a))
    for c in conds:
        if c.kind in ("count", "quality") or c.ntype == "absent":
            aff.add(c.cid)
        elif c.node in ed or c.node in nb or any(p in ed or p in nb for p in c.participants):
            aff.add(c.cid)
        elif any(p in unlocated for p in c.participants):
            aff.add(c.cid)
        elif region_touches(plan, c, boxes, mask_boxes):
            aff.add(c.cid)
    return aff


def is_askable(c, statuses):
    """Compose gating: a condition is asked only when all its compose parents currently pass."""
    for p in c.parents:
        st = statuses.get("n:" + p)
        if st != "pass":
            return False
    return True
