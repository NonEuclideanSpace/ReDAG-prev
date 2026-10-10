"""Mock executor and evaluator for testing the loop without GPUs.

The executor renders a scene from a hidden ground truth (which nodes are present or correct), tags images with the
truth, and edits by applying typed operations with configurable success, collateral-damage and duplication rates. The
evaluator answers questions from the truth with configurable error rates. Pixel changes follow the truth, so region
commit, acceptance and selective re-verification are exercised for real.
"""
import hashlib
import json
import random
import re

from PIL import Image, ImageDraw

from .executor import Executor
from .schema import CHILD_TYPES, MULTI_TYPES, box_intersects, union_box
from .vlm import BBOX_RE

OP_RE = re.compile(r"\[(add|fix|remove|split):([a-z0-9_]+)\]")


def _color(v):
    h = hashlib.md5(v.encode()).digest()
    return (60 + h[0] % 160, 60 + h[1] % 160, 60 + h[2] % 160)


class Truth:
    def __init__(self, plan, rng, e_present=0.85, a_ok=0.75, r_ok=0.7, p_extra=0.08, p_merge=0.08, p_absent=0.1):
        self.present = {}
        self.ok = {}
        self.extras = {}
        self.merged = set()
        self.absent_present = set()
        self.shift = {}
        for v in plan.order:
            t = plan.typ(v)
            if t == "entity":
                self.present[v] = rng.random() < e_present
                self.shift[v] = (rng.randint(-20, 20), rng.randint(-20, 20))
            elif t in CHILD_TYPES:
                self.ok[v] = rng.random() < a_ok
            elif t in MULTI_TYPES:
                self.ok[v] = rng.random() < r_ok
            elif t == "absent":
                if rng.random() < p_absent:
                    self.absent_present.add(v)
        for c in plan.classes():
            if rng.random() < p_extra:
                self.extras[c] = 1
        for v in plan.order:
            if plan.typ(v) == "pair" and rng.random() < p_merge:
                a, b = plan.nodes[v]["pair"]
                if self.present.get(a) and self.present.get(b):
                    self.merged.add(v)

    def to_json(self):
        return json.dumps(dict(present=self.present, ok=self.ok, extras=self.extras, merged=sorted(self.merged), absent_present=sorted(self.absent_present), shift=self.shift))

    @staticmethod
    def from_json(s):
        d = json.loads(s)
        t = Truth.__new__(Truth)
        t.present, t.ok, t.extras = d["present"], d["ok"], d["extras"]
        t.merged, t.absent_present = set(d["merged"]), set(d["absent_present"])
        t.shift = {k: tuple(v) for k, v in d["shift"].items()}
        return t

    def copy(self):
        return Truth.from_json(self.to_json())

    def box(self, plan, v):
        b = list(plan.nodes[v]["box"])
        dx, dy = self.shift.get(v, (0, 0))
        return [max(0, b[0] + dx), max(0, b[1] + dy), min(plan.W, b[2] + dx), min(plan.H, b[3] + dy)]


def render(plan, truth):
    im = Image.new("RGB", (plan.W, plan.H), (200, 200, 200))
    d = ImageDraw.Draw(im)
    d.rectangle([0, plan.H * 0.6, plan.W, plan.H], fill=(150, 160, 170))
    merged_nodes = set()
    for pid in truth.merged:
        a, b = plan.nodes[pid]["pair"]
        merged_nodes |= {a, b}
        ba, bb = truth.box(plan, a), truth.box(plan, b)
        u = union_box([ba, bb])
        cx, cy = (u[0] + u[2]) / 2, (u[1] + u[3]) / 2
        w, h = (ba[2] - ba[0]), (ba[3] - ba[1])
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=_color(a))
    for v in plan.entities():
        if not truth.present.get(v) or v in merged_nodes:
            continue
        b = truth.box(plan, v)
        d.rectangle(b, fill=_color(v))
        d.text((b[0] + 4, b[1] + 4), v[:10], fill=(0, 0, 0))
    for v in plan.order:
        t = plan.typ(v)
        if t in CHILD_TYPES and truth.ok.get(v) and truth.present.get(plan.of(v)[0]):
            b = truth.box(plan, plan.of(v)[0])
            h = hashlib.md5(v.encode()).digest()[0] % 3
            y = b[1] + (b[3] - b[1]) * (0.3 + 0.2 * h)
            d.rectangle([b[0] + 6, y, b[2] - 6, y + 10], fill=_color(v))
        if t in MULTI_TYPES and truth.ok.get(v) and all(truth.present.get(u) for u in plan.of(v)):
            bs = [truth.box(plan, u) for u in plan.of(v)]
            cs = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in bs]
            for i in range(len(cs) - 1):
                d.line([cs[i], cs[i + 1]], fill=_color(v), width=8)
    for c, k in truth.extras.items():
        for v in plan.entities():
            if plan.nodes[v].get("class") == c and truth.present.get(v):
                b = truth.box(plan, v)
                w = b[2] - b[0]
                x0 = min(plan.W - w - 1, b[2] + 30)
                d.rectangle([x0, b[1], x0 + w, b[3]], fill=_color(v))
                break
    for v in truth.absent_present:
        d.rectangle([plan.W * 0.05, plan.H * 0.05, plan.W * 0.2, plan.H * 0.15], fill=(10, 10, 10))
    im.info["truth"] = truth.to_json()
    return im


class MockExecutor(Executor):
    name = "mock"

    def __init__(self, plan, p_fix=0.6, p_collateral=0.25, p_dup=0.3, seed=0, **truth_kw):
        self.plan = plan
        self.p_fix, self.p_collateral, self.p_dup = p_fix, p_collateral, p_dup
        self.truth_kw = truth_kw
        self.seed = seed
        self.calls = 0
        self.sec = 0.0

    def draw(self, prompt, n, seed, size=1024):
        out = []
        for i in range(n):
            rng = random.Random(f"draw:{self.seed}:{seed}:{i}")
            out.append(render(self.plan, Truth(self.plan, rng, **self.truth_kw)))
        self.calls += n
        return out

    def _apply(self, truth, op, v, rng):
        plan = self.plan
        t = plan.typ(v)
        success = rng.random() < self.p_fix
        if op == "add":
            if truth.present.get(v) and rng.random() < self.p_dup:
                truth.extras[plan.nodes[v]["class"]] = truth.extras.get(plan.nodes[v]["class"], 0) + 1
            elif success:
                truth.present[v] = True
        elif op == "fix":
            if t == "entity":
                if success:
                    truth.present[v] = True
            elif success:
                truth.ok[v] = True
        elif op == "remove":
            if t == "count":
                c = plan.nodes[v]["cls"]
                if success and truth.extras.get(c, 0) > 0:
                    truth.extras[c] -= 1
                    if truth.extras[c] == 0:
                        del truth.extras[c]
            elif t == "absent" and success:
                truth.absent_present.discard(v)
        elif op == "split":
            if success:
                truth.merged.discard(v)
        if rng.random() < self.p_collateral:
            cands = [u for u in plan.entities() if truth.present.get(u) and u != v and u not in plan.participants(v)]
            if cands:
                u = rng.choice(cands)
                kids = [w for w in plan.children(u) if plan.typ(w) in CHILD_TYPES and truth.ok.get(w)]
                if kids and rng.random() < 0.7:
                    truth.ok[rng.choice(kids)] = False
                else:
                    truth.present[u] = False
        return truth

    def edit(self, canvas, instruction, n, seed):
        m = OP_RE.search(instruction)
        base = Truth.from_json(canvas.info["truth"])
        out = []
        for i in range(n):
            rng = random.Random(f"edit:{self.seed}:{seed}:{i}:{instruction}")
            tr = base.copy()
            if m:
                tr = self._apply(tr, m.group(1), m.group(2), rng)
            out.append(render(self.plan, tr))
        self.calls += n
        return out

    def commit_hook(self, canvas, cand, composite, info):
        """The committed truth takes the candidate's state inside the committed boxes and the canvas state elsewhere."""
        plan = self.plan
        t0 = Truth.from_json(composite.info.get("truth") or canvas.info["truth"])
        t1 = Truth.from_json(cand.info["truth"])
        boxes = info.get("boxes", [])
        out = t0.copy()

        def inside(b):
            return any(box_intersects(b, x) for x in boxes)

        for v in plan.entities():
            b = t1.box(plan, v) if t1.present.get(v) else t0.box(plan, v)
            if inside(b) or inside(t0.box(plan, v)):
                out.present[v] = t1.present.get(v, False)
                out.shift[v] = t1.shift.get(v, (0, 0))
        for v, ok in t1.ok.items():
            ps = plan.participants(v)
            if any(inside(t1.box(plan, u)) for u in ps):
                out.ok[v] = ok
        for c in set(t0.extras) | set(t1.extras):
            if boxes:
                out.extras[c] = t1.extras.get(c, 0)
                if out.extras.get(c, 0) == 0:
                    out.extras.pop(c, None)
        for pid in set(t0.merged) | set(t1.merged):
            a, b = plan.nodes[pid]["pair"]
            if inside(union_box([t0.box(plan, a), t0.box(plan, b)])):
                if pid in t1.merged:
                    out.merged.add(pid)
                else:
                    out.merged.discard(pid)
        for v in set(t0.absent_present) | set(t1.absent_present):
            if boxes:
                if v in t1.absent_present:
                    out.absent_present.add(v)
                else:
                    out.absent_present.discard(v)
        composite.info["truth"] = out.to_json()
        return composite


class MockVLM:
    """Answers condition questions from the truth tag with error rates f (false fail) and miss (missed problem)."""

    def __init__(self, plan, conds, f=0.05, miss=0.1, f2=0.03, seed=0, jitter=10):
        self.plan = plan
        self.conds = list(conds)
        self.q2c = {c.question: c for c in self.conds}
        self.qmin2c = {c.question_alt: c for c in self.conds if c.question_alt}
        self.f, self.miss, self.f2 = f, miss, f2
        self.seed = seed
        self.jitter = jitter
        self.coord_mode = "abs"
        self.max_side = 1024
        self.model = "mock"
        self.calls = 0
        self.fmt_fail = 0

    def check_gate(self):
        return True

    def _truth_answer(self, truth, c):
        plan = self.plan
        t = c.ntype
        if t == "entity":
            return bool(truth.present.get(c.node))
        if t in CHILD_TYPES:
            return bool(truth.present.get(c.participants[0]) and truth.ok.get(c.node))
        if t in MULTI_TYPES:
            return all(truth.present.get(u) for u in c.participants) and bool(truth.ok.get(c.node))
        if t == "background":
            return True
        if t == "absent":
            return c.node in truth.absent_present
        if c.kind == "count":
            return truth.extras.get(c.cls, 0) > 0
        if c.kind == "pair":
            return c.node not in truth.merged and all(truth.present.get(u) for u in c.participants)
        if c.kind == "quality":
            return False
        return False

    def _noisy(self, ans, key, f):
        rng = random.Random(f"{self.seed}:{key}")
        p = 0.92 - 0.1 * rng.random()
        if ans and rng.random() < f:
            return 1.0 - p
        if (not ans) and rng.random() < self.miss:
            return p
        return p if ans else 1.0 - p

    def yesno_one(self, image, question, tpl="a", inv=False):
        self.calls += 1
        c = self.q2c.get(question)
        tr = image.info.get("truth")
        if c is None and question in self.qmin2c:
            cc = self.qmin2c[question]
            truth = Truth.from_json(tr)
            n_present = sum(int(self.plan.nodes[u].get("count", 1) or 1) for u in self.plan.entities() if self.plan.nodes[u].get("class") == cc.cls and truth.present.get(u))
            p = 0.9 if n_present >= self.plan.nodes[cc.node]["n"] else 0.1
            return (1.0 - p, 0.0) if inv else (p, 0.0)
        if c is None:
            return self._audit(image, question, inv)
        truth = Truth.from_json(tr)
        ans = self._truth_answer(truth, c)
        f = self.f2 if tpl == "b" else self.f
        p = self._noisy(ans, f"{hashlib.md5(tr.encode()).hexdigest()[:8]}:{question}:{tpl}:{image.size}", f)
        if inv:
            p = 1.0 - p
        z = 2.0 if p >= 0.5 else -2.0
        return p, z

    def _audit(self, image, question, inv):
        pr = image.info.get("truth_pair")
        if not pr or pr[0] is None or pr[1] is None:
            return (0.5, 0.0)
        t0, t1 = Truth.from_json(pr[0]), Truth.from_json(pr[1])
        m = OP_RE.search(question)
        tgt = m.group(2) if m else None
        keep = {tgt} | set(self.plan.participants(tgt)) if tgt else set()
        if "clearly visible" in question:
            changed = t0.to_json() != t1.to_json()
            p = 0.9 if changed else 0.1
        elif "still show all the people" in question:
            broke = any(t0.present.get(v) and not t1.present.get(v) for v in self.plan.entities() if v not in keep)
            broke = broke or any(t0.ok.get(v) and not t1.ok.get(v) for v in t0.ok if v not in keep and not (set(self.plan.participants(v)) & keep))
            p = 0.1 if broke else 0.9
        else:
            p = 0.8
        return (1.0 - p, 0.0) if inv else (p, 0.0)

    def yesno(self, items):
        return [self.yesno_one(it["image"], it["question"], it.get("tpl", "a"), bool(it.get("inv", False))) for it in items]

    def _count_truth(self, image, cls):
        """(number of the class in the image, whether an extra copy exists) from the truth tag."""
        truth = Truth.from_json(image.info["truth"])
        n_present = sum(int(self.plan.nodes[u].get("count", 1) or 1) for u in self.plan.entities() if self.plan.nodes[u].get("class") == cls and truth.present.get(u))
        extra = int(truth.extras.get(cls, 0))
        return n_present + extra, extra > 0

    def count_many(self, items):
        """Number question: P(at most n) from the truth with the stage-1 error rate; k is the noisy count."""
        out = []
        for it in items:
            self.calls += 1
            k, extra = self._count_truth(it["image"], it["cls"])
            ans = not extra
            p = self._noisy(ans, f"count:{hashlib.md5(it['image'].info['truth'].encode()).hexdigest()[:8]}:{it['cls']}:{it['image'].size}", self.f)
            out.append((p, 2.0 if p >= 0.5 else -2.0, k if (p >= 0.5) == ans else (k + 1 if ans else max(0, k - 1))))
        return out

    def detect_count_many(self, items):
        """One box per instance: the second opinion with the stage-2 error rate."""
        out = []
        for it in items:
            self.calls += 1
            k, extra = self._count_truth(it["image"], it["cls"])
            ans = not extra
            p = self._noisy(ans, f"detect:{hashlib.md5(it['image'].info['truth'].encode()).hexdigest()[:8]}:{it['cls']}:{it['image'].size}", self.f2)
            out.append((0.95 if p >= 0.5 else 0.05, 2.9 if p >= 0.5 else -2.9, k))
        return out

    def generate_one(self, images, text, max_tokens=300, json_mode=False):
        self.calls += 1
        im = images[0]
        truth = Truth.from_json(im.info["truth"]) if "truth" in im.info else None
        plan = self.plan
        if text.startswith("Locate "):
            desc = text[len("Locate "):].split(" in the image.")[0]
            for v in plan.entities():
                if plan.nodes[v]["desc"].strip().rstrip(".") == desc:
                    if truth and truth.present.get(v):
                        b = truth.box(plan, v)
                        rng = random.Random(f"ground:{self.seed}:{v}:{im.info['truth'][:16]}")
                        j = self.jitter
                        b = [b[0] + rng.randint(-j, j), b[1] + rng.randint(-j, j), b[2] + rng.randint(-j, j), b[3] + rng.randint(-j, j)]
                        return json.dumps({"bbox_2d": [max(0, b[0]), max(0, b[1]), min(plan.W, b[2]), min(plan.H, b[3])]})
                    return "not found"
            return "not found"
        op, node, box = None, None, None
        for v in plan.order:
            d = plan.nodes[v]["desc"].strip().rstrip(".")
            if plan.typ(v) in ("count", "pair", "quality"):
                continue
            if f"must contain: {d}." in text and "not in the image yet" in text:
                op, node = "add", v
            elif f"should contain: {d}." in text and "does not match" in text:
                op, node = "fix", v
            elif f"This must also be true about it: {d}." in text:
                op, node = "fix", v
            elif f"This must be true: {d}." in text:
                op, node = "fix", v
            elif f"must not contain {d}," in text:
                op, node = "remove", v
        if op is None and "should show exactly" in text:
            for v in plan.order:
                if plan.typ(v) == "count" and plan.nodes[v]["desc"] in text:
                    op, node = "remove", v
        if op is None and "two different individuals" in text:
            for v in plan.order:
                if plan.typ(v) == "pair":
                    a, b = plan.nodes[v]["pair"]
                    if plan.nodes[a]["desc"].strip().rstrip(".") in text and plan.nodes[b]["desc"].strip().rstrip(".") in text:
                        op, node = "split", v
        if op is None:
            return "{}"
        if plan.typ(node) == "entity":
            box = truth.box(plan, node) if truth else plan.nodes[node]["box"]
        elif plan.typ(node) in CHILD_TYPES:
            box = truth.box(plan, plan.of(node)[0]) if truth else plan.planned_box(node)
        elif plan.typ(node) in MULTI_TYPES or plan.typ(node) == "pair":
            ps = plan.participants(node)
            box = union_box([truth.box(plan, u) for u in ps]) if truth else plan.planned_box(node)
        elif plan.typ(node) == "count":
            c = plan.nodes[node]["cls"]
            for v in plan.entities():
                if plan.nodes[v].get("class") == c:
                    b = truth.box(plan, v)
                    w = b[2] - b[0]
                    x0 = min(plan.W - w - 1, b[2] + 30)
                    box = [x0, b[1], x0 + w, b[3]]
                    break
        else:
            box = [int(plan.W * 0.05), int(plan.H * 0.05), int(plan.W * 0.2), int(plan.H * 0.15)]
        instr = f"[{op}:{node}] Edit the element {node.replace('_', ' ')} as required"
        return json.dumps({"instruction": instr, "bbox_2d": [int(x) for x in box]})

    def generate(self, items, max_tokens=300, json_mode=False):
        return [self.generate_one(it["images"], it["text"], max_tokens, json_mode) for it in items]

    def parse_bbox(self, text, sent_size, orig_size, min_side=12):
        m = BBOX_RE.search(text)
        if not m:
            return None
        b = [float(g) for g in m.groups()]
        if b[2] - b[0] < min_side or b[3] - b[1] < min_side:
            return None
        return [int(round(x)) for x in b]

    def ground(self, image, desc, max_area=0.7):
        txt = self.generate_one([image], f"Locate {desc.strip().rstrip('.')} in the image.")
        return self.parse_bbox(txt, image.size, image.size)

    def ground_many(self, pairs):
        return [self.ground(im, d) for im, d in pairs]

    def stats(self):
        return dict(calls=self.calls, model="mock", fmt_fail=0)
