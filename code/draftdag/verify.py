from .conditions import TEMPLATE_PLAN, VIEW_PLAN, is_askable, region_of
from .schema import box_area
from .table import Column, Rec
from .views import load_image, make_view, to_rgb

LEVEL1 = ("background", "entity", "absent", "count", "quality")


class VerifyConfig:
    def __init__(self, thr=0.5, full_side=896, crop_min=448, crop_max=1024, two_stage=True, locate=True, min_box_side=16, max_box_area=0.7, count_mode="number", rescue_min=None, confirm_band=None, three_way=False):
        self.thr = float(thr)
        # second-stage rules: a stage-1 fail is rescued only when the second view scores at least rescue_min (default thr);
        # a stage-1 pass inside confirm_band=(lo, hi) is confirmed on the second view and fails when that view fails
        self.rescue_min = float(rescue_min) if rescue_min is not None else float(thr)
        self.confirm_band = tuple(float(x) for x in confirm_band) if confirm_band else None
        self.full_side = int(full_side)
        self.crop_min = int(crop_min)
        self.crop_max = int(crop_max)
        self.two_stage = bool(two_stage)
        self.locate = bool(locate)
        self.min_box_side = int(min_box_side)
        self.max_box_area = float(max_box_area)
        # count conditions: "number" asks how many (stage 1) and lists one box per instance (stage 2) on the native image;
        # "yesno" is the old inverted "N+1 or more?" question in both stages (kept for comparison runs)
        self.count_mode = count_mode if count_mode in ("number", "yesno") else "number"
        # three_way: stage-1 yes/no questions offer "partly" as a third answer (template c); stage 2 stays plain yes/no
        self.three_way = bool(three_way)


class Verifier:
    def __init__(self, vlm, plan, conds, cfg=None):
        self.vlm = vlm
        self.plan = plan
        self.conds = list(conds)
        self.cfg = cfg or VerifyConfig()
        self.by_cid = {c.cid: c for c in self.conds}

    def _view(self, im, c, stage, region):
        spec = VIEW_PLAN[c.ntype][stage]
        if spec is None:
            return None, None, None
        kind, margin = spec
        if kind in ("crop", "boxed") and region is None:
            kind = "full"
        view, win = make_view(im, kind, region, full_side=self.cfg.full_side, crop_margin=margin, crop_min=self.cfg.crop_min, crop_max=self.cfg.crop_max)
        tpls = TEMPLATE_PLAN[kind]
        same_as_stage1 = stage == 1 and VIEW_PLAN[c.ntype][0][0] == kind and VIEW_PLAN[c.ntype][0][1] == margin
        tpl = tpls[1] if same_as_stage1 else tpls[0]
        if self.cfg.three_way and stage == 0:
            tpl = "c"
        return view, dict(kind=kind, margin=margin, win=list(win) if win else None, tpl=tpl), tpl

    def _record(self, col, c, stage, region, vinfo, p, z):
        r = col.recs.setdefault(c.cid, Rec(c.cid))
        col.cost["questions"] += 1
        r.region = region
        if stage == 0:
            r.s1, r.z1, r.v1 = float(p), float(z), vinfo
        else:
            r.s2, r.z2, r.v2 = float(p), float(z), vinfo

    def _ask(self, jobs):
        """jobs: list of (col, cond, stage, region). Runs stage views through the VLM and fills recs. Count conditions in
        count_mode "number" are asked for the number (stage 1) and for one box per instance (stage 2) instead of yes/no."""
        items, meta, cnt = [], [], {0: ([], []), 1: ([], [])}
        for col, c, stage, region in jobs:
            im = col._im
            view, vinfo, tpl = self._view(im, c, stage, region)
            if view is None:
                continue
            if c.ntype == "count" and self.cfg.count_mode == "number":
                node = self.plan.nodes[c.node]
                cnt[stage][0].append(dict(image=view, plural=node.get("plural") or node.get("cls") or c.cid, cls=node.get("cls") or c.cid, n=int(node.get("n", 1) or 1)))
                cnt[stage][1].append((col, c, stage, region, vinfo))
                continue
            items.append(dict(image=view, question=c.question, tpl=tpl, inv=c.inv))
            meta.append((col, c, stage, region, vinfo))
        res = self.vlm.yesno(items)
        for (col, c, stage, region, vinfo), (p, z) in zip(meta, res):
            self._record(col, c, stage, region, vinfo, p, z)
        for stage, (citems, cmeta) in cnt.items():
            if not citems:
                continue
            cres = self.vlm.count_many(citems) if stage == 0 else self.vlm.detect_count_many(citems)
            for (col, c, st, region, vinfo), (p, z, k) in zip(cmeta, cres):
                vinfo = dict(vinfo, method="number" if st == 0 else "detect", k=k)
                self._record(col, c, st, region, vinfo, p, z)

    def _in_band(self, s1):
        b = self.cfg.confirm_band
        return bool(b) and b[0] <= s1 < b[1]

    def _settle(self, col, c):
        r = col.recs[c.cid]
        thr = self.cfg.thr
        if r.s1 is None:
            r.status, r.s = "unasked", None
            return
        if r.s1 >= thr:
            if self._in_band(r.s1) and r.s2 is not None and not (r.v2 or {}).get("kind") == "ground-miss":
                if r.s2 >= thr:
                    r.status, r.s = "pass", 0.5 * (r.s1 + r.s2)
                else:
                    r.status, r.s = "fail", 0.5 * (r.s1 + r.s2)
            else:
                r.status, r.s = "pass", r.s1
        elif r.s2 is None:
            r.status, r.s = "fail", r.s1
        elif r.s2 >= self.cfg.rescue_min:
            r.status, r.s = "pass", r.s2
        else:
            r.status, r.s = "fail", 0.5 * (r.s1 + r.s2)

    def _two_stage(self, jobs):
        """Stage 1 on every job; stage 2 only for conditions that failed stage 1. For entities stage 2 first asks the
        evaluator to locate the described object anywhere in the image: a hit is confirmed on a crop around it, a miss
        confirms the absence. Other types use the second view of VIEW_PLAN."""
        self._ask([(col, c, 0, region) for col, c, region in jobs])
        second, ground = [], []
        for col, c, region in jobs:
            r = col.recs.get(c.cid)
            if r is None or r.s1 is None:
                continue
            if (r.s1 < self.cfg.thr or self._in_band(r.s1)) and self.cfg.two_stage and VIEW_PLAN[c.ntype][1] is not None:
                if c.ntype == "entity" and self.cfg.locate:
                    ground.append((col, c))
                else:
                    second.append((col, c, 1, region))
        if ground:
            boxes = self.vlm.ground_many([(col._im, self.plan.nodes[c.node]["desc"]) for col, c in ground])
            W, H = self.plan.W, self.plan.H
            for (col, c), b in zip(ground, boxes):
                col.cost["grounds"] += 1
                r = col.recs[c.cid]
                ok = b is not None and (b[2] - b[0]) >= self.cfg.min_box_side and (b[3] - b[1]) >= self.cfg.min_box_side and box_area(b) <= self.cfg.max_box_area * W * H
                if ok:
                    second.append((col, c, 1, [int(x) for x in b]))
                    col.boxes[c.node] = [int(x) for x in b]
                else:
                    r.s2, r.z2, r.v2 = 0.0, -30.0, dict(kind="ground-miss")
        if second:
            self._ask(second)
        for col, c, region in jobs:
            if c.cid in col.recs:
                self._settle(col, c)
                if c.ntype == "entity" and col.status(c.cid) != "pass" and c.node in col.boxes and col.recs[c.cid].v2 and col.recs[c.cid].v2.get("kind") == "crop":
                    col.boxes.pop(c.node, None)

    def _locate(self, pairs):
        """pairs: list of (col, node). Grounds entity descriptions and stores boxes."""
        if not pairs or not self.cfg.locate:
            for col, v in pairs:
                col.unlocated.add(v)
            return
        req = [(col._im, self.plan.nodes[v]["desc"]) for col, v in pairs]
        boxes = self.vlm.ground_many(req)
        W, H = self.plan.W, self.plan.H
        for (col, v), b in zip(pairs, boxes):
            col.cost["grounds"] += 1
            ok = b is not None and (b[2] - b[0]) >= self.cfg.min_box_side and (b[3] - b[1]) >= self.cfg.min_box_side and box_area(b) <= self.cfg.max_box_area * W * H
            if ok:
                col.boxes[v] = [int(x) for x in b]
                col.unlocated.discard(v)
            else:
                col.unlocated.add(v)

    def verify_batch(self, images, prevs=None, affecteds=None, hints=None, relocate=None, tags=None):
        """Verify several images in lockstep. prevs/affecteds: carry over records outside the affected set.
        hints: per image dict node->box for entities whose location is only known from the edit region.
        relocate: per image set of entity ids to re-ground even if a box is known."""
        n = len(images)
        prevs = prevs or [None] * n
        affecteds = affecteds or [None] * n
        hints = hints or [None] * n
        relocate = relocate or [set()] * n
        tags = tags or [""] * n
        cols = []
        for i in range(n):
            im = load_image(images[i])
            col = Column(tag=tags[i], image=images[i] if isinstance(images[i], str) else None)
            col._im = to_rgb(im)
            if prevs[i] is not None:
                col.boxes = dict(prevs[i].boxes)
                col.unlocated = set(prevs[i].unlocated)
                aff = affecteds[i] if affecteds[i] is not None else set(c.cid for c in self.conds)
                for cid, r in prevs[i].recs.items():
                    if cid not in aff:
                        col.recs[cid] = r.copy_as_carried()
            cols.append(col)
        jobs = []
        for i, col in enumerate(cols):
            for c in self.conds:
                if c.ntype in LEVEL1 and c.cid not in col.recs:
                    jobs.append((col, c, region_of(self.plan, c, col.boxes, hints[i])))
        self._two_stage(jobs)
        pairs = []
        for i, col in enumerate(cols):
            for v in self.plan.entities():
                if not col.passes("n:" + v):
                    continue
                fresh = not col.recs["n:" + v].status.startswith("carried")
                if v not in col.boxes or v in relocate[i] or (hints[i] and v in hints[i]) or (v in col.unlocated and fresh):
                    pairs.append((col, v))
        self._locate(pairs)
        jobs = []
        for i, col in enumerate(cols):
            sts = col.statuses()
            for c in self.conds:
                if c.ntype in LEVEL1 or c.cid in col.recs:
                    continue
                if is_askable(c, sts):
                    jobs.append((col, c, region_of(self.plan, c, col.boxes, hints[i])))
                else:
                    r = col.recs.setdefault(c.cid, Rec(c.cid))
                    r.status, r.s = "unasked", None
        self._two_stage(jobs)
        for col in cols:
            del col._im
        return cols

    def verify(self, image, prev=None, affected=None, hint=None, relocate=None, tag=""):
        return self.verify_batch([image], [prev], [affected], [hint], [relocate or set()], [tag])[0]
