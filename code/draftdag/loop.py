import json
import os
import random
import time

import numpy as np

from .accept import AcceptConfig, judge
from .commit import grids_overlap, merge_commits, region_commit
from .conditions import affected_conditions, build_conditions
from .diagnose import diagnose, write_instruction
from .schedule import Budget, frontier, independent_groups
from .stats import Stats
from .table import write_rows
from .verify import Verifier, VerifyConfig
from .views import load_image, save_png, to_rgb

MODES = ("adaptive", "draw", "repair", "draw_repair")


class LoopConfig:
    def __init__(self, **kw):
        self.mode = "adaptive"
        self.n0 = 4
        self.n_max = 16
        self.G = 4
        self.max_rounds = 8
        self.max_tries = 2
        self.max_parallel = 4
        self.calls_max = None
        self.seconds_max = None
        self.c_e = 2.0
        self.size = 1024
        self.seed = 0
        self.explore = 0.0
        self.protect_h = 0.3
        self.lam_q = 0.5
        self.commit = dict(cell=8, thr=0.05, grow=2, feather=6, speck=3, rel=0.3, tone=True)
        self.verify = {}
        self.accept = {}
        self.stats_path = None
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f"unknown loop config key: {k}")
            setattr(self, k, v)
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")

    def to_dict(self):
        return {k: getattr(self, k) for k in vars(self)}


def _col_summary(plan, conds, col):
    return dict(image=col.image, phi=None, status={c.cid: col.status(c.cid) for c in conds}, score={c.cid: col.score(c.cid) for c in conds if col.status(c.cid) in ("pass", "fail")}, ntype={c.cid: c.ntype for c in conds}, boxes=col.boxes, unlocated=sorted(col.unlocated), cost=col.cost)


class Loop:
    def __init__(self, executor, vlm, cfg, out_dir, stats=None):
        self.ex = executor
        self.vlm = vlm
        self.cfg = cfg
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.stats = stats or Stats.load(cfg.stats_path)
        self.vcfg = VerifyConfig(**cfg.verify)
        self.acfg = AcceptConfig(**cfg.accept)
        self.events = os.path.join(out_dir, "events.jsonl")

    def log(self, **kw):
        kw["t"] = time.strftime("%F %T")
        with open(self.events, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw, ensure_ascii=False) + "\n")

    def _save(self, im, d, name):
        p = os.path.join(d, name)
        save_png(im, p)
        return p

    def _protect(self, plan, col, target):
        if self.stats is None:
            return []
        out = []
        for u in plan.entities():
            if u in target.get("participants", []) or u == target["node"]:
                continue
            if u in col.boxes and col.passes("n:" + u):
                for a, b in plan.interfere_pairs():
                    if {a, b} & set(target.get("participants", [])) and u in (a, b) and self.stats.interference_of(target["op"], target["ntype"]) >= self.cfg.protect_h:
                        out.append(col.boxes[u])
        return out

    def _draw(self, plan, n, seed, d, start):
        t0 = time.time()
        ims = self.ex.draw(plan.prompt, n, seed, self.cfg.size)
        sec = time.time() - t0
        if getattr(self.ex, "last_sec", None) is not None:
            sec = float(self.ex.last_sec)
        paths = [self._save(im, d, f"draft_{start + i:02d}.png") for i, im in enumerate(ims)]
        return ims, paths, sec

    def run_prompt(self, plan, tag=""):
        pid = plan.id + tag
        d = os.path.join(self.out, "img", pid)
        done = os.path.join(d, "done.json")
        if os.path.exists(done):
            return json.load(open(done, encoding="utf-8"))
        os.makedirs(d, exist_ok=True)
        self.ex.pid = plan.id
        rng = random.Random(f"{self.cfg.seed}:{pid}")
        conds = build_conditions(plan)
        ver = Verifier(self.vlm, plan, conds, self.vcfg)
        budget = Budget(self.cfg.calls_max, self.cfg.seconds_max, self.cfg.c_e)
        trace = dict(pid=pid, config=self.cfg.to_dict(), stage1=dict(), rounds=[], final=dict())
        rows = []

        drafts, dpaths, cols = [], [], []
        sec_draw = 0.0
        decision, q_hat, m_hat = "stop", None, None
        while True:
            n = self.cfg.n0 if not drafts else min(self.cfg.n0, self.cfg.n_max - len(drafts))
            if n <= 0 or not budget.room(n):
                break
            ims, paths, sec = self._draw(plan, n, rng.randrange(2 ** 31), d, len(drafts))
            budget.add_drafts(len(ims), sec)
            sec_draw += sec
            t0 = time.time()
            new_cols = ver.verify_batch(paths, tags=[f"draft{len(drafts) + i}" for i in range(len(paths))])
            budget.add_eval(time.time() - t0)
            for i, c in enumerate(new_cols):
                rows.extend(c.rows(pid, 0, len(drafts) + i))
            drafts += ims
            dpaths += paths
            cols += new_cols
            phis = [c.phi(conds, self.cfg.lam_q) for c in cols]
            pick = int(np.argmax(phis))
            q_hat = float(np.mean([c.all_pass(conds) for c in cols]))
            m_hat = float(len(cols[pick].failed(conds, ("node", "count", "pair"))))
            all_ok = cols[pick].all_pass(conds)
            if self.cfg.mode == "repair":
                decision = "repair"
            elif self.cfg.mode == "draw_repair":
                decision = "repair" if (len(drafts) >= self.cfg.n0) else "draw"
            elif self.cfg.mode == "draw":
                decision = "stop" if all_ok else "draw"
            else:
                if all_ok:
                    decision = "stop"
                else:
                    ops = [t["op"] for t in diagnose(plan, conds, cols[pick]) if t["op"] != "skip"]
                    decision = "repair" if self.stats.decide(q_hat, m_hat, ops) else "draw"
            self.log(pid=pid, stage=1, n_drafts=len(drafts), pick=pick, phi=phis[pick], q_hat=q_hat, m_hat=m_hat, decision=decision)
            if decision != "draw" or len(drafts) >= self.cfg.n_max:
                break
        if not drafts:
            raise RuntimeError(f"{pid}: budget too small for a single draft")
        phis = [c.phi(conds, self.cfg.lam_q) for c in cols]
        pick = int(np.argmax(phis))
        trace["stage1"] = dict(n_drafts=len(drafts), drafts=dpaths, phi=phis, pick=pick, q_hat=q_hat, m_hat=m_hat, decision=decision, sec_draw=sec_draw, columns=[_col_summary(plan, conds, c) for c in cols])
        canvas, cpath, col = drafts[pick], dpaths[pick], cols[pick]
        tried = {}
        if decision == "repair":
            for r in range(self.cfg.max_rounds):
                if col.all_pass(conds) or budget.exhausted():
                    break
                t0 = time.time()
                targets = [t for t in diagnose(plan, conds, col, self.vlm, canvas) if t["op"] != "skip"]
                fr = frontier(plan, conds, col, targets, tried, self.cfg.max_tries)
                if not fr:
                    self.log(pid=pid, round=r, note="no frontier targets", failed=[t["key"] for t in targets])
                    break
                written = []
                for t in fr:
                    w = write_instruction(self.vlm, plan, canvas, t)
                    if w is None:
                        tried[t["key"]] = tried.get(t["key"], 0) + 1
                        self.log(pid=pid, round=r, note="instruction unparsable", target=t["key"])
                        continue
                    written.append(w)
                budget.add_eval(time.time() - t0)
                if not written:
                    continue
                groups = independent_groups(plan, written, self.cfg.max_parallel, stats=self.stats)
                group = groups[0]
                rd = dict(round=r, canvas=cpath, n_targets=len(targets), n_frontier=len(fr), groups=[[t["key"] for t in g] for g in groups], branches=[])
                accepted = []
                for t in group:
                    tried[t["key"]] = tried.get(t["key"], 0) + 1
                    if not budget.room(self.cfg.G):
                        break
                    seed = rng.randrange(2 ** 31)
                    t1 = time.time()
                    cands = self.ex.edit(canvas, t["instruction"], self.cfg.G, seed)
                    sec_edit = time.time() - t1
                    budget.add_edits(len(cands), sec_edit)
                    t1 = time.time()
                    edited = [t["node"]] + list(t.get("participants", []))
                    protect = self._protect(plan, col, t)
                    comps, infos, paths, raws = [], [], [], []
                    for i, cand in enumerate(cands):
                        raws.append(self._save(cand, d, f"r{r:02d}_{t['node']}_raw{i}.png"))
                        comp, info = region_commit(canvas, cand, scope=[t["region"]] if t.get("region") else [], protect=protect, **self.cfg.commit)
                        comp = self.ex.commit_hook(canvas, cand, comp, info)
                        comps.append(comp)
                        infos.append(info)
                        paths.append(self._save(comp, d, f"r{r:02d}_{t['node']}_c{i}.png"))
                    affs = [affected_conditions(plan, conds, col.boxes, info["boxes"] or ([t["region"]] if t.get("region") else []), edited, col.unlocated) for info in infos]
                    hint = {t["node"]: t["region"]} if t["op"] == "add" and t.get("region") else None
                    ccols = ver.verify_batch(paths, prevs=[col] * len(paths), affecteds=affs, hints=[hint] * len(paths), relocate=[set(edited)] * len(paths), tags=[f"r{r}_{t['node']}_c{i}" for i in range(len(paths))])
                    judged = []
                    for i, cc in enumerate(ccols):
                        jd = judge(plan, conds, self.vlm, canvas, col, comps[i], cc, infos[i], t, affs[i], self.acfg)
                        jd["phi"] = cc.phi(conds, self.cfg.lam_q)
                        jd["share"] = infos[i]["share"]
                        jd["path"] = paths[i]
                        jd["raw"] = raws[i]
                        judged.append(jd)
                        rows.extend(cc.rows(pid, r + 1, i))
                    sec_eval = time.time() - t1
                    budget.add_eval(sec_eval)
                    ok = [i for i, jd in enumerate(judged) if jd["accept"]]
                    chosen = None
                    if ok:
                        chosen = max(ok, key=lambda i: (judged[i]["phi"], -judged[i]["share"]))
                        if len(ok) > 1 and self.cfg.explore > 0 and rng.random() < self.cfg.explore:
                            chosen = rng.choice([i for i in ok if i != chosen])
                    br = dict(target=t["key"], node=t["node"], op=t["op"], ntype=t["ntype"], instruction=t["instruction"], region=t.get("region"), bbox=t.get("bbox"), protect=protect, seed=seed, sec_edit=sec_edit, sec_eval=sec_eval, candidates=[dict(path=jd["path"], raw=jd["raw"], accept=jd["accept"], phi=jd["phi"], share=jd["share"], rules=jd["rules"], commit=dict(share=infos[i]["share"], raw=infos[i]["raw"], kept=infos[i]["kept"], dropped=infos[i]["dropped"], boxes=infos[i]["boxes"])) for i, jd in enumerate(judged)], accepted=chosen is not None, chosen=chosen, affected=sorted(affs[0]))
                    rd["branches"].append(br)
                    if chosen is not None:
                        accepted.append(dict(target=t, comp=comps[chosen], col=ccols[chosen], info=infos[chosen], aff=affs[chosen], phi=judged[chosen]["phi"]))
                merged = []
                for a in accepted:
                    if all(not grids_overlap(a["info"]["grid"], b["info"]["grid"], 1) for b in merged):
                        merged.append(a)
                deferred = [a["target"]["key"] for a in accepted if a not in merged]
                for k in deferred:
                    tried[k] = max(0, tried.get(k, 1) - 1)
                if merged:
                    if len(merged) == 1:
                        new_canvas = merged[0]["comp"]
                    else:
                        new_canvas = merge_commits(canvas, [(a["comp"], a["info"]["grid"]) for a in merged])
                        for a in merged:
                            new_canvas = self.ex.commit_hook(canvas, a["comp"], new_canvas, a["info"])
                    cpath = self._save(new_canvas, d, f"canvas_r{r:02d}.png")
                    aff = set()
                    edited = []
                    for a in merged:
                        aff |= set(a["aff"])
                        edited += [a["target"]["node"]] + list(a["target"].get("participants", []))
                    t1 = time.time()
                    if len(merged) == 1:
                        col = merged[0]["col"]
                        col.image = cpath
                    else:
                        col = ver.verify(cpath, prev=col, affected=aff, relocate=set(edited), tag=f"canvas_r{r}")
                    budget.add_eval(time.time() - t1)
                    rows.extend(col.rows(pid, r + 1, -1))
                    canvas = load_image(cpath)
                rd.update(merged=[a["target"]["key"] for a in merged], deferred=deferred, phi_after=col.phi(conds, self.cfg.lam_q), n_pass=col.n_pass(conds), n_fail=len(col.failed(conds, ("node", "count", "pair"))), budget=budget.to_dict())
                trace["rounds"].append(rd)
                self.log(pid=pid, round=r, merged=rd["merged"], n_fail=rd["n_fail"], calls=budget.calls)
        fpath = self._save(canvas, d, "final.png")
        trace["final"] = dict(image=fpath, phi=col.phi(conds, self.cfg.lam_q), n_pass=col.n_pass(conds), n_cond=col.n_scored(conds), all_pass=col.all_pass(conds), column=_col_summary(plan, conds, col), budget=budget.to_dict(), executor=self.ex.stats(), vlm=self.vlm.stats())
        with open(os.path.join(d, "trace.json"), "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=1)
        write_rows(os.path.join(self.out, "table.jsonl"), rows)
        summ = dict(pid=pid, final=fpath, all_pass=trace["final"]["all_pass"], n_pass=trace["final"]["n_pass"], n_cond=trace["final"]["n_cond"], rounds=len(trace["rounds"]), calls=budget.calls, elapsed=budget.elapsed(), decision=decision, n_drafts=len(drafts))
        with open(done, "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False)
        return summ
