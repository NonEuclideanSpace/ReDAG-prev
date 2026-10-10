"""Flow-matching regression on accepted edits with condition-table weights. Warm-up before GRPO only; it learns
'do something' and 'leave the rest alone', which region commit already guarantees at inference."""
import json
import math
import os
import random
import time

import numpy as np
import torch

from .commit import region_commit
from .conditions import build_conditions, region_of
from .credit import box_to_grid
from .schema import Plan, add_generated_nodes
from .train import column_from_rows
from .views import load_image, to_rgb


def accepted_from_runs(run_dirs, plan_dir, out_path):
    rows = []
    for rd in run_dirs:
        root = os.path.join(rd, "img")
        if not os.path.isdir(root):
            continue
        table = {}
        tpath = os.path.join(rd, "table.jsonl")
        if os.path.exists(tpath):
            with open(tpath, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    table.setdefault((r["pid"], r["image"]), {})[r["cid"]] = r
        for pid in sorted(os.listdir(root)):
            tp = os.path.join(root, pid, "trace.json")
            if not os.path.exists(tp):
                continue
            tr = json.load(open(tp, encoding="utf-8"))
            for rnd in tr.get("rounds", []):
                for br in rnd.get("branches", []):
                    if br.get("chosen") is None:
                        continue
                    cj = br["candidates"][br["chosen"]]
                    rows.append(dict(pid=pid, plan=os.path.join(plan_dir, pid.split("+")[0] + ".json"), canvas=rnd["canvas"], target_image=cj["path"], instruction=br["instruction"], op=br["op"], node=br["node"], region=br.get("region"), column=table.get((pid, rnd["canvas"])), commit_boxes=cj["commit"]["boxes"]))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def token_weights(plan, conds, col, commit_boxes, region, gh, gw, omega=0.5):
    """Edit region gets omega of the weight (uniform); the rest is split across live conditions in proportion to
    psi*s, uniform inside each condition's region outside the edit."""
    edit = np.zeros((gh, gw), dtype=bool)
    for b in list(commit_boxes or []) + ([region] if region else []):
        edit |= box_to_grid(b, plan.W, plan.H, gh, gw)
    keep = ~edit
    w = np.zeros((gh, gw), dtype=np.float64)
    tot = 0.0
    for c in conds:
        if c.kind == "quality" or not col.passes(c.cid):
            continue
        r = region_of(plan, c, col.boxes)
        m = (box_to_grid(r, plan.W, plan.H, gh, gw) if r is not None else np.ones((gh, gw), dtype=bool)) & keep
        if not m.any():
            continue
        val = c.psi * col.score(c.cid, 0.0)
        w[m] += val / m.sum()
        tot += val
    N = gh * gw
    out = np.zeros((gh, gw), dtype=np.float64)
    if edit.any():
        out[edit] = omega * N / edit.sum()
    if tot > 0 and keep.any():
        out[keep] = (1 - omega) * N * w[keep] / tot
    elif keep.any():
        out[keep] = (1 - omega) * N / keep.sum()
    return out.reshape(-1)


class Warmup:
    def __init__(self, policy, samples, out_dir, lr=5e-5, lora_rank=32, batch=2, max_steps=300, omega=0.5, seed=0, save_every=50):
        self.policy = policy
        self.samples = samples
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.params = policy.add_lora(lora_rank)
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=0.0)
        self.batch, self.max_steps, self.omega, self.save_every = batch, max_steps, omega, save_every
        self.rng = random.Random(seed)
        self.plans = {}
        self.step = 0

    def plan_of(self, s):
        if s["plan"] not in self.plans:
            self.plans[s["plan"]] = add_generated_nodes(Plan.load(s["plan"]))
        return self.plans[s["plan"]]

    def example(self, s):
        plan = self.plan_of(s)
        conds = build_conditions(plan)
        canvas = to_rgb(load_image(s["canvas"]))
        target = to_rgb(load_image(s["target_image"]))
        cond = self.policy.encode(s["instruction"], canvas, key=(s["instruction"], s["canvas"]))
        x1 = self.policy.encode_target(target, cond)
        col = column_from_rows(s.get("column"), s["canvas"])
        for cid, r in col.recs.items():
            if cid.startswith("n:") and r.region and plan.typ(cid[2:]) == "entity" and r.status and r.status.endswith("pass"):
                col.boxes[cid[2:]] = r.region
        gh, gw = cond["grid"]
        w = token_weights(plan, conds, col, s.get("commit_boxes"), s.get("region"), gh, gw, self.omega)
        return cond, x1, torch.as_tensor(w, dtype=torch.float32)

    def fit(self):
        halt = os.path.join(self.out, "halt")
        log = open(os.path.join(self.out, "warmup.jsonl"), "a", encoding="utf-8")
        while self.step < self.max_steps and not os.path.exists(halt):
            self.opt.zero_grad(set_to_none=True)
            tot = 0.0
            for s in self.rng.sample(self.samples, min(self.batch, len(self.samples))):
                cond, x1, w = self.example(s)
                loss = self.policy.fm_loss(cond, x1, w, generator=torch.Generator("cpu").manual_seed(self.rng.randrange(2 ** 31)))
                (loss / self.batch).backward()
                tot += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.opt.step()
            self.step += 1
            log.write(json.dumps(dict(step=self.step, loss=tot / self.batch, t=time.strftime("%F %T"))) + "\n")
            log.flush()
            if self.step % self.save_every == 0:
                self.policy.save_lora(os.path.join(self.out, f"lora_w{self.step:05d}.safetensors"))
        self.policy.save_lora(os.path.join(self.out, f"lora_w{self.step:05d}.safetensors"))
