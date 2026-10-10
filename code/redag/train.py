"""DAGFlow-GRPO trainer: groups of candidates from the loop's own states, verified with the condition table, credited
per region, updated with the region-level clipped surrogate. Everything GPU-specific lives in policy_qi21.py."""
import json
import os
import random
import time

import numpy as np
import torch

from .commit import region_commit
from .conditions import affected_conditions, build_conditions
from .credit import box_to_grid, partition, region_credit
from .grpo import region_ratio_loss
from .schema import Plan, add_generated_nodes
from .table import Column, Rec
from .verify import Verifier, VerifyConfig
from .views import load_image, to_rgb


def states_from_runs(run_dirs, plan_dir, out_path, min_region=0.0):
    """Every executed branch in the traces becomes a training state: canvas, instruction, target, boxes, column."""
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
                canvas = rnd["canvas"]
                col = table.get((pid, canvas))
                for br in rnd.get("branches", []):
                    rows.append(dict(pid=pid, plan=os.path.join(plan_dir, pid.split("+")[0] + ".json"), canvas=canvas, instruction=br["instruction"], target=dict(key=br["target"], cid=br["target"], node=br["node"], op=br["op"], ntype=br["ntype"], participants=[], region=br.get("region")), boxes=None, column=col, run=rd, round=rnd["round"]))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def column_from_rows(rows_by_cid, image):
    col = Column(image=image)
    if not rows_by_cid:
        return col
    for cid, r in rows_by_cid.items():
        rec = Rec(cid)
        for k in ("s1", "z1", "v1", "s2", "z2", "v2", "s", "status", "region"):
            setattr(rec, k, r.get(k))
        col.recs[cid] = rec
    return col


class TrainConfig:
    def __init__(self, **kw):
        self.G = 8
        self.groups_per_step = 4
        self.steps = 10
        self.train_steps_per_group = 4
        self.noise_level = 0.7
        self.res = 768
        self.lr = 5e-5
        self.alpha = 0.5
        self.alpha_warmup = 50
        self.beta_kl = 0.0
        self.eps = 1e-4
        self.grad_clip = 1.0
        self.lora_rank = 32
        self.max_updates = 400
        self.save_every = 20
        self.seed = 0
        self.lam_q = 0.5
        self.verify = {}
        self.commit = dict(cell=8, thr=0.05, grow=2, feather=0, speck=3, rel=0.3, tone=True)
        for k, v in kw.items():
            if not hasattr(self, k):
                raise ValueError(f"unknown train config key: {k}")
            setattr(self, k, v)

    def to_dict(self):
        return {k: getattr(self, k) for k in vars(self)}


class Trainer:
    def __init__(self, policy, vlm, states, cfg, out_dir):
        self.policy = policy
        self.vlm = vlm
        self.states = states
        self.cfg = cfg
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.params = policy.add_lora(cfg.lora_rank)
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=0.0)
        self.step = 0
        self.rng = random.Random(cfg.seed)
        self.plans = {}
        self.log_path = os.path.join(out_dir, "train.jsonl")
        self.vcfg = VerifyConfig(**cfg.verify)

    def plan_of(self, st):
        p = st["plan"]
        if p not in self.plans:
            self.plans[p] = add_generated_nodes(Plan.load(p))
        return self.plans[p]

    def alpha(self):
        a = self.cfg.alpha
        if self.cfg.alpha_warmup > 0:
            return a * min(1.0, self.step / float(self.cfg.alpha_warmup))
        return a

    def log(self, **kw):
        kw["step"] = self.step
        kw["t"] = time.strftime("%F %T")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw, ensure_ascii=False) + "\n")

    def sample_group(self, st):
        """Rollout + verify + credit for one state. Returns the group dict or None when the group has no signal."""
        plan = self.plan_of(st)
        conds = build_conditions(plan)
        canvas = to_rgb(load_image(st["canvas"]))
        col0 = column_from_rows(st.get("column"), st["canvas"])
        if st.get("boxes"):
            col0.boxes = dict(st["boxes"])
        else:
            for cid, r in col0.recs.items():
                if cid.startswith("n:") and r.region and plan.typ(cid[2:]) == "entity" and r.status and r.status.endswith("pass"):
                    col0.boxes[cid[2:]] = r.region
        cond = self.policy.encode(st["instruction"], canvas, key=(st["instruction"], st["canvas"]))
        traj = self.policy.rollout(cond, self.cfg.G, self.rng.randrange(2 ** 31))
        gh, gw = cond["grid"]
        target = st["target"]
        tnodes = [target["node"]] + list(plan.participants(target["node"]))
        masks, affs = [], []
        for im in traj["images"]:
            _, info = region_commit(canvas, im, scope=[target["region"]] if target.get("region") else [], **self.cfg.commit)
            m = box_to_grid(target["region"], plan.W, plan.H, gh, gw) if target.get("region") else np.zeros((gh, gw), dtype=bool)
            for b in info["boxes"]:
                m |= box_to_grid(b, plan.W, plan.H, gh, gw)
            masks.append(m)
            affs.append(affected_conditions(plan, conds, col0.boxes, info["boxes"] or ([target["region"]] if target.get("region") else []), tnodes, col0.unlocated))
        ver = Verifier(self.vlm, plan, conds, self.vcfg)
        hint = {target["node"]: target["region"]} if target["op"] == "add" and target.get("region") else None
        cols = ver.verify_batch(traj["images"], prevs=[col0] * len(traj["images"]), affecteds=affs, hints=[hint] * len(traj["images"]), relocate=[set(tnodes)] * len(traj["images"]))
        rids, names = partition(plan, col0.boxes, tnodes, masks, gh, gw)
        cr = region_credit(plan, conds, cols, tnodes, names, rids, alpha=self.alpha(), lam_q=self.cfg.lam_q)
        if float(np.std(cr["phi"])) < 1e-6:
            return None
        return dict(cond=cond, traj=traj, rids=torch.stack([torch.as_tensor(r.reshape(-1), dtype=torch.long) for r in rids]), adv=torch.as_tensor(cr["adv"], dtype=torch.float32), w=torch.as_tensor(cr["w"], dtype=torch.float32), phi=cr["phi"], target_scores=[c.score(target["cid"], 0.0) if "cid" in target else 0.0 for c in cols], state=st, names=names)

    def update(self, groups):
        self.opt.zero_grad(set_to_none=True)
        stats = dict(loss=0.0, clipfrac=0.0, n=0)
        T = self.cfg.steps
        for g in groups:
            steps = self.rng.sample(range(T), min(T, self.cfg.train_steps_per_group))
            for t in steps:
                for i in range(self.cfg.G):
                    x_t = g["traj"]["latents"][t][i:i + 1]
                    x_n = g["traj"]["latents"][t + 1][i:i + 1]
                    lp_new = self.policy.logprob(g["cond"], x_t, x_n, t, g["traj"]["sigmas"])
                    lp_old = g["traj"]["logp"][t][i:i + 1].to(lp_new.device)
                    loss, st = region_ratio_loss(lp_new, lp_old, g["rids"][i:i + 1].to(lp_new.device), g["adv"][i:i + 1].to(lp_new.device), g["w"][i:i + 1].to(lp_new.device), self.cfg.eps, self.cfg.eps)
                    (loss / (len(groups) * len(steps) * self.cfg.G)).backward()
                    stats["loss"] += float(loss.detach())
                    stats["clipfrac"] += st["clipfrac"]
                    stats["n"] += 1
        torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
        self.opt.step()
        self.step += 1
        return dict(loss=stats["loss"] / max(1, stats["n"]), clipfrac=stats["clipfrac"] / max(1, stats["n"]))

    def fit(self):
        halt = os.path.join(self.out, "halt")
        while self.step < self.cfg.max_updates and not os.path.exists(halt):
            batch = self.rng.sample(self.states, min(len(self.states), self.cfg.groups_per_step))
            groups = []
            t0 = time.time()
            for st in batch:
                g = self.sample_group(st)
                if g is not None:
                    groups.append(g)
            if not groups:
                self.log(note="no signal in batch")
                continue
            sec_sample = time.time() - t0
            t0 = time.time()
            up = self.update(groups)
            self.log(**up, groups=len(groups), alpha=self.alpha(), sec_sample=round(sec_sample, 1), sec_update=round(time.time() - t0, 1), phi_mean=float(np.mean([np.mean(g["phi"]) for g in groups])), target_mean=float(np.mean([np.mean(g["target_scores"]) for g in groups])))
            if self.step % self.cfg.save_every == 0:
                self.policy.save_lora(os.path.join(self.out, f"lora_s{self.step:05d}.safetensors"))
        self.policy.save_lora(os.path.join(self.out, f"lora_s{self.step:05d}.safetensors"))
