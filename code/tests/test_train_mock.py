import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import lab_plan

from redag.conditions import build_conditions
from redag.grpo import sde_coeffs, sde_mean, sde_step, token_logprob
from redag.loop import Loop, LoopConfig
from redag.mock import MockExecutor, MockVLM
from redag.schema import strip_generated
from redag.train import TrainConfig, Trainer, states_from_runs


class DummyPolicy:
    """Same interface as QI21Policy on a tiny random model; images come from the mock executor so the mock VLM can score them."""

    def __init__(self, ex, steps=4, noise_level=0.7, grid=(12, 12), C=8):
        self.ex = ex
        self.steps = steps
        self.noise_level = noise_level
        self.grid = grid
        self.C = C
        self.device = torch.device("cpu")
        self.net = torch.nn.Linear(C, C)
        self.saved = []

    def add_lora(self, rank=4):
        return list(self.net.parameters())

    def save_lora(self, path):
        self.saved.append(path)

    def encode(self, instruction, canvas, key=None):
        gh, gw = self.grid
        sig = list(np.linspace(1.0, 1.0 / self.steps, self.steps)) + [0.0]
        return dict(n_tokens=gh * gw, grid=self.grid, sigmas=sig, instruction=instruction, canvas=canvas)

    def _pred(self, cond, x, sigma):
        return self.net(x) * 0.1 + x * 0.5

    def rollout(self, cond, G, seed, sde_steps=None):
        gen = torch.Generator().manual_seed(int(seed))
        x = torch.randn((G, cond["n_tokens"], self.C), generator=gen)
        sig = cond["sigmas"]
        lat, lps = [x.clone()], []
        with torch.no_grad():
            for i in range(len(sig) - 1):
                v = self._pred(cond, x, sig[i])
                x_next, mean, std = sde_step(x, v, sig[i], sig[i + 1], sig[1], self.noise_level, generator=gen)
                lps.append(token_logprob(x_next, mean, std))
                x = x_next
                lat.append(x.clone())
        images = self.ex.edit(cond["canvas"], cond["instruction"], G, seed)
        return dict(latents=torch.stack(lat), logp=torch.stack(lps), sigmas=sig, images=images, seed=seed)

    def logprob(self, cond, x_t, x_next, step, sigmas):
        v = self._pred(cond, x_t, sigmas[step])
        mean = sde_mean(x_t, v, sigmas[step], sigmas[step + 1], sigmas[1], self.noise_level)
        _, _, std = sde_coeffs(sigmas[step], sigmas[step + 1], sigmas[1], self.noise_level)
        return token_logprob(x_next, mean, std)


def test_trainer_plumbing():
    plan = lab_plan()
    conds = build_conditions(plan)
    out = tempfile.mkdtemp(prefix="dagtrain_")
    try:
        plan_dir = os.path.join(out, "plans")
        os.makedirs(plan_dir)
        strip_generated(plan).save(os.path.join(plan_dir, plan.id + ".json"))
        ex = MockExecutor(plan, p_fix=0.6, p_collateral=0.3, p_dup=0.3, seed=9, e_present=0.6, a_ok=0.6, r_ok=0.5, p_extra=0.2, p_merge=0.2, p_absent=0.2)
        vlm = MockVLM(plan, conds, f=0.0, miss=0.0, f2=0.0, seed=9)
        run_dir = os.path.join(out, "run")
        loop = Loop(ex, vlm, LoopConfig(mode="repair", n0=2, G=2, max_rounds=3, max_tries=1, max_parallel=2, seed=1), run_dir)
        loop.run_prompt(plan)
        states = states_from_runs([run_dir], plan_dir, os.path.join(out, "states.jsonl"))
        assert states, "no training states from the run"
        assert all(s["column"] is not None for s in states)
        pol = DummyPolicy(ex)
        cfg = TrainConfig(G=3, groups_per_step=2, steps=4, train_steps_per_group=2, max_updates=2, save_every=1, lr=1e-3, alpha_warmup=0)
        tr = Trainer(pol, vlm, states, cfg, os.path.join(out, "train"))
        g = tr.sample_group(states[0])
        if g is not None:
            assert g["rids"][0].shape[0] == 12 * 12
            assert g["adv"].shape[0] == 3 and g["w"].shape == g["adv"].shape
            assert abs(float(g["w"].sum(1)[0]) - 1) < 1e-5
        before = [p.detach().clone() for p in pol.net.parameters()]
        tr.fit()
        after = [p.detach() for p in pol.net.parameters()]
        logs = [json.loads(l) for l in open(tr.log_path)]
        assert logs
        if any("loss" in l for l in logs):
            assert any(not torch.equal(a, b) for a, b in zip(before, after)), "parameters did not change after an update"
        assert pol.saved
        print("train log:", logs[-1])
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    test_trainer_plumbing()
    print("train tests ok")
