import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import lab_plan

from draftdag.conditions import build_conditions
from draftdag.credit import box_to_grid, partition, region_credit
from draftdag.grpo import ode_step, region_ratio_loss, sde_coeffs, sde_mean, sde_step, token_logprob
from draftdag.table import Column, Rec


def _col(conds, scores, statuses=None):
    col = Column()
    for c in conds:
        r = Rec(c.cid)
        if c.cid in scores:
            r.s = scores[c.cid]
            r.status = "pass" if r.s >= 0.5 else "fail"
        else:
            r.status = "unasked"
        col.recs[c.cid] = r
    return col


def test_region_credit_separates_target_and_damage():
    plan = lab_plan()
    conds = build_conditions(plan)
    gh = gw = 32
    boxes = {v: plan.nodes[v]["box"] for v in plan.entities()}
    target = ["researcher_c"]
    tmask = box_to_grid(boxes["researcher_c"], plan.W, plan.H, gh, gw)
    rids, names = partition(plan, boxes, target, [tmask] * 3, gh, gw)
    assert names[0] == "target" and names[-1] == "bg"
    assert rids[0][tmask].min() == 0 and rids[0][tmask].max() == 0
    base = {c.cid: 0.9 for c in conds if c.kind != "quality"}
    base.update({"q:q_vis": 0.9, "q:q_art": 0.9})
    s_good = dict(base, **{"n:researcher_c": 0.9})
    s_damage = dict(base, **{"n:researcher_c": 0.9, "n:a_pipette": 0.1})
    s_fail = dict(base, **{"n:researcher_c": 0.1})
    cols = [_col(conds, s_good), _col(conds, s_damage), _col(conds, s_fail)]
    cr = region_credit(plan, conds, cols, target, names, rids, alpha=0.5)
    j_a = names.index("researcher_a")
    assert cr["adv"][1, j_a] < cr["adv"][0, j_a], "damage to researcher_a must lower its region advantage"
    assert cr["adv"][2, 0] < cr["adv"][0, 0], "failed target must lower the target advantage"
    assert abs(cr["w"].sum(1) - 1).max() < 1e-6
    assert cr["tok_adv"].shape == (3, gh * gw)
    assert np.isclose(cr["phi"][0] - cr["phi"][1], 0.8, atol=1e-6)


def test_sde_math_reduces_to_ode_and_logprob_is_consistent():
    torch.manual_seed(0)
    x = torch.randn(2, 16, 4)
    v = torch.randn(2, 16, 4)
    sig = [1.0, 0.8, 0.5, 0.2, 0.0]
    x_ode = ode_step(x, v, sig[1], sig[2])
    x_sde0, mean0, std0 = sde_step(x, v, sig[1], sig[2], sig[1], 0.0, noise=torch.zeros_like(x))
    assert torch.allclose(x_ode, x_sde0, atol=1e-6) and std0 == 0.0
    dt, sd, step_std = sde_coeffs(1.0, 0.8, 0.8, 0.7)
    assert dt == pytest_approx(-0.2) and sd == pytest_approx(0.7 * math.sqrt(1.0 / 0.2))
    x1, mean, step_std = sde_step(x, v, sig[1], sig[2], sig[1], 0.7, generator=torch.Generator().manual_seed(1))
    lp = token_logprob(x1, mean, step_std)
    assert lp.shape == (2, 16)
    lp2 = token_logprob(mean, mean, step_std)
    assert (lp2 > lp).all()
    m2 = sde_mean(x, v, sig[1], sig[2], sig[1], 0.7)
    assert torch.allclose(m2, mean)


def pytest_approx(v, tol=1e-9):
    class A:
        def __eq__(self, o):
            return abs(o - v) < tol

    return A()


def test_region_ratio_loss_gradient_direction():
    torch.manual_seed(0)
    B, N, R = 2, 40, 3
    lp_old = torch.randn(B, N) * 0.01
    theta = torch.zeros(B, N, requires_grad=True)
    lp_new = lp_old + theta
    rid = torch.tensor([[0] * 10 + [1] * 20 + [2] * 10] * B)
    adv = torch.tensor([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    w = torch.full((B, R), 1.0 / 3)
    loss, st = region_ratio_loss(lp_new, lp_old, rid, adv, w, eps_lo=0.5, eps_hi=0.5)
    loss.backward()
    g = theta.grad
    assert (g[0, :10] < 0).all() and (g[0, 10:30] > 0).all()
    assert (g[1, :10] > 0).all() and (g[1, 10:30] < 0).all()
    assert torch.allclose(g[:, 30:], torch.zeros(B, 10))
    assert st["regions"] == 6


if __name__ == "__main__":
    test_region_credit_separates_target_and_damage()
    test_sde_math_reduces_to_ode_and_logprob_is_consistent()
    test_region_ratio_loss_gradient_direction()
    print("credit/grpo tests ok")
