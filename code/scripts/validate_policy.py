"""GPU validation of the training adapter before any GRPO run. Three checks, all must pass:
1. ODE path of QI21Policy reproduces QwenImage21Pipeline output for the same seed and schedule (no KV cache, bf16):
   mean absolute pixel difference small (< 0.02 in [0,1]) and visually identical.
2. SDE rollout with noise_level 0 equals the ODE rollout exactly.
3. Log-prob recomputation under unchanged parameters reproduces the rollout log-probs: per-token ratio exp(new-old)
   within 1e-3 of 1 for every step.
Writes gates/policy_qi21.json."""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from common import P0

from redag.executor_qi21 import load_pipe
from redag.paths import gates_dir
from redag.policy_qi21 import QI21Policy

ap = argparse.ArgumentParser()
ap.add_argument("--canvas", required=True, help="an image to edit")
ap.add_argument("--instruction", default="Add a small red apple on the table. Keep everything else in the image unchanged.")
ap.add_argument("--res", type=int, default=768)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--seed", type=int, default=3)
args = ap.parse_args()

pipe, device = load_pipe()
pol = QI21Policy(pipe, device, res=args.res, steps=args.steps, noise_level=0.7)
canvas = Image.open(args.canvas).convert("RGB")
cond = pol.encode(args.instruction, canvas)
out = dict(res=args.res, steps=args.steps, grid=cond["grid"], n_tokens=cond["n_tokens"], sigmas=cond["sigmas"])

with torch.no_grad():
    g = torch.Generator(device=device).manual_seed(args.seed)
    x0 = torch.randn((1, cond["n_tokens"], pol.in_channels), generator=g, device=device, dtype=torch.float32)
    lat = x0.to(cond["pe"].dtype)
    ref = pipe(prompt=args.instruction, image=canvas, num_inference_steps=args.steps, true_cfg_scale=1.0, num_images_per_prompt=1, latents=lat, output_resolution=args.res, use_kv_cache=False, sigmas=list(np.linspace(1.0, 1.0 / args.steps, args.steps))).images[0].convert("RGB").resize(canvas.size)
    x = x0.clone()
    sig = cond["sigmas"]
    for i in range(len(sig) - 1):
        v = pol._pred(cond, x, sig[i])
        x = x + v * (sig[i + 1] - sig[i])
    mine = pol.decode(x, cond)[0]
    a = np.asarray(ref, dtype=np.float32) / 255.0
    b = np.asarray(mine, dtype=np.float32) / 255.0
    out["ode_vs_pipeline_mad"] = float(np.abs(a - b).mean())
    ref.save(os.path.join(gates_dir(), "policy_check_pipeline.png"))
    mine.save(os.path.join(gates_dir(), "policy_check_ode.png"))

    pol.noise_level = 0.0
    t_ode = pol.rollout(cond, 1, args.seed)
    pol.noise_level = 0.7
    t_sde = pol.rollout(cond, 2, args.seed)
    x_ode_direct = x0.clone()
    for i in range(len(sig) - 1):
        x_ode_direct = x_ode_direct + pol._pred(cond, x_ode_direct, sig[i]) * (sig[i + 1] - sig[i])
    out["sde0_vs_ode_max"] = float((t_ode["latents"][-1][0].float().to(device) - x_ode_direct[0].cpu().float().to(device)).abs().max())

ratios = []
for t in range(len(sig) - 1):
    with torch.no_grad():
        lp_new = pol.logprob(cond, t_sde["latents"][t], t_sde["latents"][t + 1], t, sig)
    lp_old = t_sde["logp"][t].to(device)
    r = torch.exp(lp_new - lp_old)
    ratios.append(dict(step=t, max_dev=float((r - 1).abs().max()), mean=float(r.mean())))
out["logprob_ratio"] = ratios
out["pass"] = out["ode_vs_pipeline_mad"] < 0.02 and out["sde0_vs_ode_max"] < 1e-3 and all(r["max_dev"] < 1e-3 for r in ratios)
with open(os.path.join(gates_dir(), "policy_qi21.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
