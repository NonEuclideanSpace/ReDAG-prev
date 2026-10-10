import argparse
import json
import os

from common import P0, add_vlm_args, load_json, make_vlm, plan_dir

from redag.executor_qi21 import load_pipe
from redag.paths import gates_dir
from redag.policy_qi21 import QI21Policy
from redag.train import TrainConfig, Trainer, states_from_runs

ap = argparse.ArgumentParser(description="DAGFlow-GRPO on Qwen-Image-2.1 from the loop's own states.")
ap.add_argument("--runs", nargs="+", required=True, help="runs whose branches become training states")
ap.add_argument("--set", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--config", default=os.path.join(P0, "configs", "train.default.json"))
ap.add_argument("--warm-lora", default=None, help="optional warm-up LoRA to start from")
ap.add_argument("--min-states", type=int, default=200)
add_vlm_args(ap)
args = ap.parse_args()

g = os.path.join(gates_dir(), "policy_qi21.json")
if not (os.path.exists(g) and json.load(open(g)).get("pass")):
    raise SystemExit("policy adapter not validated: run scripts/validate_policy.py first")
cfg = TrainConfig(**load_json(args.config, {}))
out = os.path.join(P0, "runs", args.out)
os.makedirs(out, exist_ok=True)
states = states_from_runs([r if os.path.isabs(r) else os.path.join(P0, "runs", r) for r in args.runs], plan_dir(args.set, args.tag), os.path.join(out, "states.jsonl"))
ops = {}
for s in states:
    ops[s["target"]["op"]] = ops.get(s["target"]["op"], 0) + 1
print("states", len(states), "by op", ops, flush=True)
if len(states) < args.min_states or len([o for o in ops if ops[o] >= 20]) < 2:
    raise SystemExit("not enough states or op coverage; the data floor in idea v4 section 3.7 is not met")
vlm = make_vlm(args, require_gate=True)
pipe, device = load_pipe()
pol = QI21Policy(pipe, device, res=cfg.res, steps=cfg.steps, noise_level=cfg.noise_level)
tr = Trainer(pol, vlm, states, cfg, out)
if args.warm_lora:
    import safetensors.torch as st

    sd = st.load_file(args.warm_lora)
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(pol.transformer, sd)
with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
    json.dump(dict(cfg.to_dict(), args=vars(args)), f, ensure_ascii=False, indent=1)
tr.fit()
print("DONE", flush=True)
