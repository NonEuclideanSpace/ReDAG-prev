# 01-rl-demo Project Overview

This directory contains the first exploration under `draft-dag`: two experimental implementations based on `DraftDAG-Plan-Demo.pdf`. `test0` fully implements the "Environment" column on the left of the plan, without training. `test1` adds the "Policy Optimization" column on the right. Both support the complete workflow locally in mock mode; API integration only requires changes to `config.yaml`.

## Directory Structure

```
test0/            Environment: planner → executors (node-by-node drafting) → store → render → review → compose → generate → final review
  dag.py          DAG data structures, validation, and dirty-state propagation
  scheduler.py    DAG execution, node/scene/final-image reviews, attribution, spec edits, incremental recomputation, and cost tracking
  attribution.py  Attribution of errors from all three review stages to nodes
  checks.py       Programmatic checks: syntax, bounding boxes, counts, colors, pose consistency, and layout relations
  store.py        Node cache with reuse based on spec hashes
  render.py       Playwright rendering with a pure PIL rasterization fallback
  compose.py      Assembly of node fragments into a complete scene draft according to the layout
  generate.py     Final-image generation client using the same protocol as the GenClaw FLUX server
  baseline.py     GenClaw-style baseline that drafts the entire scene in one shot
  bench.py        Complex-scene benchmark, fault-injection localization experiments, judging, and reporting
  bench_cases.py  16 benchmark cases with ground-truth DAGs and checklists
  mock*.py        Offline mocks for planning, drafting, layout, review, judging, and generation, with configurable error rates
test1/            Policy optimization
  actions.py      Full editing action set: edit_spec / split / merge / add / delete / reorder / reuse / adjust_final / stop
  env.py          RL environment wrapping the scheduler: drafting/editing observations and local/attributed rewards
  reward.py       r_i (syntax + check pass rate + pose consistency [+ VLM]), a_i (attributed scene/final-image review rewards), R_i = r_i + a_i
  advantage.py    Within-group mean baseline for drafting; affected-subgraph reward improvement minus weighted cost for editing
  policy.py       mock (tabular policy with offline learning) / api (OpenAI-compatible) / hf (Qwen3-8B + LoRA)
  grpo.py         GRPO loss and updates: clipped probability ratios and k3 KL
  rollout.py      Sampling for the three stages
  stages.py       Three-stage curriculum with each stage's contribution measured separately on the test0 benchmark
server/qwen_controlnet_server.py  Qwen-Image + ControlNet-Union generation worker; the mock backend requires no GPU
tests/            pytest coverage for the complete mock workflow, actions, rewards, and GRPO smoke tests with a small model
```

## Installation

```
pip install -r requirements.txt
playwright install chromium
pip install -r requirements-rl.txt
```

## Running Offline

```
python -m test0 --mode mock --error-rate 0.3 run --case girl_umbrella_cat
python -m test0 --mode mock --error-rate 0.3 bench --faults 2
python -m test1 --mode mock --policy mock --error-rate 0.3 curriculum --iters 10,6,6 --group 4
python -m pytest tests -q
```

## API Integration

Copy `config.example.yaml` to `config.yaml` and set `OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL_NAME`. Planning, drafting, and review share these settings; you can also specify a model for each role through `PLANNER_MODEL / EXECUTOR_MODEL / REVIEW_MODEL`. Configure the judge separately through `JUDGE_*`, then run:

```
python server/qwen_controlnet_server.py --backend qwen-controlnet --port 8188
python -m test0 --mode api --generate remote bench --faults 2
python -m test1 --mode api --policy hf --policy-model Qwen/Qwen3-8B train --stage 1 --iters 50 --group 4
```

Use `--mode api --generate mock` to test only the LLM/VLM components. Use `--policy api` to validate the action interface with the same gateway as the policy, without training.

## Metrics

The benchmark reports four core metrics: checklist pass rate (assessed by an independent judge), error localization (attribution hit rate and precision after injecting color/remove faults), repair cost (cost incurred during the repair phase after failed reviews and the proportion of nodes redrawn), and reuse rate (cache hits divided by node requests).
