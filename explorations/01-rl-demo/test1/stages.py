from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from test0.backends import Backends, build_backends
from test0.bench import run_bench
from test0.bench_cases import CASES, case_by_id
from test0.config import Config
from test0.dag import DAG, Node
from test0.executors import build_draft_prompt, draft_system_prompt, normalize_fragment
from test0.pipeline import write_json
from test0.scheduler import Op, Scheduler
from test0.store import Store

from .actions import Action, StopOp, op_from_action, parse_action
from .env import DraftDAGEnv
from .policy import Policy, build_policy
from .rollout import rollout_scene, rollout_stage1

STAGES: Dict[int, Dict[str, Any]] = {
    1: {"name": "local", "attributed": False, "final": False, "edits": 0},
    2: {"name": "attributed", "attributed": True, "final": True, "edits": 0},
    3: {"name": "edits", "attributed": True, "final": True, "edits": None},
}


class PolicyDrafter:
    def __init__(self, policy: Policy, backends: Backends):
        self.policy = policy
        self.b = backends

    def draft(self, node: Node, dag: DAG, deps: Dict[str, str], feedback: Sequence[str] = (), phase: Optional[str] = None) -> str:
        system = draft_system_prompt(node.type)
        user = build_draft_prompt(node, dag, deps, feedback)
        sample = self.policy.sample(system, user, 1, "draft", {"node": node, "dag": dag, "deps": deps, "feedback": list(feedback), "backends": self.b})[0]
        try:
            return normalize_fragment(sample.text, node)
        except (ValueError, Exception):
            return sample.text


class PolicyEditor:
    def __init__(self, policy: Policy, backends: Backends, env: DraftDAGEnv):
        self.policy = policy
        self.b = backends
        self.env = env
        self.log: List[Dict[str, Any]] = []

    def __call__(self, scheduler: Scheduler, review: Dict[str, Any], blamed: Dict[str, List[Dict[str, Any]]], stage: str) -> List[Op]:
        from .env import Episode

        ep = Episode(tag="eval", prompt=scheduler.dag.prompt, dag=scheduler.dag, scheduler=scheduler, run_dir=str(scheduler.run_dir))
        if stage == "final":
            ep.final_review = review
            ep.blamed_final = blamed
        else:
            ep.scene_review = review
            ep.blamed_scene = blamed
        system, user = self.env.edit_prompt(ep, stage)
        context = self.env.edit_context(ep, stage)
        sample = self.policy.sample(system, user, 1, "edit", context)[0]
        try:
            action = parse_action(sample.text)
            op = op_from_action(action, scheduler.dag, self.b.store)
        except (ValueError, TypeError, KeyError) as exc:
            self.log.append({"stage": stage, "action": "invalid", "error": str(exc)[:200], "fallback": "default"})
            return scheduler.default_ops(review, blamed, stage)
        self.log.append({"stage": stage, "action": action.name, "reason": action.reason})
        if isinstance(op, StopOp):
            return []
        return [op]


def stage_config(stage: int, cfg: Config) -> Dict[str, Any]:
    spec = dict(STAGES[int(stage)])
    if spec["edits"] is None:
        spec["edits"] = int(cfg.get("TRAIN_MAX_EDITS", 2) or 2)
    return spec


def training_prompts(case_ids: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    cases = [case_by_id(c) for c in case_ids] if case_ids else list(CASES)
    return [{"id": c["id"], "prompt": c["prompt"], "hint": c.get("hint")} for c in cases]


def train(stage: int, cfg: Config, policy: Policy, backends: Backends, iterations: int, group: int, out_dir: str, case_ids: Optional[Sequence[str]] = None, use_hint: bool = True, log=print) -> Dict[str, Any]:
    spec = stage_config(stage, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = DraftDAGEnv(cfg, backends, str(out / "rollouts"), use_vlm_local=bool(cfg.get("TRAIN_LOCAL_VLM", False)))
    prompts = training_prompts(case_ids)
    history: List[Dict[str, Any]] = []
    started = time.time()
    for it in range(int(iterations)):
        item = prompts[it % len(prompts)]
        tag = f"s{stage}-it{it:04d}-{item['id']}"
        hint = item["hint"] if use_hint else None
        if not spec["attributed"]:
            rollout = rollout_stage1(env, policy, item["prompt"], hint, group, tag)
        else:
            rollout = rollout_scene(env, policy, item["prompt"], hint, group, tag, with_final=bool(spec["final"]), edits=int(spec["edits"]), cfg=cfg)
        stats = policy.update(rollout["train"])
        entry = {"iteration": it, "case": item["id"], "stage": stage, "samples": len(rollout["train"]), "metrics": rollout["metrics"], "update": {k: v for k, v in stats.items() if k != "tables"}, "seconds": round(time.time() - started, 1)}
        history.append(entry)
        log(json.dumps({"it": it, "case": item["id"], **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in rollout["metrics"].items()}, "loss": stats.get("loss")}, ensure_ascii=False))
        write_json(out / "history.json", history)
        if "tables" in stats:
            write_json(out / "policy_tables.json", stats["tables"])
        with (out / "samples.jsonl").open("a", encoding="utf-8") as handle:
            for t in rollout["train"]:
                row = t.to_dict()
                row["prompt"] = row["prompt"][:1500]
                row["completion"] = row["completion"][:2500]
                handle.write(json.dumps({"tag": tag, **row}, ensure_ascii=False, default=str) + "\n")
    policy.save(str(out / "policy"))
    return {"history": history, "out_dir": str(out), "stage": stage}


def evaluate(stage: int, cfg: Config, policy: Policy, backends: Backends, out_dir: str, case_ids: Optional[Sequence[str]] = None, faults: int = 0, systems: Sequence[str] = ("dag",), with_baseline: bool = False) -> Dict[str, Any]:
    spec = stage_config(stage, cfg)
    env = DraftDAGEnv(cfg, backends, str(Path(out_dir) / "editor"))
    drafter = PolicyDrafter(policy, backends)
    editor = PolicyEditor(policy, backends, env) if spec["edits"] else None
    chosen = list(systems) + (["baseline"] if with_baseline and "baseline" not in systems else [])
    report = run_bench(cfg, case_ids=case_ids, systems=chosen, faults=faults, out_dir=out_dir, backends=backends, drafter=drafter, editor=editor, quiet=True)
    report["editor_log"] = editor.log if editor else []
    return report


def curriculum(cfg: Config, backends: Backends, policy: Policy, iterations: Sequence[int], group: int, out_dir: str, case_ids: Optional[Sequence[str]] = None, eval_cases: Optional[Sequence[str]] = None, log=print) -> Dict[str, Any]:
    out = Path(out_dir)
    summary: Dict[str, Any] = {"stages": {}}
    base = evaluate(1, cfg, policy, backends, str(out / "eval-stage0"), case_ids=eval_cases)
    summary["stages"]["0"] = {"eval": base["summary"].get("dag")}
    log("stage 0 (untrained): " + json.dumps({k: base["summary"]["dag"].get(k) for k in ("judge_pass_rate", "scene_pass_rate", "cost_total", "cost_repair")}))
    for stage, iters in zip((1, 2, 3), iterations):
        result = train(stage, cfg, policy, backends, iters, group, str(out / f"train-stage{stage}"), case_ids=case_ids, log=log)
        report = evaluate(stage, cfg, policy, backends, str(out / f"eval-stage{stage}"), case_ids=eval_cases)
        summary["stages"][str(stage)] = {"train": result["history"][-1] if result["history"] else None, "eval": report["summary"].get("dag")}
        log(f"stage {stage}: " + json.dumps({k: report["summary"]["dag"].get(k) for k in ("judge_pass_rate", "scene_pass_rate", "cost_total", "cost_repair")}))
    write_json(out / "curriculum.json", summary)
    return summary
