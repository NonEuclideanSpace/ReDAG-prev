from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from test0 import svg as S
from test0.attribution import attribute
from test0.checks import check_fragment, issues_from_checks
from test0.dag import ACTION, DAG, DRAFT_TYPES, Node
from test0.executors import build_draft_prompt, draft_system_prompt, normalize_fragment
from test0.pipeline import write_json
from test0.scheduler import Scheduler

from .actions import ACTION_SCHEMA, Action, apply_action
from .reward import attributed_rewards, local_reward, scene_score, weight

EDIT_SYSTEM = """\
You are the editing policy of DraftDAG. A scene is drawn as a DAG of nodes (background, entities, objects, actions, one scene layout node and one final generation node). You receive the user prompt, the current DAG, the latest review with every issue attributed to node ids, the cost spent so far and cached fragments that could be reused. Decide ONE next editing action that fixes the most important remaining problem with the least re-drawing, or stop when the draft satisfies the prompt.
""" + ACTION_SCHEMA


@dataclass
class Episode:
    tag: str
    prompt: str
    dag: DAG
    scheduler: Scheduler
    run_dir: str
    hint: Optional[Dict[str, Any]] = None
    drafts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scene_review: Optional[Dict[str, Any]] = None
    final_review: Optional[Dict[str, Any]] = None
    blamed_scene: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    blamed_final: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    edits: List[Dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def scene_passed(self) -> Optional[bool]:
        return None if self.scene_review is None else bool(self.scene_review.get("passed"))

    def final_passed(self) -> Optional[bool]:
        return None if self.final_review is None else bool(self.final_review.get("passed"))


class DraftDAGEnv:
    def __init__(self, cfg: Any, backends: Any, run_root: str, use_vlm_local: bool = False):
        self.cfg = cfg
        self.b = backends
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.use_vlm_local = bool(use_vlm_local)
        self.counter = 0

    def plan(self, prompt: str, hint: Optional[Dict[str, Any]] = None) -> DAG:
        self.b.cost.phase = "plan"
        catalog = self.b.store.catalog() if self.b.store is not None else []
        return self.b.planner.plan(prompt, catalog, hint)

    def reset(self, prompt: str, hint: Optional[Dict[str, Any]] = None, tag: str = "", dag: Optional[DAG] = None) -> Episode:
        self.counter += 1
        tag = tag or f"ep{self.counter:05d}"
        run_dir = self.run_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        self.b.cost.events.clear()
        if dag is None:
            dag = self.plan(prompt, hint)
        if hasattr(self.b.reviewer, "reference_dag"):
            self.b.reviewer.reference_dag = dag.clone()
        scheduler = Scheduler(dag, self.b, self.cfg, str(run_dir))
        self.b.cost.phase = "initial"
        return Episode(tag=tag, prompt=prompt, dag=dag, scheduler=scheduler, run_dir=str(run_dir), hint=hint)

    def draft_order(self, ep: Episode, only_dirty: bool = True) -> List[str]:
        out = []
        for nid in ep.dag.topological():
            node = ep.dag[nid]
            if node.type not in DRAFT_TYPES:
                continue
            if only_dirty and not node.dirty and node.output:
                continue
            out.append(nid)
        return out

    def draft_prompt(self, ep: Episode, node_id: str) -> Tuple[str, str]:
        node = ep.dag[node_id]
        deps = ep.scheduler.deps_outputs(node)
        return draft_system_prompt(node.type), build_draft_prompt(node, ep.dag, deps, node.feedback)

    def draft_context(self, ep: Episode, node_id: str) -> Dict[str, Any]:
        node = ep.dag[node_id]
        return {"node": node, "dag": ep.dag, "deps": ep.scheduler.deps_outputs(node), "feedback": list(node.feedback), "backends": self.b}

    def evaluate_draft(self, ep: Episode, node_id: str, text: str, tag: str = "s") -> Dict[str, Any]:
        node = ep.dag[node_id]
        try:
            fragment = normalize_fragment(text, node)
        except (ValueError, Exception):
            fragment = str(text)
        checks = check_fragment(fragment, node, ep.dag)
        png = None
        review = None
        if not checks["fatal"]:
            png = ep.scheduler.render_fragment(node, fragment, tag)
            if self.use_vlm_local and checks["passed"]:
                review = self.b.reviewer.node(node, fragment, png, ep.dag)
        local = local_reward(checks, review, self.cfg)
        return {"fragment": fragment, "checks": checks, "review": review, "png": png, "local": local}

    def commit_draft(self, ep: Episode, node_id: str, evaluation: Dict[str, Any], sample_meta: Optional[Dict[str, Any]] = None) -> None:
        node = ep.dag[node_id]
        checks = evaluation["checks"]
        review = evaluation.get("review")
        node.output = evaluation["fragment"]
        node.png = evaluation.get("png")
        node.checks = checks
        node.review = review or {"passed": checks["passed"], "issues": issues_from_checks(checks, node_id), "model": "checks"}
        node.source = "policy"
        node.store_key = ep.scheduler.store_key(node)
        node.dirty = False
        node.version += 1
        node.feedback = []
        passed = bool(checks["passed"] and (review is None or review.get("passed")))
        if passed and self.b.store is not None:
            self.b.store.put(node.store_key, node, node.output, node.png, checks, review)
        ep.drafts[node_id] = {"local": evaluation["local"], "passed": passed, "version": node.version, "meta": sample_meta or {}}
        ep.scheduler.reuse["demanded"] += 1
        ep.scheduler.reuse["executed"] += 1

    def run_scene(self, ep: Episode) -> Dict[str, Any]:
        scene_id = ep.dag.scene_id()
        if ep.dag[scene_id].dirty:
            ep.scheduler.execute_scene()
        review = ep.scheduler.review_scene()
        ep.scene_review = review
        ep.blamed_scene = attribute(review.get("issues", []), ep.dag, "scene")
        return review

    def run_final(self, ep: Episode) -> Dict[str, Any]:
        final_id = ep.dag.final_id()
        if ep.dag[final_id].dirty:
            ep.scheduler.execute_final()
        review = ep.scheduler.review_final()
        ep.final_review = review
        ep.blamed_final = attribute(review.get("issues", []), ep.dag, "final")
        return review

    def node_rewards(self, ep: Episode, include_final: bool = True) -> Dict[str, Dict[str, float]]:
        attributed = attributed_rewards(ep.dag, ep.blamed_scene, ep.scene_passed(), ep.blamed_final if include_final else None, ep.final_passed() if include_final else None, self.cfg)
        out: Dict[str, Dict[str, float]] = {}
        for n in ep.dag.draft_nodes():
            local = float(ep.drafts.get(n.id, {}).get("local", {}).get("value", local_reward(n.checks or {"fatal": True, "score": 0.0, "items": []}, None, self.cfg)["value"]))
            a = float(attributed.get(n.id, 0.0))
            out[n.id] = {"local": local, "attributed": a, "total": local + a}
        return out

    def subgraph_value(self, ep: Episode, node_ids: Optional[Set[str]] = None) -> float:
        rewards = self.node_rewards(ep, include_final=False)
        ids = [nid for nid in (node_ids if node_ids is not None else rewards.keys()) if nid in rewards]
        value = sum(rewards[nid]["total"] for nid in ids)
        return value + scene_score(ep.scene_review, self.cfg)

    def store_candidates(self, ep: Episode, node_ids: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        out = []
        if self.b.store is None:
            return out
        for nid in node_ids:
            if nid not in ep.dag or ep.dag[nid].type not in DRAFT_TYPES:
                continue
            for cand in self.b.store.candidates(ep.dag[nid], limit=limit):
                out.append({"node": nid, "key": cand["key"], "desc": cand.get("desc"), "spec": cand.get("spec")})
        return out

    def edit_prompt(self, ep: Episode, stage: str) -> Tuple[str, str]:
        review = ep.final_review if stage == "final" and ep.final_review is not None else ep.scene_review
        blamed = ep.blamed_final if stage == "final" and ep.final_review is not None else ep.blamed_scene
        issues = []
        for nid, items in blamed.items():
            for i in items:
                issues.append({"attributed_to": nid, "type": i.get("type"), "severity": i.get("severity"), "description": i.get("description"), "fix_instruction": i.get("fix_instruction")})
        candidates = self.store_candidates(ep, [nid for nid in blamed.keys()])
        cost = self.b.cost.summary()
        lines = [
            f"USER PROMPT: {ep.prompt}",
            "CURRENT DAG:",
            ep.dag.summary(),
            f"LATEST {stage.upper()} REVIEW: passed={bool(review and review.get('passed'))}",
            f"ISSUES WITH ATTRIBUTION: {json.dumps(issues, ensure_ascii=False)}",
            f"COST SO FAR: units={cost['units']:.1f} calls={json.dumps(cost['counts'])}",
            f"EDITS ALREADY APPLIED: {json.dumps([e.get('op') for e in ep.edits], ensure_ascii=False)}",
            f"REUSABLE STORE CANDIDATES: {json.dumps(candidates, ensure_ascii=False)}",
            "Return the action JSON now.",
        ]
        return EDIT_SYSTEM, "\n".join(lines)

    def edit_context(self, ep: Episode, stage: str) -> Dict[str, Any]:
        return {"dag": ep.dag, "review": ep.final_review if stage == "final" and ep.final_review is not None else ep.scene_review, "blamed": ep.blamed_final if stage == "final" and ep.final_review is not None else ep.blamed_scene, "store": self.b.store, "backends": self.b, "stage": stage, "episode": ep}

    def apply_edit(self, ep: Episode, action: Action) -> Dict[str, Any]:
        try:
            op, affected = apply_action(action, ep.dag, self.b.store)
        except (ValueError, KeyError, TypeError) as exc:
            info = {"ok": False, "error": str(exc)[:300], "op": action.name, "affected": []}
            ep.edits.append(info)
            return info
        info = {"ok": True, "op": op.describe(), "affected": sorted(affected), "stop": op.name == "stop"}
        ep.edits.append(info)
        ep.scheduler.repairs.append({"stage": "policy", "ops": [op.describe()], "affected": sorted(affected), "issues": 0})
        ep.scheduler._pending_repair = True
        return info

    def affected_drafts(self, ep: Episode) -> List[str]:
        return self.draft_order(ep, only_dirty=True)

    def result(self, ep: Episode) -> Dict[str, Any]:
        result = ep.scheduler.result()
        result["run_dir"] = ep.run_dir
        result["seconds"] = round(time.time() - ep.started, 3)
        result["status"] = "completed"
        result["edits"] = ep.edits
        result["node_rewards"] = self.node_rewards(ep)
        write_json(Path(ep.run_dir) / "result.json", result)
        return result
