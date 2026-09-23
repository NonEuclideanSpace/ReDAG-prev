from __future__ import annotations

from typing import Any, Dict, List, Optional

from test0.dag import ACTION, BACKGROUND, DAG, DRAFT_TYPES

REWARD_DEFAULTS = {
    "REWARD_LOCAL_SYNTAX": 1.0,
    "REWARD_LOCAL_CHECKS": 1.0,
    "REWARD_LOCAL_POSE": 0.5,
    "REWARD_LOCAL_VLM": 0.5,
    "REWARD_SCENE_PASS": 1.0,
    "REWARD_FINAL_PASS": 1.0,
    "REWARD_ISSUE_MAJOR": 1.0,
    "REWARD_ISSUE_MINOR": 0.3,
    "REWARD_EDIT_COST_LAMBDA": 0.05,
    "REWARD_INVALID_ACTION": -0.5,
}


def weight(cfg: Any, key: str) -> float:
    value = cfg.get(key) if hasattr(cfg, "get") else None
    if value is None:
        value = REWARD_DEFAULTS[key]
    return float(value)


def local_reward(checks: Dict[str, Any], review: Optional[Dict[str, Any]], cfg: Any) -> Dict[str, Any]:
    w_syntax = weight(cfg, "REWARD_LOCAL_SYNTAX")
    w_checks = weight(cfg, "REWARD_LOCAL_CHECKS")
    w_pose = weight(cfg, "REWARD_LOCAL_POSE")
    w_vlm = weight(cfg, "REWARD_LOCAL_VLM")
    syntax = 0.0 if checks.get("fatal") else 1.0
    check_rate = float(checks.get("score") or 0.0) if syntax else 0.0
    pose_item = next((i for i in checks.get("items", []) if i["name"] == "pose"), None)
    pose = 1.0 if (pose_item is None or pose_item.get("skipped") or pose_item.get("passed")) else 0.0
    if not syntax:
        pose = 0.0
    total_w = w_syntax + w_checks + w_pose
    value = w_syntax * syntax + w_checks * check_rate + w_pose * pose
    vlm = None
    if review is not None and review.get("model") not in (None, "checks"):
        vlm = 1.0 if review.get("passed") else 0.0
        value += w_vlm * vlm
        total_w += w_vlm
    return {"value": value / total_w if total_w else 0.0, "syntax": syntax, "checks": check_rate, "pose": pose, "vlm": vlm}


def _issue_penalty(issue: Dict[str, Any], cfg: Any) -> float:
    return weight(cfg, "REWARD_ISSUE_MAJOR") if str(issue.get("severity", "major")) == "major" else weight(cfg, "REWARD_ISSUE_MINOR")


def attributed_rewards(dag: DAG, blamed_scene: Optional[Dict[str, List[Dict[str, Any]]]], scene_passed: Optional[bool], blamed_final: Optional[Dict[str, List[Dict[str, Any]]]], final_passed: Optional[bool], cfg: Any) -> Dict[str, float]:
    nodes = [n.id for n in dag.draft_nodes()]
    out = {nid: 0.0 for nid in nodes}
    if not nodes:
        return out
    if scene_passed is not None:
        if scene_passed:
            bonus = weight(cfg, "REWARD_SCENE_PASS") / len(nodes)
            for nid in nodes:
                out[nid] += bonus
        for nid, issues in (blamed_scene or {}).items():
            if nid in out:
                out[nid] -= sum(_issue_penalty(i, cfg) for i in issues)
    if final_passed is not None:
        if final_passed:
            bonus = weight(cfg, "REWARD_FINAL_PASS") / len(nodes)
            for nid in nodes:
                out[nid] += bonus
        for nid, issues in (blamed_final or {}).items():
            if nid in out:
                out[nid] -= sum(_issue_penalty(i, cfg) for i in issues)
    return out


def total_reward(local: float, attributed: float) -> float:
    return float(local) + float(attributed)


def scene_score(review: Optional[Dict[str, Any]], cfg: Any) -> float:
    if review is None:
        return 0.0
    if review.get("passed"):
        return weight(cfg, "REWARD_SCENE_PASS")
    return -sum(_issue_penalty(i, cfg) for i in review.get("issues", []))
