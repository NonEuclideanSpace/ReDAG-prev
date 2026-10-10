from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from test0.dag import DAG

from .actions import Action, parse_action
from .advantage import clip, edit_advantage, group_advantages
from .env import DraftDAGEnv, Episode
from .policy import Policy, Sample, TrainSample
from .reward import weight


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def draft_group(env: DraftDAGEnv, policy: Policy, ep: Episode, node_id: str, group: int) -> Tuple[List[Sample], List[Dict[str, Any]]]:
    system, user = env.draft_prompt(ep, node_id)
    context = env.draft_context(ep, node_id)
    samples = policy.sample(system, user, group, "draft", context)
    evaluations = [env.evaluate_draft(ep, node_id, s.text, tag=f"g{k}") for k, s in enumerate(samples)]
    return samples, evaluations


def rollout_stage1(env: DraftDAGEnv, policy: Policy, prompt: str, hint: Optional[Dict[str, Any]], group: int, tag: str) -> Dict[str, Any]:
    ep = env.reset(prompt, hint, tag=tag)
    train: List[TrainSample] = []
    node_rewards: List[float] = []
    for node_id in env.draft_order(ep):
        samples, evaluations = draft_group(env, policy, ep, node_id, group)
        rewards = [e["local"]["value"] for e in evaluations]
        advantages = group_advantages(rewards)
        for s, e, r, a in zip(samples, evaluations, rewards, advantages):
            train.append(TrainSample(prompt=s.prompt, completion=s.text, advantage=clip(a), reward=r, kind="draft", group=f"{tag}:{node_id}", meta=dict(s.meta)))
        best = max(range(len(rewards)), key=lambda i: rewards[i])
        env.commit_draft(ep, node_id, evaluations[best], samples[best].meta)
        node_rewards.append(rewards[best])
    scene_review = env.run_scene(ep)
    result = env.result(ep)
    return {"train": train, "metrics": {"mean_local": _mean([t.reward for t in train]), "best_local": _mean(node_rewards), "scene_passed": bool(scene_review.get("passed")), "cost": result["cost"]["units"]}, "episodes": [ep]}


def _draft_all(env: DraftDAGEnv, policy: Policy, ep: Episode, record: Dict[str, List[Dict[str, Any]]]) -> None:
    for node_id in env.draft_order(ep):
        samples, evaluations = draft_group(env, policy, ep, node_id, 1)
        env.commit_draft(ep, node_id, evaluations[0], samples[0].meta)
        record.setdefault(node_id, []).append({"sample": samples[0], "local": evaluations[0]["local"]["value"], "episode": ep})


def rollout_scene(env: DraftDAGEnv, policy: Policy, prompt: str, hint: Optional[Dict[str, Any]], group: int, tag: str, with_final: bool, edits: int, cfg: Any) -> Dict[str, Any]:
    base = env.plan(prompt, hint)
    episodes: List[Episode] = []
    per_node: Dict[str, List[Dict[str, Any]]] = {}
    train: List[TrainSample] = []
    edit_stats: List[Dict[str, Any]] = []
    scene_passes = 0
    final_passes = 0
    lam = weight(cfg, "REWARD_EDIT_COST_LAMBDA")
    for k in range(group):
        ep = env.reset(prompt, hint, tag=f"{tag}-r{k}", dag=base.clone())
        episodes.append(ep)
        records: Dict[str, List[Dict[str, Any]]] = {}
        _draft_all(env, policy, ep, records)
        env.run_scene(ep)
        if with_final:
            env.run_final(ep)
        rewards = env.node_rewards(ep, include_final=with_final)
        for node_id, items in records.items():
            for item in items:
                total = rewards.get(node_id, {}).get("total", item["local"])
                per_node.setdefault(node_id, []).append({"sample": item["sample"], "reward": total, "local": item["local"]})
        if ep.scene_passed():
            scene_passes += 1
        if with_final and ep.final_passed():
            final_passes += 1
        if edits > 0:
            train.extend(edit_phase(env, policy, ep, edits, lam, edit_stats, "final" if with_final else "scene"))
        env.result(ep)
    for node_id, items in per_node.items():
        advantages = group_advantages([i["reward"] for i in items])
        for item, a in zip(items, advantages):
            s = item["sample"]
            train.append(TrainSample(prompt=s.prompt, completion=s.text, advantage=clip(a), reward=item["reward"], kind="draft", group=f"{tag}:{node_id}", meta=dict(s.meta)))
    metrics = {
        "mean_total": _mean([t.reward for t in train if t.kind == "draft"]),
        "mean_local": _mean([i["local"] for items in per_node.values() for i in items]),
        "scene_pass_rate": scene_passes / float(group),
        "final_pass_rate": (final_passes / float(group)) if with_final else None,
        "edit_steps": len(edit_stats),
        "edit_mean_advantage": _mean([e["advantage"] for e in edit_stats]),
        "edit_invalid": sum(1 for e in edit_stats if not e["ok"]),
        "edit_stop_rate": (sum(1 for e in edit_stats if e["action"] == "stop") / len(edit_stats)) if edit_stats else None,
        "cost": _mean([env.b.cost.units()]),
    }
    return {"train": train, "metrics": metrics, "episodes": episodes}


def edit_phase(env: DraftDAGEnv, policy: Policy, ep: Episode, max_edits: int, lam: float, stats: List[Dict[str, Any]], stage: str) -> List[TrainSample]:
    train: List[TrainSample] = []
    for step in range(max_edits):
        system, user = env.edit_prompt(ep, stage)
        context = env.edit_context(ep, stage)
        sample = policy.sample(system, user, 1, "edit", context)[0]
        before_ids = {n.id for n in ep.dag.draft_nodes()}
        before = env.subgraph_value(ep)
        mark = env.b.cost.mark()
        try:
            action = parse_action(sample.text)
        except (ValueError, TypeError) as exc:
            adv = weight(env.cfg, "REWARD_INVALID_ACTION")
            train.append(TrainSample(prompt=sample.prompt, completion=sample.text, advantage=adv, reward=adv, kind="edit", group=f"{ep.tag}:edit{step}", meta=dict(sample.meta)))
            stats.append({"ok": False, "action": "invalid", "advantage": adv, "error": str(exc)[:120]})
            continue
        info = env.apply_edit(ep, action)
        if not info["ok"]:
            adv = weight(env.cfg, "REWARD_INVALID_ACTION")
            train.append(TrainSample(prompt=sample.prompt, completion=sample.text, advantage=adv, reward=adv, kind="edit", group=f"{ep.tag}:edit{step}", meta=dict(sample.meta)))
            stats.append({"ok": False, "action": action.name, "advantage": adv, "error": info.get("error")})
            continue
        if info.get("stop"):
            train.append(TrainSample(prompt=sample.prompt, completion=sample.text, advantage=0.0, reward=0.0, kind="edit", group=f"{ep.tag}:edit{step}", meta=dict(sample.meta)))
            stats.append({"ok": True, "action": "stop", "advantage": 0.0, "affected": []})
            break
        affected_drafts = env.affected_drafts(ep)
        records: Dict[str, List[Dict[str, Any]]] = {}
        _draft_all(env, policy, ep, records)
        env.run_scene(ep)
        if stage == "final":
            env.run_final(ep)
        after_ids = {n.id for n in ep.dag.draft_nodes()}
        scope = (set(info["affected"]) | set(affected_drafts) | (after_ids ^ before_ids)) & (after_ids | before_ids)
        after = env.subgraph_value(ep)
        cost_units = env.b.cost.since(mark)
        adv = clip(edit_advantage(before, after, cost_units, lam))
        train.append(TrainSample(prompt=sample.prompt, completion=sample.text, advantage=adv, reward=after - before, kind="edit", group=f"{ep.tag}:edit{step}", meta=dict(sample.meta)))
        stats.append({"ok": True, "action": action.name, "advantage": adv, "delta": after - before, "cost": cost_units, "affected": sorted(scope)})
        if ep.scene_passed() and (stage != "final" or ep.final_passed()):
            break
    return train
