from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from test0.checks import check_fragment
from test0.dag import ACTION, BACKGROUND, DAG, DRAFT_TYPES, ENTITY, FINAL, OBJECT, SCENE, Node
from test0.llm import parse_json
from test0.scheduler import AdjustFinal, EditSpec, Op

ACTIONS = ("edit_spec", "split", "merge", "add", "delete", "reorder", "reuse", "adjust_final", "stop")

ACTION_SCHEMA = """\
Output exactly ONE JSON object describing your next action (no prose):
{"action": "stop", "reason": "..."}
{"action": "edit_spec", "node": "<id>", "spec": {<full revised spec>}, "reason": "..."}
{"action": "split", "node": "<id>", "parts": [{"id": "<new id>", "type": "entity|object", "spec": {...}}, ...], "reason": "..."}
{"action": "merge", "nodes": ["<id>", "<id>"], "merged": {"id": "<new id>", "type": "entity|object", "spec": {...}}, "reason": "..."}
{"action": "add", "node": {"id": "<new id>", "type": "background|entity|object|action", "spec": {...}, "deps": []}, "reason": "..."}
{"action": "delete", "node": "<id>", "reason": "..."}
{"action": "reorder", "order": ["<placeable id in drawing order, back to front>", ...], "reason": "..."}
{"action": "reuse", "node": "<id>", "key": "<store key from the candidates list>", "reason": "..."}
{"action": "adjust_final", "reason": "..."}
Rules: node ids must exist (except new ids in split/merge/add). Every edit re-executes only the affected subgraph and costs budget; choose "stop" when the current draft already satisfies the prompt.
"""


@dataclass
class Action:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = {"action": self.name}
        out.update(self.args)
        if self.reason:
            out["reason"] = self.reason
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def parse_action(text: str) -> Action:
    data = parse_json(text)
    if not isinstance(data, dict):
        raise ValueError("action must be a JSON object")
    name = str(data.get("action") or data.get("name") or "").strip().lower()
    if name not in ACTIONS:
        raise ValueError(f"unknown action {name!r}")
    args = {k: v for k, v in data.items() if k not in ("action", "name", "reason")}
    return Action(name=name, args=args, reason=str(data.get("reason") or ""), raw=text)


def _new_node(item: Dict[str, Any], default_type: str = OBJECT) -> Node:
    nid = str(item.get("id") or "").strip()
    ntype = str(item.get("type") or default_type).strip().lower()
    spec = item.get("spec") if isinstance(item.get("spec"), dict) else {"desc": str(item.get("desc") or nid)}
    spec = dict(spec)
    spec.setdefault("desc", nid)
    if ntype in (ENTITY, OBJECT):
        spec["count"] = max(1, int(spec.get("count") or 1))
        spec.setdefault("colors", [])
        spec.setdefault("pose", "standing" if ntype == ENTITY else "none")
        spec.setdefault("facing", "front" if ntype == ENTITY else "none")
    deps = [str(d) for d in (item.get("deps") or [])]
    if ntype == ACTION:
        deps = list(dict.fromkeys(deps + [str(p) for p in (spec.get("participants") or [])]))
    return Node(id=nid, type=ntype, spec=spec, deps=deps)


def _replace_references(dag: DAG, old: str, new_ids: List[str]) -> None:
    for n in dag.nodes.values():
        if n.type == ACTION:
            parts = n.participants()
            if old in parts:
                idx = parts.index(old)
                parts = parts[:idx] + [p for p in new_ids if p not in parts] + parts[idx + 1:]
                n.spec["participants"] = parts
                poses = n.spec.get("poses") if isinstance(n.spec.get("poses"), dict) else {}
                if old in poses:
                    pose = poses.pop(old)
                    for p in new_ids:
                        poses.setdefault(p, pose)
                    n.spec["poses"] = poses
                n.deps = list(dict.fromkeys([d for d in n.deps if d != old] + parts))
        if n.type == SCENE:
            rels = []
            for rel in n.spec.get("relations") or []:
                if not isinstance(rel, dict):
                    continue
                if rel.get("a") == old or rel.get("b") == old:
                    for p in new_ids:
                        item = dict(rel)
                        if item.get("a") == old:
                            item["a"] = p
                        if item.get("b") == old:
                            item["b"] = p
                        if item["a"] != item["b"]:
                            rels.append(item)
                else:
                    rels.append(rel)
            n.spec["relations"] = rels
            order = n.spec.get("z_order")
            if isinstance(order, list) and old in order:
                idx = order.index(old)
                n.spec["z_order"] = order[:idx] + new_ids + order[idx + 1:]


def _commit(dag: DAG, snapshot: Dict[str, Any]) -> None:
    dag.normalize()
    errors = dag.validate()
    if errors:
        restored = DAG.from_dict(snapshot)
        dag.nodes = restored.nodes
        dag.canvas = restored.canvas
        raise ValueError("; ".join(errors))


class SplitOp(Op):
    name = "split"

    def __init__(self, node_id: str, parts: List[Dict[str, Any]], reason: str = ""):
        self.node_id = node_id
        self.parts = parts
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        snapshot = dag.to_dict()
        if self.node_id not in dag or dag[self.node_id].type not in (ENTITY, OBJECT, BACKGROUND):
            raise ValueError(f"cannot split {self.node_id!r}")
        if len(self.parts) < 2:
            raise ValueError("split needs at least two parts")
        original = dag[self.node_id]
        new_nodes = [_new_node(p, original.type) for p in self.parts]
        ids = [n.id for n in new_nodes]
        if len(set(ids)) != len(ids) or any((not i) or (i in dag and i != self.node_id) for i in ids):
            raise ValueError("split part ids must be new and unique")
        _replace_references(dag, self.node_id, ids)
        dag.remove(self.node_id)
        for n in new_nodes:
            dag.add(n)
        _commit(dag, snapshot)
        return dag.mark_dirty(ids)

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "node": self.node_id, "parts": [p.get("id") for p in self.parts], "reason": self.reason}


class MergeOp(Op):
    name = "merge"

    def __init__(self, node_ids: List[str], merged: Dict[str, Any], reason: str = ""):
        self.node_ids = node_ids
        self.merged = merged
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        snapshot = dag.to_dict()
        if len(self.node_ids) < 2:
            raise ValueError("merge needs at least two nodes")
        for nid in self.node_ids:
            if nid not in dag or dag[nid].type not in (ENTITY, OBJECT):
                raise ValueError(f"cannot merge {nid!r}")
        merged = _new_node(self.merged, dag[self.node_ids[0]].type)
        if not merged.id or (merged.id in dag and merged.id not in self.node_ids):
            raise ValueError("merged id must be new")
        for nid in self.node_ids:
            _replace_references(dag, nid, [merged.id])
        for nid in self.node_ids:
            dag.remove(nid)
        dag.add(merged)
        for n in dag.nodes.values():
            if n.type == ACTION:
                n.spec["participants"] = list(dict.fromkeys(n.participants()))
                n.deps = list(dict.fromkeys(n.deps))
        _commit(dag, snapshot)
        return dag.mark_dirty([merged.id])

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "nodes": list(self.node_ids), "merged": self.merged.get("id"), "reason": self.reason}


class AddOp(Op):
    name = "add"

    def __init__(self, node: Dict[str, Any], reason: str = ""):
        self.node = node
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        snapshot = dag.to_dict()
        node = _new_node(self.node)
        if not node.id or node.id in dag:
            raise ValueError("added node needs a new id")
        if node.type not in DRAFT_TYPES:
            raise ValueError(f"cannot add node of type {node.type!r}")
        if node.type == BACKGROUND and any(n.type == BACKGROUND for n in dag.nodes.values()):
            raise ValueError("scene already has a background")
        dag.add(node)
        _commit(dag, snapshot)
        return dag.mark_dirty([node.id])

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "node": self.node.get("id"), "type": self.node.get("type"), "reason": self.reason}


class DeleteOp(Op):
    name = "delete"

    def __init__(self, node_id: str, reason: str = ""):
        self.node_id = node_id
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        snapshot = dag.to_dict()
        if self.node_id not in dag or dag[self.node_id].type in (SCENE, FINAL):
            raise ValueError(f"cannot delete {self.node_id!r}")
        downstream = dag.downstream([self.node_id])
        dag.remove(self.node_id)
        for n in list(dag.nodes.values()):
            if n.type == ACTION and not n.participants():
                downstream |= dag.downstream([n.id])
                dag.remove(n.id)
        scene = dag[dag.scene_id()]
        scene.spec["relations"] = [r for r in (scene.spec.get("relations") or []) if isinstance(r, dict) and r.get("a") in dag and r.get("b") in dag]
        if isinstance(scene.spec.get("z_order"), list):
            scene.spec["z_order"] = [z for z in scene.spec["z_order"] if z in dag]
        _commit(dag, snapshot)
        return dag.mark_dirty([d for d in downstream if d in dag])

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "node": self.node_id, "reason": self.reason}


class ReorderOp(Op):
    name = "reorder"

    def __init__(self, order: List[str], reason: str = ""):
        self.order = [str(o) for o in order]
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        placeables = [n.id for n in dag.placeables()]
        unknown = [o for o in self.order if o not in placeables]
        if unknown:
            raise ValueError(f"reorder references non-placeable nodes {unknown}")
        scene = dag[dag.scene_id()]
        rest = [p for p in placeables if p not in self.order]
        scene.spec["z_order"] = self.order + rest
        return dag.mark_dirty([scene.id])

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "order": list(self.order), "reason": self.reason}


class ReuseOp(Op):
    name = "reuse"

    def __init__(self, node_id: str, key: str, store: Any, reason: str = ""):
        self.node_id = node_id
        self.key = key
        self.store = store
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        if self.node_id not in dag or dag[self.node_id].type not in DRAFT_TYPES:
            raise ValueError(f"cannot reuse into {self.node_id!r}")
        entry = self.store.get(self.key) if self.store is not None else None
        if entry is None:
            raise ValueError(f"store key {self.key!r} not found")
        node = dag[self.node_id]
        if entry.get("type") != node.type:
            raise ValueError("store entry type mismatch")
        fragment = entry["fragment_text"]
        node.output = fragment
        node.png = self.store.png_path(self.key)
        node.checks = check_fragment(fragment, node, dag)
        node.review = {"passed": bool(node.checks["passed"]), "issues": [], "model": "store"}
        node.source = "store"
        node.store_key = self.key
        node.dirty = False
        node.version += 1
        node.feedback = []
        return dag.mark_dirty(dag.dependents(self.node_id))

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "node": self.node_id, "key": self.key, "reason": self.reason}


class StopOp(Op):
    name = "stop"

    def __init__(self, reason: str = ""):
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        return set()

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "reason": self.reason}


def op_from_action(action: Action, dag: DAG, store: Any) -> Op:
    a = action.args
    if action.name == "stop":
        return StopOp(action.reason)
    if action.name == "edit_spec":
        nid = str(a.get("node") or "")
        if nid not in dag:
            raise ValueError(f"unknown node {nid!r}")
        spec = a.get("spec")
        if not isinstance(spec, dict):
            raise ValueError("edit_spec needs a spec object")
        node = dag[nid]
        merged = dict(node.spec)
        merged.update(spec)
        if node.type == ACTION:
            merged["participants"] = node.spec.get("participants", [])
        feedback = [str(a.get("feedback"))] if a.get("feedback") else []
        return EditSpec(nid, merged, feedback=feedback, reason=action.reason)
    if action.name == "split":
        return SplitOp(str(a.get("node") or ""), [p for p in (a.get("parts") or []) if isinstance(p, dict)], action.reason)
    if action.name == "merge":
        return MergeOp([str(n) for n in (a.get("nodes") or [])], a.get("merged") if isinstance(a.get("merged"), dict) else {}, action.reason)
    if action.name == "add":
        return AddOp(a.get("node") if isinstance(a.get("node"), dict) else {}, action.reason)
    if action.name == "delete":
        return DeleteOp(str(a.get("node") or ""), action.reason)
    if action.name == "reorder":
        return ReorderOp(list(a.get("order") or []), action.reason)
    if action.name == "reuse":
        return ReuseOp(str(a.get("node") or ""), str(a.get("key") or ""), store, action.reason)
    if action.name == "adjust_final":
        return AdjustFinal(dag.final_id() or "final", action.reason)
    raise ValueError(f"unsupported action {action.name}")


def apply_action(action: Action, dag: DAG, store: Any) -> Tuple[Op, Set[str]]:
    op = op_from_action(action, dag, store)
    affected = set(op.apply(dag))
    return op, affected
