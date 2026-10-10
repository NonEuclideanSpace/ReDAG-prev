from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

BACKGROUND = "background"
ENTITY = "entity"
OBJECT = "object"
ACTION = "action"
SCENE = "scene"
FINAL = "final"
DRAFT_TYPES = (BACKGROUND, ENTITY, OBJECT, ACTION)
NODE_TYPES = DRAFT_TYPES + (SCENE, FINAL)
POSES = ("standing", "sitting", "lying", "walking", "running", "jumping", "flying", "swimming", "crouching", "kneeling", "reaching", "holding", "none")
FACINGS = ("left", "right", "front", "back", "none")
RELATIONS = ("left_of", "right_of", "above", "below", "on", "under", "inside", "near", "far_from", "larger_than", "smaller_than", "behind", "in_front_of")

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,39}$")


@dataclass
class Node:
    id: str
    type: str
    spec: Dict[str, Any]
    deps: List[str] = field(default_factory=list)
    output: Any = None
    png: Optional[str] = None
    checks: Optional[Dict[str, Any]] = None
    review: Optional[Dict[str, Any]] = None
    dirty: bool = True
    version: int = 0
    feedback: List[str] = field(default_factory=list)
    source: str = ""
    store_key: str = ""

    def spec_key(self) -> str:
        payload = {"type": self.type, "spec": canonical_spec(self.spec)}
        return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def desc(self) -> str:
        return str(self.spec.get("desc") or self.id)

    def count(self) -> int:
        try:
            return max(1, int(self.spec.get("count", 1) or 1))
        except (TypeError, ValueError):
            return 1

    def colors(self) -> List[str]:
        raw = self.spec.get("colors") or []
        if isinstance(raw, str):
            raw = [raw]
        return [str(c).strip().lower() for c in raw if str(c).strip()]

    def participants(self) -> List[str]:
        raw = self.spec.get("participants") or []
        return [str(p) for p in raw]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "spec": copy.deepcopy(self.spec),
            "deps": list(self.deps),
            "output": self.output if self.type != SCENE else copy.deepcopy(self.output),
            "png": self.png,
            "checks": copy.deepcopy(self.checks),
            "review": copy.deepcopy(self.review),
            "dirty": self.dirty,
            "version": self.version,
            "feedback": list(self.feedback),
            "source": self.source,
            "store_key": self.store_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            spec=dict(data.get("spec") or {}),
            deps=[str(d) for d in (data.get("deps") or [])],
            output=data.get("output"),
            png=data.get("png"),
            checks=data.get("checks"),
            review=data.get("review"),
            dirty=bool(data.get("dirty", True)),
            version=int(data.get("version", 0)),
            feedback=[str(f) for f in (data.get("feedback") or [])],
            source=str(data.get("source") or ""),
            store_key=str(data.get("store_key") or ""),
        )


def canonical_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sorted(spec.items()):
        if k in {"position", "position_hint", "z", "z_order", "relations"}:
            continue
        if isinstance(v, str):
            out[k] = " ".join(v.strip().lower().split())
        elif isinstance(v, list):
            out[k] = [(" ".join(str(x).strip().lower().split()) if isinstance(x, str) else x) for x in v]
        elif isinstance(v, dict):
            out[k] = canonical_spec(v)
        else:
            out[k] = v
    return out


class DAG:
    def __init__(self, prompt: str, canvas: Tuple[int, int], nodes: Iterable[Node] = ()):
        self.prompt = prompt
        self.canvas = (int(canvas[0]), int(canvas[1]))
        self.nodes: Dict[str, Node] = {}
        for n in nodes:
            self.add(n)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    def __getitem__(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def add(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node

    def remove(self, node_id: str) -> None:
        self.nodes.pop(node_id)
        for other in self.nodes.values():
            if node_id in other.deps:
                other.deps = [d for d in other.deps if d != node_id]
            parts = other.spec.get("participants")
            if isinstance(parts, list) and node_id in parts:
                other.spec["participants"] = [p for p in parts if p != node_id]
            poses = other.spec.get("poses")
            if isinstance(poses, dict) and node_id in poses:
                poses.pop(node_id)

    def set_deps(self, node_id: str, deps: Iterable[str]) -> None:
        self.nodes[node_id].deps = list(dict.fromkeys(str(d) for d in deps))

    def dependents(self, node_id: str) -> List[str]:
        return [n.id for n in self.nodes.values() if node_id in n.deps]

    def downstream(self, node_ids: Iterable[str]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(node_ids)
        while stack:
            cur = stack.pop()
            for dep in self.dependents(cur):
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        return seen

    def upstream(self, node_ids: Iterable[str]) -> Set[str]:
        seen: Set[str] = set()
        stack = list(node_ids)
        while stack:
            cur = stack.pop()
            for dep in self.nodes[cur].deps:
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        return seen

    def topological(self) -> List[str]:
        indeg = {nid: 0 for nid in self.nodes}
        for n in self.nodes.values():
            for d in n.deps:
                if d in indeg:
                    indeg[n.id] += 1
        order: List[str] = []
        ready = [nid for nid, d in indeg.items() if d == 0]
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            for dep in self.dependents(cur):
                indeg[dep] -= 1
                if indeg[dep] == 0:
                    ready.append(dep)
        if len(order) != len(self.nodes):
            raise ValueError("dag contains a cycle")
        return order

    def scene_id(self) -> Optional[str]:
        for n in self.nodes.values():
            if n.type == SCENE:
                return n.id
        return None

    def final_id(self) -> Optional[str]:
        for n in self.nodes.values():
            if n.type == FINAL:
                return n.id
        return None

    def draft_nodes(self) -> List[Node]:
        return [n for n in self.nodes.values() if n.type in DRAFT_TYPES]

    def consumed(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for n in self.nodes.values():
            if n.type == ACTION:
                for p in n.participants():
                    out[p] = n.id
        return out

    def placeables(self) -> List[Node]:
        consumed = self.consumed()
        return [n for n in self.draft_nodes() if n.id not in consumed]

    def mark_dirty(self, node_ids: Iterable[str]) -> Set[str]:
        ids = set(node_ids)
        affected = set(ids) | self.downstream(ids)
        for nid in affected:
            if nid in self.nodes:
                self.nodes[nid].dirty = True
        return affected

    def normalize(self) -> None:
        scene = self.scene_id()
        if scene is None:
            self.add(Node(id="scene", type=SCENE, spec={"desc": self.prompt, "relations": []}, deps=[]))
            scene = "scene"
        for n in self.nodes.values():
            if n.type == ACTION:
                parts = n.participants()
                n.deps = list(dict.fromkeys(list(n.deps) + parts))
        self.nodes[scene].deps = [n.id for n in self.placeables()]
        final = self.final_id()
        if final is None:
            self.add(Node(id="final", type=FINAL, spec={"desc": "final image", "seed": 0, "strength": None, "control_scale": None}, deps=[scene]))
        else:
            self.nodes[final].deps = [scene]
        if not isinstance(self.nodes[scene].spec.get("relations"), list):
            self.nodes[scene].spec["relations"] = []

    def validate(self) -> List[str]:
        errors: List[str] = []
        for nid, n in self.nodes.items():
            if not _ID_RE.match(nid):
                errors.append(f"invalid node id {nid!r}")
            if n.type not in NODE_TYPES:
                errors.append(f"node {nid}: unknown type {n.type!r}")
            if nid in n.deps:
                errors.append(f"node {nid}: depends on itself")
            for d in n.deps:
                if d not in self.nodes:
                    errors.append(f"node {nid}: unknown dependency {d!r}")
            if not isinstance(n.spec, dict):
                errors.append(f"node {nid}: spec must be an object")
                continue
            if n.type in (ENTITY, OBJECT):
                if n.count() < 1:
                    errors.append(f"node {nid}: count must be >= 1")
            if n.type == ACTION:
                parts = n.participants()
                if len(parts) < 1:
                    errors.append(f"node {nid}: action needs participants")
                for p in parts:
                    if p not in self.nodes:
                        errors.append(f"node {nid}: unknown participant {p!r}")
                    elif self.nodes[p].type not in (ENTITY, OBJECT):
                        errors.append(f"node {nid}: participant {p!r} must be entity or object")
                    elif p not in n.deps:
                        errors.append(f"node {nid}: participant {p!r} missing from deps")
        scenes = [n for n in self.nodes.values() if n.type == SCENE]
        finals = [n for n in self.nodes.values() if n.type == FINAL]
        if len(scenes) != 1:
            errors.append(f"expected exactly one scene node, found {len(scenes)}")
        if len(finals) > 1:
            errors.append("more than one final node")
        consumed = self.consumed()
        for p, owners in _multi_owners(self).items():
            if len(owners) > 1:
                errors.append(f"node {p} is a participant of several actions: {owners}")
        if scenes:
            scene_deps = set(scenes[0].deps)
            expected = {n.id for n in self.placeables()}
            missing = expected - scene_deps
            if missing:
                errors.append(f"scene is missing placeables: {sorted(missing)}")
            extra = scene_deps - expected
            for e in extra:
                if e in consumed:
                    errors.append(f"scene places {e} which is consumed by action {consumed[e]}")
        if not self.draft_nodes():
            errors.append("dag has no draft nodes")
        try:
            self.topological()
        except ValueError as exc:
            errors.append(str(exc))
        return errors

    def summary(self) -> str:
        lines = [f"canvas: {self.canvas[0]}x{self.canvas[1]}"]
        for nid in self.topological():
            n = self.nodes[nid]
            state = "dirty" if n.dirty else "clean"
            spec = json.dumps({k: v for k, v in n.spec.items() if k != "notes"}, ensure_ascii=False)
            lines.append(f"- {nid} [{n.type}, v{n.version}, {state}] deps={n.deps} spec={spec}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "canvas": list(self.canvas), "nodes": [n.to_dict() for n in self.nodes.values()]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAG":
        canvas = data.get("canvas") or [1024, 1024]
        dag = cls(str(data.get("prompt") or ""), (int(canvas[0]), int(canvas[1])))
        for item in data.get("nodes") or []:
            dag.add(Node.from_dict(item))
        return dag

    def clone(self) -> "DAG":
        return DAG.from_dict(copy.deepcopy(self.to_dict()))


def _multi_owners(dag: DAG) -> Dict[str, List[str]]:
    owners: Dict[str, List[str]] = {}
    for n in dag.nodes.values():
        if n.type == ACTION:
            for p in n.participants():
                owners.setdefault(p, []).append(n.id)
    return owners


def _clean_id(raw: Any, used: Set[str], fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "").strip()).strip("_")
    if not s or not s[0].isalpha():
        s = fallback
    s = s[:40]
    base, k = s, 2
    while s in used:
        s = f"{base}_{k}"
        k += 1
    used.add(s)
    return s


def build_dag(prompt: str, data: Dict[str, Any]) -> DAG:
    if not isinstance(data, dict):
        raise ValueError("planner output must be a JSON object")
    canvas_raw = data.get("canvas") or {}
    if isinstance(canvas_raw, dict):
        w = int(canvas_raw.get("width") or 1024)
        h = int(canvas_raw.get("height") or 1024)
    elif isinstance(canvas_raw, (list, tuple)) and len(canvas_raw) == 2:
        w, h = int(canvas_raw[0]), int(canvas_raw[1])
    else:
        w, h = 1024, 1024
    w = max(256, min(2048, (w // 16) * 16))
    h = max(256, min(2048, (h // 16) * 16))
    dag = DAG(prompt, (w, h))
    used: Set[str] = set()
    rename: Dict[str, str] = {}
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("planner output has no nodes")
    prepared = []
    for i, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            continue
        ntype = str(item.get("type") or "").strip().lower()
        if ntype == "final":
            continue
        original = str(item.get("id") or f"node{i}")
        nid = _clean_id(original, used, f"node{i}")
        rename[original] = nid
        spec = item.get("spec")
        if not isinstance(spec, dict):
            spec = {"desc": str(spec) if spec else original}
        spec = dict(spec)
        if "desc" not in spec:
            spec["desc"] = str(item.get("desc") or original)
        if ntype in (ENTITY, OBJECT):
            spec["count"] = max(1, int(spec.get("count") or 1))
            spec.setdefault("colors", [])
            spec.setdefault("pose", "standing" if ntype == ENTITY else "none")
            spec.setdefault("facing", "front" if ntype == ENTITY else "none")
        if ntype == BACKGROUND:
            spec.setdefault("elements", [])
            spec.setdefault("colors", [])
        prepared.append((nid, ntype, spec, [str(d) for d in (item.get("deps") or [])]))
    for nid, ntype, spec, deps in prepared:
        deps = [rename.get(d, d) for d in deps]
        if ntype == ACTION:
            parts = [rename.get(str(p), str(p)) for p in (spec.get("participants") or [])]
            spec["participants"] = parts
            poses = spec.get("poses")
            if isinstance(poses, dict):
                spec["poses"] = {rename.get(str(k), str(k)): str(v) for k, v in poses.items()}
            deps = list(dict.fromkeys(deps + parts))
        if ntype == SCENE:
            rels = []
            for rel in spec.get("relations") or []:
                if isinstance(rel, dict) and rel.get("a") and rel.get("b"):
                    rels.append({"a": rename.get(str(rel["a"]), str(rel["a"])), "rel": str(rel.get("rel") or "near"), "b": rename.get(str(rel["b"]), str(rel["b"]))})
            spec["relations"] = rels
        dag.add(Node(id=nid, type=ntype, spec=spec, deps=deps))
    dag.normalize()
    errors = dag.validate()
    if errors:
        raise ValueError("invalid dag: " + "; ".join(errors))
    return dag
