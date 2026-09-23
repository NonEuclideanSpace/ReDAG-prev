from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from . import svg as S
from .dag import ACTION, BACKGROUND, DAG, ENTITY, OBJECT, Node
from .llm import OpenAICompatible, parse_json
from .prompts import DRAFT_SYSTEM_ACTION, DRAFT_SYSTEM_BACKGROUND, DRAFT_SYSTEM_ENTITY_OBJECT, SCENE_SYSTEM


def draft_system_prompt(node_type: str) -> str:
    if node_type == BACKGROUND:
        return DRAFT_SYSTEM_BACKGROUND
    if node_type == ACTION:
        return DRAFT_SYSTEM_ACTION
    return DRAFT_SYSTEM_ENTITY_OBJECT


def build_draft_prompt(node: Node, dag: DAG, deps: Dict[str, str], feedback: Sequence[str] = ()) -> str:
    spec = {k: v for k, v in node.spec.items()}
    lines = [
        f"SCENE PROMPT (context only, draw ONLY this node): {dag.prompt}",
        f"SCENE CANVAS: {dag.canvas[0]}x{dag.canvas[1]}",
        f"NODE ID: {node.id}",
        f"NODE TYPE: {node.type}",
        f"NODE SPEC: {json.dumps(spec, ensure_ascii=False)}",
    ]
    if node.type in (ENTITY, OBJECT):
        lines.append(f"REQUIRED: count={node.count()} colors={node.colors()} pose={spec.get('pose', 'none')} facing={spec.get('facing', 'none')}")
    if node.type == ACTION:
        lines.append("PARTICIPANT FRAGMENTS (embed each exactly once):")
        for pid in node.participants():
            frag = deps.get(pid) or (dag[pid].output if pid in dag and isinstance(dag[pid].output, str) else "")
            lines.append(f"--- participant {pid} ({dag[pid].type if pid in dag else 'unknown'}: {dag[pid].desc() if pid in dag else ''}) ---")
            lines.append(frag or "(missing)")
        poses = spec.get("poses") if isinstance(spec.get("poses"), dict) else {}
        lines.append(f"REQUIRED POSES: {json.dumps(poses, ensure_ascii=False)}")
    if feedback:
        lines.append("FEEDBACK FROM PREVIOUS ATTEMPT (fix all of these):")
        for f in feedback:
            lines.append(f"- {f}")
    lines.append(f'Output the <g data-node="{node.id}" ...> fragment now.')
    return "\n".join(lines)


def build_layout_prompt(dag: DAG, feedback: Sequence[str] = ()) -> str:
    W, H = dag.canvas
    scene = dag[dag.scene_id()]
    lines = [f"USER PROMPT: {dag.prompt}", f"CANVAS: width={W} height={H}", "PLACEABLE NODES:"]
    for n in dag.placeables():
        aspect = 1.0
        if isinstance(n.output, str) and n.output:
            try:
                bbox = S.declared_bbox(S.parse_fragment(n.output, n.id))
                if bbox:
                    aspect = bbox[2] / bbox[3]
            except (ValueError, Exception):
                aspect = 1.0
        extra = f" participants={n.participants()}" if n.type == ACTION else ""
        count = n.count() if n.type in (ENTITY, OBJECT) else 1
        lines.append(f"- {n.id} [{n.type}] {n.desc()} | count={count} | fragment_aspect_w_over_h={aspect:.2f}{extra}")
    rels = scene.spec.get("relations") or []
    lines.append(f"RELATIONS: {json.dumps(rels, ensure_ascii=False)}")
    lines.append(f"SCENE NOTES: {json.dumps(scene.spec.get('notes') or [], ensure_ascii=False)}")
    if isinstance(scene.spec.get("z_order"), list):
        lines.append(f"REQUIRED DRAWING ORDER (back to front, assign increasing z): {scene.spec['z_order']}")
    if feedback:
        lines.append("FEEDBACK FROM PREVIOUS LAYOUT (fix all of these):")
        for f in feedback:
            lines.append(f"- {f}")
    lines.append("Return the placements JSON now.")
    return "\n".join(lines)


def normalize_fragment(text: str, node: Node) -> str:
    root = S.parse_fragment(text, node.id)
    root, _removed = S.sanitize(root)
    root.set("data-node", node.id)
    root.set("data-type", node.type)
    if S.declared_bbox(root) is None:
        b = S.bounds(root)
        if b is not None:
            root.set("data-bbox", f"0 0 {int(max(1, b[2] + b[0])) } {int(max(1, b[3] + b[1]))}")
    if root.get("data-pose") is None:
        root.set("data-pose", str(node.spec.get("pose") or "none"))
    if root.get("data-facing") is None:
        root.set("data-facing", str(node.spec.get("facing") or "none"))
    return S.serialize(root)


class LLMDrafter:
    def __init__(self, llm: OpenAICompatible, retries: int = 1):
        self.llm = llm
        self.retries = int(retries)

    def draft(self, node: Node, dag: DAG, deps: Dict[str, str], feedback: Sequence[str] = (), phase: Optional[str] = None) -> str:
        user = build_draft_prompt(node, dag, deps, feedback)
        last_error = ""
        for attempt in range(self.retries + 1):
            text = self.llm.chat("executor", draft_system_prompt(node.type), user + (f"\n\nYour previous output was invalid: {last_error}" if last_error else ""), node=node.id, phase=phase, temperature=0.7)
            try:
                return normalize_fragment(text, node)
            except (ValueError, Exception) as exc:
                last_error = str(exc)[:300]
        return f'<g data-node="{node.id}" data-type="{node.type}" data-bbox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="#cccccc"/></g>'


class LLMLayouter:
    def __init__(self, llm: OpenAICompatible, retries: int = 1):
        self.llm = llm
        self.retries = int(retries)

    def layout(self, dag: DAG, feedback: Sequence[str] = (), phase: Optional[str] = None) -> Dict[str, Any]:
        user = build_layout_prompt(dag, feedback)
        last_error = ""
        for attempt in range(self.retries + 1):
            text = self.llm.chat("executor", SCENE_SYSTEM, user + (f"\n\nYour previous output was invalid: {last_error}" if last_error else ""), json_mode=True, node=dag.scene_id(), phase=phase, temperature=0.3)
            try:
                data = parse_json(text)
            except ValueError as exc:
                last_error = str(exc)
                continue
            if isinstance(data, list):
                data = {"placements": data}
            if isinstance(data, dict) and isinstance(data.get("placements"), list):
                return {"placements": data["placements"]}
            last_error = "missing placements list"
        return {"placements": []}
