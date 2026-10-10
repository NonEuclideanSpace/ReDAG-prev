from __future__ import annotations

import copy
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageFilter

from . import mock_templates as T
from . import svg as S
from .checks import check_fragment, check_layout, issues_from_checks
from .dag import ACTION, BACKGROUND, DAG, ENTITY, FINAL, OBJECT, SCENE, Node, build_dag

DRAFT_VARIANTS = ("correct", "wrong_color", "wrong_count", "wrong_pose", "malformed", "off_bbox", "missing_participant")
LAYOUT_VARIANTS = ("correct", "outside", "overlap", "omit", "relation")
ACTION_VERBS = {
    "holding": "holding", "holds": "holding", "hold": "holding", "carrying": "holding", "carries": "holding",
    "riding": "riding", "rides": "riding", "ride": "riding",
    "pushing": "pushing", "pushes": "pushing", "pulling": "pulling", "pulls": "pulling",
    "hugging": "hugging", "hugs": "hugging", "feeding": "feeding", "feeds": "feeding",
    "kicking": "kicking", "kicks": "kicking", "reading": "reading", "reads": "reading",
    "petting": "petting", "pets": "petting", "throwing": "throwing", "throws": "throwing",
    "sitting on": "sitting_on", "sits on": "sitting_on", "lying on": "lying_on", "lies on": "lying_on",
    "standing on": "standing_on", "stands on": "standing_on", "wearing": "wearing", "wears": "wearing",
}
RELATION_PHRASES = [
    ("to the left of", "left_of"), ("left of", "left_of"), ("to the right of", "right_of"), ("right of", "right_of"),
    ("on top of", "on"), ("above", "above"), ("below", "below"), ("under", "under"), ("underneath", "under"), ("inside", "inside"),
    ("in front of", "in_front_of"), ("behind", "behind"), ("next to", "near"), ("beside", "near"), ("near", "near"), ("far from", "far_from"),
    ("on", "on"), ("in", "inside"),
]
POSE_WORDS = {"sitting": "sitting", "sits": "sitting", "seated": "sitting", "standing": "standing", "stands": "standing", "lying": "lying", "lies": "lying", "running": "running", "runs": "running", "walking": "walking", "walks": "walking", "jumping": "jumping", "jumps": "jumping", "flying": "flying", "flies": "flying", "swimming": "swimming", "swims": "swimming", "crouching": "crouching", "kneeling": "kneeling", "reaching": "reaching"}
STOP = {"a", "an", "the", "of", "and", "with", "some", "several", "many", "few", "is", "are", "there", "in", "on", "at", "very", "big", "small", "large", "little", "tiny", "huge", "tall", "short", "long", "old", "young", "cute", "happy", "sad", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "its", "their", "his", "her", "while", "as", "that", "which", "who", "each", "other", "another", "all", "both", "front", "back", "left", "right", "top", "bottom", "side", "middle", "center", "corner", "sky", "ground"}
SIZE_WORDS = {"big", "large", "huge", "giant", "tall", "small", "little", "tiny"}


class MockRuntime:
    def __init__(self, seed: int = 0, error_rate: float = 0.0, cost: Any = None):
        self.rng = random.Random(int(seed))
        self.error_rate = float(error_rate)
        self.calls = 0
        self.cost = cost

    def record(self, kind: str, node: Optional[str] = None, label: str = "", phase: Optional[str] = None) -> None:
        if self.cost is not None:
            self.cost.record(kind, node=node, phase=phase, label=label)

    def chance(self, p: float) -> bool:
        self.calls += 1
        return self.rng.random() < p

    def choice(self, seq: Sequence[Any]) -> Any:
        self.calls += 1
        return self.rng.choice(list(seq))


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z][a-z\-]*", text.lower())


def _color_tokens(tokens: Sequence[str]) -> List[str]:
    out = []
    for t in tokens:
        key = S.normalize_color_name(t)
        if key in S.NAMED_COLORS or key in S._CSS_EXTRA:
            out.append(t.replace("-", ""))
    return out


def _phrase_parts(phrase: str) -> Dict[str, Any]:
    toks = _tokens(phrase)
    count = 1
    for t in toks:
        if t in T.NUMBER_WORDS and t not in ("a", "an"):
            count = T.NUMBER_WORDS[t]
    digits = re.findall(r"\b(\d{1,2})\b", phrase)
    if digits:
        count = int(digits[0])
    colors = _color_tokens(toks)
    pose = None
    facing = None
    for i, t in enumerate(toks):
        if t in POSE_WORDS:
            pose = POSE_WORDS[t]
        if t in ("facing", "looking", "turned") and i + 1 < len(toks) and toks[i + 1] in ("left", "right", "front", "back", "away"):
            facing = "back" if toks[i + 1] == "away" else toks[i + 1]
    nouns = [t for t in toks if t not in STOP and t not in POSE_WORDS and S.normalize_color_name(t) not in S.NAMED_COLORS and S.normalize_color_name(t) not in S._CSS_EXTRA and t not in T.NUMBER_WORDS and t not in ("facing", "looking", "turned")]
    entity_nouns = [t for t in nouns if T.singular(t) in T.ENTITY_WORDS]
    noun = T.singular(entity_nouns[0]) if entity_nouns else (T.singular(nouns[-1]) if nouns else "thing")
    attrs = [t for t in toks if t in SIZE_WORDS]
    return {"count": count, "colors": colors, "pose": pose, "facing": facing, "noun": noun, "attributes": attrs, "phrase": phrase.strip()}


def _split_clauses(prompt: str) -> List[str]:
    text = prompt.replace("\n", " ")
    parts = re.split(r",|;|\.|\bwhile\b|\band\b|\bwith\b", text)
    return [p.strip() for p in parts if p.strip()]


class MockPlanner:
    def __init__(self, runtime: MockRuntime):
        self.runtime = runtime

    def plan(self, prompt: str, catalog: Optional[List[Dict[str, Any]]] = None, hint: Optional[Dict[str, Any]] = None) -> DAG:
        self.runtime.record("llm", node="planner", label="mock_plan", phase="plan")
        if hint:
            return build_dag(prompt, hint)
        return build_dag(prompt, self.plan_json(prompt, catalog or []))

    def plan_json(self, prompt: str, catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
        text = prompt.lower()
        nodes: List[Dict[str, Any]] = []
        used: Dict[str, str] = {}
        bg_elements = [w for w in dict.fromkeys(_tokens(text)) if w in T.BACKGROUND_WORDS]
        relations: List[Dict[str, str]] = []
        actions: List[Dict[str, Any]] = []

        def ensure_node(parts: Dict[str, Any]) -> str:
            noun = parts["noun"]
            key = "_".join(parts["colors"][:1] + [noun])
            if key in used:
                return used[key]
            base = re.sub(r"[^a-z0-9_]+", "_", key).strip("_") or "thing"
            nid = base
            k = 2
            while any(n["id"] == nid for n in nodes):
                nid = f"{base}_{k}"
                k += 1
            ntype = ENTITY if noun in T.ENTITY_WORDS else OBJECT
            spec = {
                "desc": parts["phrase"] if parts["phrase"] else noun,
                "noun": noun,
                "count": parts["count"],
                "colors": parts["colors"][:2],
                "pose": parts["pose"] or ("standing" if ntype == ENTITY else "none"),
                "facing": parts["facing"] or ("front" if ntype == ENTITY else "none"),
                "attributes": parts["attributes"],
            }
            for entry in catalog:
                cspec = entry.get("spec") or {}
                if entry.get("type") == ntype and cspec.get("noun") == noun and [c for c in cspec.get("colors", [])] == spec["colors"] and int(cspec.get("count", 1)) == spec["count"] and cspec.get("pose") == spec["pose"] and cspec.get("facing") == spec["facing"]:
                    spec = dict(cspec)
                    break
            nodes.append({"id": nid, "type": ntype, "spec": spec, "deps": []})
            used[key] = nid
            return nid

        remaining = text
        for verb_phrase, verb in sorted(ACTION_VERBS.items(), key=lambda kv: -len(kv[0])):
            for m in re.finditer(rf"([a-z0-9 \-]+?)\s+{re.escape(verb_phrase)}\s+([a-z0-9 \-]+)", remaining):
                left = _phrase_parts(m.group(1).split(",")[-1])
                right_text = m.group(2)
                right_text = re.split(r"\b(?:to the left of|left of|to the right of|right of|on top of|above|below|under|next to|beside|near|behind|in front of|while|and|with)\b", right_text)[0]
                right = _phrase_parts(right_text)
                if left["noun"] == "thing" or right["noun"] == "thing":
                    continue
                a = ensure_node(left)
                b = ensure_node(right)
                if a == b:
                    continue
                poses = {a: ("sitting" if verb in ("sitting_on", "riding") else "lying" if verb == "lying_on" else "holding" if verb in ("holding", "reading", "feeding", "hugging", "petting") else "standing" if verb in ("standing_on", "pushing", "pulling", "kicking", "throwing", "wearing") else "standing")}
                actions.append({"id": f"{verb}_{a}_{b}"[:40], "type": ACTION, "spec": {"desc": m.group(0).strip(), "verb": verb, "participants": [a, b], "poses": poses}, "deps": [a, b]})
                remaining = remaining.replace(m.group(0), f" {m.group(1)} , {m.group(2)} ")
        for phrase, rel in RELATION_PHRASES:
            pattern = rf"([a-z0-9 \-]+?)\s+{re.escape(phrase)}\s+([a-z0-9 \-]+)"
            for m in re.finditer(pattern, remaining):
                left = _phrase_parts(m.group(1).split(",")[-1])
                right_text = re.split(r"\b(?:while|and|with)\b", m.group(2))[0]
                right = _phrase_parts(right_text)
                if left["noun"] == "thing" or right["noun"] == "thing":
                    continue
                if left["noun"] in T.BACKGROUND_WORDS or right["noun"] in T.BACKGROUND_WORDS:
                    continue
                a = ensure_node(left)
                b = ensure_node(right)
                if a != b:
                    relations.append({"a": a, "rel": rel, "b": b})
                remaining = remaining.replace(m.group(0), f" {left['phrase']} , {right_text} ")
        for clause in _split_clauses(remaining):
            parts = _phrase_parts(clause)
            if parts["noun"] == "thing" or parts["noun"] in T.BACKGROUND_WORDS:
                continue
            ensure_node(parts)
        bg_colors = []
        for m in re.finditer(r"(\w+)\s+(sky|grass|sea|ocean|wall|floor|ground|field|sand|water)", text):
            if m.group(1) in S.NAMED_COLORS:
                bg_colors.append(m.group(1))
        if bg_elements or not nodes:
            nodes.insert(0, {"id": "background", "type": BACKGROUND, "spec": {"desc": "background: " + ", ".join(bg_elements or ["plain ground"]), "elements": bg_elements or ["ground"], "colors": bg_colors[:2]}, "deps": []})
        nodes.extend(actions)
        wide = len([n for n in nodes if n["type"] in (ENTITY, OBJECT)]) >= 3
        canvas = {"width": 1280, "height": 768} if wide else {"width": 1024, "height": 1024}
        nodes.append({"id": "scene", "type": SCENE, "spec": {"desc": prompt, "relations": relations}, "deps": []})
        return {"canvas": canvas, "nodes": nodes}

    def revise_spec(self, node: Node, issues: List[Dict[str, Any]], dag: DAG) -> Dict[str, Any]:
        self.runtime.record("llm", node=node.id, label="mock_revise", phase="repair")
        spec = json.loads(json.dumps(node.spec))
        notes = list(spec.get("notes") or [])
        for issue in issues:
            text = str(issue.get("fix_instruction") or issue.get("description") or "").strip()
            if text and text not in notes:
                notes.append(text)
        spec["notes"] = notes[-6:]
        return spec


def _noun_of(node: Node) -> str:
    noun = node.spec.get("noun")
    if noun:
        return str(noun)
    return _phrase_parts(node.desc())["noun"]


def _verb_of(node: Node) -> str:
    verb = node.spec.get("verb")
    if verb:
        return str(verb)
    desc = node.desc().lower()
    for phrase, name in sorted(ACTION_VERBS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(phrase)}\b", desc):
            return name
    return "holding"


def _wrap(node: Node, w: float, h: float, body: str, pose: str, facing: str) -> str:
    return f'<g data-node="{node.id}" data-type="{node.type}" data-bbox="0 0 {int(w)} {int(h)}" data-pose="{pose}" data-facing="{facing}">{body}</g>'


class MockDrafter:
    def __init__(self, runtime: MockRuntime, canvas_hint: Tuple[int, int] = (1024, 1024)):
        self.runtime = runtime
        self.canvas_hint = canvas_hint

    def pick_variant(self, node: Node, feedback: Sequence[str], rate_scale: float = 1.0) -> str:
        p = self.runtime.error_rate * rate_scale * (0.3 if feedback else 1.0)
        if not self.runtime.chance(p):
            return "correct"
        options = ["wrong_color", "malformed", "off_bbox"]
        if node.type in (ENTITY, OBJECT):
            options.append("wrong_count")
        if node.type == ENTITY:
            options.append("wrong_pose")
        if node.type == ACTION:
            options.extend(["missing_participant", "wrong_pose"])
        if node.type == BACKGROUND:
            options = ["wrong_color", "malformed"]
        return self.runtime.choice(options)

    def draft(self, node: Node, dag: DAG, deps: Dict[str, str], feedback: Sequence[str] = (), variant: Optional[str] = None, rate_scale: float = 1.0, phase: Optional[str] = None) -> str:
        chosen = variant or self.pick_variant(node, feedback, rate_scale)
        self.runtime.record("llm", node=node.id, label="mock_draft")
        if node.type == BACKGROUND:
            frag = self._background(node, dag)
        elif node.type == ACTION:
            frag = self._action(node, dag, deps, chosen)
        else:
            frag = self._subject(node)
        return self.corrupt(frag, node, chosen)

    def _subject(self, node: Node, color_override: Optional[str] = None) -> str:
        noun = _noun_of(node)
        colors = node.colors()
        color = color_override or (colors[0] if colors else T.default_color(noun))
        secondary = colors[1] if len(colors) > 1 else None
        pose = str(node.spec.get("pose") or ("standing" if node.type == ENTITY else "none"))
        facing = str(node.spec.get("facing") or ("front" if node.type == ENTITY else "none"))
        cw, ch = T.default_aspect(noun, pose)
        count = node.count()
        if count == 1:
            body = "".join(T.draw_noun(noun, color, cw, ch, pose, facing, secondary))
            return _wrap(node, cw, ch, body, pose, facing)
        cells = T.grid_layout(count, cw * min(count, 3), ch * math.ceil(count / 3), cw / ch)
        W = cw * min(count, 3)
        H = ch * math.ceil(count / 3)
        parts = []
        for i, (x, y, w, h) in enumerate(cells):
            s = min(w / cw, h / ch)
            inner = "".join(T.draw_noun(noun, color, cw, ch, pose, facing, secondary))
            parts.append(f'<g data-instance="{i + 1}" transform="translate({x + (w - cw * s) / 2:.1f} {y + (h - ch * s) / 2:.1f}) scale({s:.4f})">{inner}</g>')
        return _wrap(node, W, H, "".join(parts), pose, facing)

    def _background(self, node: Node, dag: DAG) -> str:
        W, H = dag.canvas
        w, h = 640, int(round(640 * H / W))
        body = "".join(T.background_elements(node.spec.get("elements") or [], node.colors(), w, h))
        return _wrap(node, w, h, body, "none", "none")

    def _action(self, node: Node, dag: DAG, deps: Dict[str, str], variant: str) -> str:
        parts = node.participants()
        poses = node.spec.get("poses") if isinstance(node.spec.get("poses"), dict) else {}
        verb = _verb_of(node)
        frags = []
        for pid in parts:
            text = deps.get(pid)
            if text is None and pid in dag and dag[pid].output:
                text = dag[pid].output
            if text is None:
                text = self._subject(dag[pid]) if pid in dag else f'<g data-node="{pid}" data-type="object" data-bbox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="#888888"/></g>'
            try:
                root = S.parse_fragment(text, pid)
            except (ValueError, ET.ParseError):
                root = S.parse_fragment(self._subject(dag[pid]) if pid in dag else f'<g data-node="{pid}" data-type="object" data-bbox="0 0 100 100"><rect x="10" y="10" width="80" height="80" fill="#888888"/></g>', pid)
            bbox = S.declared_bbox(root) or (0, 0, 100, 100)
            required = poses.get(pid)
            if required:
                root.set("data-pose", str(required))
            frags.append((pid, root, bbox))
        if not frags:
            return _wrap(node, 200, 200, '<rect x="20" y="20" width="160" height="160" fill="#888888"/>', "none", "none")
        a_id, a_root, a_box = frags[0]
        if len(frags) == 1:
            W, H = a_box[2], a_box[3]
            body = f'<g transform="translate(0 0)">{S.serialize(a_root)}</g>'
            return _wrap(node, W, H, body, verb, "front")
        b_id, b_root, b_box = frags[1]
        verb_key = verb
        if verb_key in ("riding", "sitting_on", "standing_on", "lying_on"):
            scale_b = 1.0
            scale_a = (b_box[2] * 0.7) / a_box[2]
            W = b_box[2]
            H = b_box[3] + a_box[3] * scale_a * 0.7
            a_pos = (W / 2 - a_box[2] * scale_a / 2, 0)
            b_pos = (0, a_box[3] * scale_a * 0.7)
        elif verb_key in ("holding", "reading", "feeding", "hugging", "petting", "wearing", "throwing"):
            scale_a = 1.0
            scale_b = min(0.5 * a_box[3] / b_box[3], 0.6 * a_box[2] / b_box[2])
            W = a_box[2] + b_box[2] * scale_b * 0.8
            H = a_box[3]
            a_pos = (0, 0)
            b_pos = (a_box[2] * 0.8, a_box[3] * 0.15)
        else:
            scale_a = 1.0
            scale_b = min(1.0, a_box[3] / b_box[3])
            W = a_box[2] + b_box[2] * scale_b + 10
            H = max(a_box[3], b_box[3] * scale_b)
            a_pos = (0, H - a_box[3])
            b_pos = (a_box[2] + 10, H - b_box[3] * scale_b)
        body = (
            f'<g transform="translate({b_pos[0]:.1f} {b_pos[1]:.1f}) scale({scale_b:.4f})">{S.serialize(b_root)}</g>'
            f'<g transform="translate({a_pos[0]:.1f} {a_pos[1]:.1f}) scale({scale_a:.4f})">{S.serialize(a_root)}</g>'
        )
        for pid, root, box in frags[2:]:
            body += f'<g transform="translate({W:.1f} {max(0.0, H - box[3]):.1f}) scale(1)">{S.serialize(root)}</g>'
            W += box[2]
        if variant == "missing_participant":
            body = f'<g transform="translate({a_pos[0]:.1f} {a_pos[1]:.1f}) scale({scale_a:.4f})">{S.serialize(a_root)}</g>'
        if variant == "wrong_pose":
            b_root.set("data-pose", "none")
            a_root.set("data-pose", "flying")
            body = (
                f'<g transform="translate({b_pos[0]:.1f} {b_pos[1]:.1f}) scale({scale_b:.4f})">{S.serialize(b_root)}</g>'
                f'<g transform="translate({a_pos[0]:.1f} {a_pos[1]:.1f}) scale({scale_a:.4f})">{S.serialize(a_root)}</g>'
            )
        return _wrap(node, W, H, body, verb, "front")

    def corrupt(self, fragment: str, node: Node, variant: str) -> str:
        if variant in ("correct", "missing_participant") or (variant == "wrong_pose" and node.type == ACTION):
            return fragment
        if variant == "wrong_color":
            colors = node.colors()
            if not colors:
                return fragment
            target = T.hexcolor(colors[0])
            alternatives = [c for c in ("red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown") if not S.color_matches(S.parse_color(c), colors[0]) and not any(S.color_matches(S.parse_color(c), other) for other in colors[1:])]
            chosen_color = self.runtime.choice(alternatives)
            fragment = fragment.replace(target, T.hexcolor(chosen_color))
            return fragment.replace(T.darker(colors[0]), T.darker(chosen_color))
        if variant == "wrong_count":
            root = S.parse_fragment(fragment, node.id)
            inst = S.instances(root)
            parent_map = {child: parent for parent in root.iter() for child in parent}
            if len(inst) >= 2:
                parent_map[inst[-1]].remove(inst[-1])
                return S.serialize(root)
            bbox = S.declared_bbox(root) or (0, 0, 100, 100)
            if len(inst) == 1:
                dup = copy.deepcopy(inst[0])
                dup.set("data-instance", str(len(inst) + 1))
                dup.set("transform", f"translate({bbox[2] * 0.5:.1f} {bbox[3] * 0.5:.1f}) scale(0.5)")
                parent_map[inst[0]].append(dup)
                return S.serialize(root)
            children = list(root)
            for child in children:
                root.remove(child)
            first = ET.SubElement(root, "g", {"data-instance": "1"})
            for child in children:
                first.append(child)
            second = ET.SubElement(root, "g", {"data-instance": "2", "transform": f"translate({bbox[2]:.1f} 0)"})
            for child in children:
                second.append(copy.deepcopy(child))
            root.set("data-bbox", f"0 0 {int(bbox[2] * 2)} {int(bbox[3])}")
            return S.serialize(root)
        if variant == "wrong_pose":
            root = S.parse_fragment(fragment, node.id)
            facing = str(node.spec.get("facing") or "front")
            wrong = {"left": "right", "right": "left", "front": "back", "back": "front", "none": "left"}.get(facing, "left")
            root.set("data-facing", wrong)
            if facing in ("left", "right"):
                bbox = S.declared_bbox(root) or (0, 0, 100, 100)
                for elem in root.iter():
                    if elem.get("data-anchor") == "face":
                        elem.set("cx", f"{bbox[2] - float(elem.get('cx', '0')):.1f}")
            return S.serialize(root)
        if variant == "malformed":
            idx = fragment.rfind("</g>")
            return fragment[:idx] if idx > 0 else fragment + "<rect"
        if variant == "off_bbox":
            root = S.parse_fragment(fragment, node.id)
            bbox = S.declared_bbox(root) or (0, 0, 100, 100)
            children = list(root)
            for child in children:
                root.remove(child)
            shifted = S.parse_fragment(f'<g transform="translate({bbox[2] * 0.4:.1f} {bbox[3] * 0.4:.1f})">' + "".join(S.serialize(c) for c in children) + "</g>")
            root.append(shifted)
            return S.serialize(root)
        return fragment


class MockLayouter:
    def __init__(self, runtime: MockRuntime):
        self.runtime = runtime

    def pick_variant(self, feedback: Sequence[str], rate_scale: float = 1.0) -> str:
        p = self.runtime.error_rate * rate_scale * (0.3 if feedback else 1.0)
        if not self.runtime.chance(p):
            return "correct"
        return self.runtime.choice(LAYOUT_VARIANTS[1:])

    def layout(self, dag: DAG, feedback: Sequence[str] = (), variant: Optional[str] = None, rate_scale: float = 1.0, phase: Optional[str] = None) -> Dict[str, Any]:
        chosen = variant or self.pick_variant(feedback, rate_scale)
        self.runtime.record("llm", node=dag.scene_id(), label="mock_layout")
        W, H = dag.canvas
        placeables = dag.placeables()
        consumed = dag.consumed()
        scene = dag[dag.scene_id()]
        rels = [r for r in (scene.spec.get("relations") or []) if isinstance(r, dict)]
        aspects: Dict[str, float] = {}
        for n in placeables:
            if n.type == BACKGROUND:
                continue
            aspect = 1.0
            if n.output:
                try:
                    bbox = S.declared_bbox(S.parse_fragment(n.output, n.id))
                    if bbox:
                        aspect = bbox[2] / bbox[3]
                except (ValueError, Exception):
                    aspect = 1.0
            aspects[n.id] = max(0.25, min(5.0, aspect))
        subjects = [n.id for n in placeables if n.type != BACKGROUND]
        order = list(subjects)

        def ref(x: str) -> str:
            return consumed.get(x, x)

        for _ in range(3):
            for r in rels:
                a, b = ref(str(r["a"])), ref(str(r["b"]))
                if a not in order or b not in order or a == b:
                    continue
                rel = str(r.get("rel"))
                if rel == "left_of" and order.index(a) > order.index(b):
                    order.remove(a)
                    order.insert(order.index(b), a)
                if rel == "right_of" and order.index(a) < order.index(b):
                    order.remove(a)
                    order.insert(order.index(b) + 1, a)
        stacked: Dict[str, Tuple[str, str]] = {}
        for r in rels:
            a, b = ref(str(r["a"])), ref(str(r["b"]))
            rel = str(r.get("rel"))
            if a == b or a not in order or b not in order:
                continue
            if rel in ("on", "above"):
                stacked[a] = (b, "above")
            elif rel in ("below", "under"):
                stacked[a] = (b, "below")
            elif rel == "inside":
                stacked[a] = (b, "inside")
        for a in list(stacked.keys()):
            b = stacked[a][0]
            if b in stacked and stacked[b][0] == a:
                stacked.pop(a)
        columns = [nid for nid in order if nid not in stacked]
        if not columns and order:
            columns = [order[0]]
            stacked.pop(order[0], None)
        weights = {nid: math.sqrt(aspects.get(nid, 1.0)) for nid in columns}
        margin = W * 0.04
        gap = W * 0.025
        avail = W - 2 * margin - gap * max(0, len(columns) - 1)
        total = sum(weights.values()) or 1.0
        boxes: Dict[str, List[float]] = {}
        x = margin
        band_top, band_bottom = H * 0.14, H * 0.92
        for nid in columns:
            cw = avail * weights[nid] / total
            aspect = aspects.get(nid, 1.0)
            ch = cw / aspect
            max_h = band_bottom - band_top
            if ch > max_h:
                ch = max_h
                cw = ch * aspect
            boxes[nid] = [x, band_bottom - ch, x + cw, band_bottom]
            x += cw + gap
        groups: Dict[str, List[str]] = {}
        for a in order:
            if a in stacked:
                groups.setdefault(stacked[a][0], []).append(a)

        def member_sizes(group: List[str], base_w: float) -> Dict[str, Tuple[float, float]]:
            ws = {m: math.sqrt(aspects.get(m, 1.0)) for m in group}
            tot = sum(ws.values()) or 1.0
            inner_gap = base_w * 0.04
            span = base_w - inner_gap * max(0, len(group) - 1)
            sizes = {}
            for m in group:
                mw = span * ws[m] / tot
                sizes[m] = (mw, mw / aspects.get(m, 1.0))
            return sizes

        band_h = band_bottom - band_top
        for base, members in groups.items():
            if base not in boxes:
                continue
            above = [m for m in members if stacked[m][1] == "above"]
            below = [m for m in members if stacked[m][1] == "below"]
            bb = boxes[base]
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            extra = 0.0
            if above:
                extra += max(h for _w, h in member_sizes(above, bw).values())
            if below:
                extra += max(h for _w, h in member_sizes(below, bw).values())
            if bh + extra > band_h and bh + extra > 0:
                k = band_h / (bh + extra)
                bw, bh = bw * k, bh * k
                boxes[base] = [bb[0], bb[3] - bh, bb[0] + bw, bb[3]]
        for base, members in groups.items():
            if base not in boxes:
                continue
            bb = boxes[base]
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            above = [m for m in members if stacked[m][1] == "above"]
            below = [m for m in members if stacked[m][1] == "below"]
            inside = [m for m in members if stacked[m][1] == "inside"]
            for group, how in ((above, "above"), (below, "below")):
                if not group:
                    continue
                sizes = member_sizes(group, bw)
                inner_gap = bw * 0.04
                cx = bb[0]
                for m in group:
                    mw, mh = sizes[m]
                    if how == "above":
                        boxes[m] = [cx, bb[1] - mh, cx + mw, bb[1]]
                    else:
                        boxes[m] = [cx, bb[3], cx + mw, bb[3] + mh]
                    cx += mw + inner_gap
            for m in inside:
                aw = bw * 0.8
                ah = aw / aspects.get(m, 1.0)
                if ah > bh * 0.6:
                    ah = bh * 0.6
                    aw = ah * aspects.get(m, 1.0)
                mx, my = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                boxes[m] = [mx - aw / 2, my - ah / 2, mx + aw / 2, my + ah / 2]
        nested = [a for a, (b, how) in stacked.items() if b in stacked]
        for _ in range(2):
            for a in nested:
                b, how = stacked[a]
                if b not in boxes:
                    continue
                bb = boxes[b]
                bw, bh = bb[2] - bb[0], bb[3] - bb[1]
                if how == "inside":
                    aw = bw * 0.8
                    ah = aw / aspects.get(a, 1.0)
                    if ah > bh * 0.6:
                        ah = bh * 0.6
                        aw = ah * aspects.get(a, 1.0)
                    mx, my = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                    boxes[a] = [mx - aw / 2, my - ah / 2, mx + aw / 2, my + ah / 2]
                elif how == "above":
                    aw = bw * 0.7
                    ah = min(aw / aspects.get(a, 1.0), max(H * 0.06, bb[1] - H * 0.03))
                    aw = min(aw, ah * aspects.get(a, 1.0))
                    mx = (bb[0] + bb[2]) / 2
                    boxes[a] = [mx - aw / 2, bb[1] - ah, mx + aw / 2, bb[1]]
                else:
                    aw = bw * 0.7
                    ah = min(aw / aspects.get(a, 1.0), max(H * 0.06, H * 0.98 - bb[3]))
                    aw = min(aw, ah * aspects.get(a, 1.0))
                    mx = (bb[0] + bb[2]) / 2
                    boxes[a] = [mx - aw / 2, bb[3], mx + aw / 2, bb[3] + ah]
        subject_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes.values())
        coverage = subject_area / float(W * H) if boxes else 1.0
        if coverage < 0.3 and columns:
            s = min(1.35, math.sqrt(0.32 / max(coverage, 1e-6)))
            for nid, b in list(boxes.items()):
                cx = (b[0] + b[2]) / 2
                bw, bh = (b[2] - b[0]) * s, (b[3] - b[1]) * s
                boxes[nid] = [cx - bw / 2, b[3] - bh, cx + bw / 2, b[3]]
            x = margin
            for nid in columns:
                b = boxes[nid]
                bw = b[2] - b[0]
                shift = x - b[0]
                for m in [nid] + groups.get(nid, []):
                    mb = boxes[m]
                    boxes[m] = [mb[0] + shift, mb[1], mb[2] + shift, mb[3]]
                x += bw + gap
            span = x - gap - margin
            if span > W - 2 * margin:
                k = (W - 2 * margin) / span
                for nid, b in list(boxes.items()):
                    boxes[nid] = [margin + (b[0] - margin) * k, b[3] - (b[3] - b[1]) * k, margin + (b[2] - margin) * k, b[3]]
        low = max((b[3] for b in boxes.values()), default=0.0)
        if low > H * 0.98:
            shift = low - H * 0.98
            for nid, b in list(boxes.items()):
                boxes[nid] = [b[0], b[1] - shift, b[2], b[3] - shift]
        for nid, b in boxes.items():
            boxes[nid] = [max(0.0, b[0]), max(0.0, b[1]), min(float(W), b[2]), min(float(H), b[3])]
        if chosen == "outside" and boxes:
            victim = self.runtime.choice(list(boxes.keys()))
            b = boxes[victim]
            boxes[victim] = [b[0] + W * 0.3, b[1], b[2] + W * 0.3, b[3]]
        if chosen == "overlap" and len(boxes) >= 2:
            ids = list(boxes.keys())
            boxes[ids[1]] = list(boxes[ids[0]])
        if chosen == "omit" and boxes:
            victim = self.runtime.choice(list(boxes.keys()))
            boxes.pop(victim)
        if chosen == "relation" and rels:
            r = rels[0]
            a, b = ref(str(r["a"])), ref(str(r["b"]))
            if a in boxes and b in boxes:
                boxes[a], boxes[b] = boxes[b], boxes[a]
        placements: List[Dict[str, Any]] = []
        for n in placeables:
            if n.type == BACKGROUND:
                placements.append({"node": n.id, "bbox": [0, 0, W, H], "z": 0})
        z_order = scene.spec.get("z_order") if isinstance(scene.spec.get("z_order"), list) else None
        draw_order = [nid for nid in z_order if nid in boxes] + [nid for nid in order if nid in boxes and nid not in (z_order or [])] if z_order else [nid for nid in order if nid in boxes]
        z = 1
        for nid in draw_order:
            placements.append({"node": nid, "bbox": [round(v, 1) for v in boxes[nid]], "z": z})
            z += 1
        return {"placements": placements}


def _issue(itype: str, desc: str, nodes: List[str], fix: str, severity: str = "major") -> Dict[str, Any]:
    return {"type": itype, "severity": severity, "description": desc, "nodes": nodes, "fix_instruction": fix, "source": "mock"}


class MockReviewer:
    def __init__(self, runtime: MockRuntime):
        self.runtime = runtime
        self.attribute_rate = 0.7
        self.reference_dag: Optional[DAG] = None

    def _reference_issues(self, dag: DAG) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        if self.reference_dag is None:
            return issues
        current = [n for n in dag.draft_nodes() if n.type in (ENTITY, OBJECT)]
        for ref in self.reference_dag.draft_nodes():
            if ref.type not in (ENTITY, OBJECT):
                continue
            noun = _noun_of(ref)
            colors = ref.colors()
            matches = [n for n in current if _noun_of(n) == noun and (not colors or (n.colors() and S.color_matches(S.parse_color(n.colors()[0]) or (0, 0, 0), colors[0])))]
            if not matches:
                issues.append(_issue("missing_object", f"{ref.desc()} required by the prompt is absent from the draft", [dag.scene_id()], f"add {ref.desc()} back"))
                continue
            total = sum(n.count() for n in matches)
            if total < ref.count():
                issues.append(_issue("count_mismatch", f"prompt requires {ref.count()} of {ref.desc()} but only {total} are drawn", [matches[0].id], f"draw {ref.count()} instances"))
        return issues

    def node(self, node: Node, fragment: str, png: Optional[str], dag: DAG) -> Dict[str, Any]:
        self.runtime.record("vlm", node=node.id, label="mock_node_review")
        report = check_fragment(fragment, node, dag)
        issues = issues_from_checks(report, node.id)
        return {"passed": report["passed"], "issues": issues, "model": "mock"}

    def _hide_attribution(self, issues: List[Dict[str, Any]], dag: DAG) -> List[Dict[str, Any]]:
        out = []
        for issue in issues:
            issue = dict(issue)
            nodes = issue.get("nodes") or []
            if nodes and not self.runtime.chance(self.attribute_rate):
                descs = ["the overall layout" if n == dag.scene_id() else dag[n].desc() for n in nodes if n in dag]
                issue["description"] = f"{issue['description']} (about: {'; '.join(descs)})"
                issue["nodes"] = []
            out.append(issue)
        return out

    def scene(self, dag: DAG, layout: Dict[str, Any], composed_svg: str, png: Optional[str], overlay: Optional[str]) -> Dict[str, Any]:
        self.runtime.record("vlm", node=dag.scene_id(), label="mock_scene_review")
        issues: List[Dict[str, Any]] = []
        layout_report = check_layout(layout, dag)
        for f in layout_report.get("failures", []):
            issues.append(_issue(issues_from_checks({"failures": [f]}, dag.scene_id())[0]["type"], f"{f['name']}: {f['detail']}", [dag.scene_id()], f"fix layout {f['name']}"))
        try:
            root = S.parse_svg_text(composed_svg)
        except (ValueError, Exception):
            return {"passed": False, "issues": [_issue("other", "composed svg unparseable", [dag.scene_id()], "recompose")], "model": "mock"}
        for n in dag.placeables():
            g = root.find(f".//g[@data-node='{n.id}']")
            if g is None or S.shape_count(g) == 0:
                issues.append(_issue("missing_object", f"{n.desc()} is not in the draft", [n.id], "draw and place it"))
                continue
            report = check_fragment(S.serialize(g), n, dag)
            for item in issues_from_checks(report, n.id):
                if item["type"] in ("count_mismatch", "color_mismatch", "pose_mismatch", "missing_object"):
                    issues.append(item)
            if n.type == ACTION:
                for pid in n.participants():
                    inner = g.find(f".//g[@data-node='{pid}']")
                    if inner is not None and pid in dag:
                        rep = check_fragment(S.serialize(inner), dag[pid], dag)
                        for item in issues_from_checks(rep, pid):
                            if item["type"] in ("count_mismatch", "color_mismatch"):
                                item["nodes"] = [n.id]
                                issues.append(item)
        issues.extend(self._reference_issues(dag))
        issues = self._hide_attribution(issues, dag)
        return {"passed": not any(i["severity"] == "major" for i in issues), "issues": issues, "model": "mock"}

    def final(self, dag: DAG, layout: Dict[str, Any], composed_svg: str, draft_png: Optional[str], final_png: Optional[str], round_index: int) -> Dict[str, Any]:
        base = self.scene(dag, layout, composed_svg, draft_png, None)
        self.runtime.record("vlm", node="final", label="mock_final_review")
        issues = list(base["issues"])
        if round_index == 0 and self.runtime.chance(self.runtime.error_rate * 0.5):
            victim = self.runtime.choice([n.id for n in dag.placeables() if n.type != BACKGROUND] or [dag.scene_id()])
            issues.append(_issue("draft_drift", f"generation dropped or moved {dag[victim].desc()}", ["final"], "regenerate with stronger structure control"))
        return {"passed": not any(i["severity"] == "major" for i in issues), "issues": issues, "model": "mock"}


class MockJudge:
    def __init__(self, runtime: MockRuntime, cost: Any = None):
        self.runtime = runtime
        self.cost = cost

    def judge(self, prompt: str, checks: List[Dict[str, Any]], image_path: Optional[str], structure: Dict[str, Any]) -> Dict[str, Any]:
        if self.cost is not None:
            self.cost.record("vlm", node="judge", label="mock_judge")
        units = structure.get("units") or []
        by_id = {u["id"]: u for u in units}
        results = []
        for chk in checks:
            ctype = str(chk.get("type") or "presence")
            passed, evidence = self._evaluate(ctype, chk, by_id, units, structure)
            results.append({"id": chk.get("id"), "passed": bool(passed), "evidence": evidence})
        return {"results": results, "model": "mock"}

    def _find(self, key: Optional[str], by_id: Dict[str, Any], units: List[Dict[str, Any]], text: str = "") -> Optional[Dict[str, Any]]:
        if key and key in by_id:
            return by_id[key]
        toks = set(_tokens(text or str(key or "")))
        best, best_score = None, 0.0
        for u in units:
            utoks = set(_tokens(str(u.get("desc") or "")) + [T.singular(t) for t in _tokens(str(u.get("desc") or ""))])
            if key:
                utoks |= set(str(u["id"]).split("_"))
            inter = len(toks & utoks)
            if inter > best_score:
                best, best_score = u, inter
        return best

    def _evaluate(self, ctype: str, chk: Dict[str, Any], by_id, units, structure) -> Tuple[bool, str]:
        if ctype in ("presence", "count", "color", "pose"):
            u = self._find(chk.get("node"), by_id, units, str(chk.get("text") or ""))
            if u is None or not u.get("present", True):
                return False, "unit not found"
            if ctype == "presence":
                return bool(u.get("valid", True)), f"unit {u['id']} present"
            if ctype == "count":
                return int(u.get("count", 0)) == int(chk.get("value", 1)), f"count={u.get('count')}"
            if ctype == "color":
                want = str(chk.get("value") or "")
                found = u.get("colors") or []
                return any(S.color_matches(tuple(c), want) for c in found), f"colors={[S.nearest_color_name(c) for c in found][:5]}"
            if ctype == "pose":
                from .checks import _pose_ok
                return _pose_ok(str(u.get("pose") or ""), str(chk.get("value") or "")), f"pose={u.get('pose')}"
        if ctype in ("left_of", "right_of", "above", "below", "on", "under", "inside", "near", "larger_than", "smaller_than", "in_front_of", "behind"):
            from .checks import relation_holds
            a = self._find(chk.get("a"), by_id, units, str(chk.get("a_text") or chk.get("text") or ""))
            b = self._find(chk.get("b"), by_id, units, str(chk.get("b_text") or ""))
            if a is None or b is None or not a.get("bbox") or not b.get("bbox"):
                return False, "units for relation not found"
            if a is b:
                return True, "same composite unit"
            return relation_holds(ctype, a["bbox"], b["bbox"]), f"{a['id']} {ctype} {b['id']}"
        if ctype == "interaction":
            u = self._find(chk.get("node"), by_id, units, str(chk.get("text") or ""))
            if u is None:
                return False, "interaction unit not found"
            return bool(u.get("valid", True)) and bool(u.get("present", True)), f"interaction {u['id']}"
        return True, "unknown check type, assumed pass"


class MockGenerator:
    def __init__(self, runtime: MockRuntime):
        self.runtime = runtime

    def generate(self, draft_png: str, prompt: str, params: Dict[str, Any], out_path: str) -> Dict[str, Any]:
        img = Image.open(draft_png).convert("RGB")
        seed = int(params.get("seed") or 0)
        rng = random.Random(seed)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        px = blurred.load()
        w, h = blurred.size
        for _ in range(int(w * h * 0.002)):
            x, y = rng.randrange(w), rng.randrange(h)
            r, g, b = px[x, y]
            px[x, y] = (max(0, min(255, r + rng.randint(-25, 25))), max(0, min(255, g + rng.randint(-25, 25))), max(0, min(255, b + rng.randint(-25, 25))))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        blurred.save(out_path, format="PNG")
        meta = {"backend": "mock", "seed": seed, "strength": params.get("strength"), "control_scale": params.get("control_scale"), "prompt": prompt[:200]}
        Path(out_path).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta


class MockMonolithic:
    def __init__(self, runtime: MockRuntime, planner: MockPlanner, drafter: MockDrafter, layouter: MockLayouter, error_scale: float = 1.5):
        self.runtime = runtime
        self.planner = planner
        self.drafter = drafter
        self.layouter = layouter
        self.error_scale = float(error_scale)

    def draft(self, prompt: str, feedback: Sequence[str] = (), hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from .compose import compose

        dag = self.planner.plan(prompt, [], hint)
        self.runtime.record("llm", node="baseline", label="mock_monolithic")
        scale = self.error_scale * (0.6 if feedback else 1.0)
        for nid in dag.topological():
            n = dag[nid]
            if n.type in (SCENE, FINAL):
                continue
            deps = {d: dag[d].output for d in n.deps if dag[d].output}
            n.output = self.drafter.draft(n, dag, deps, feedback, rate_scale=scale)
        layout = self.layouter.layout(dag, feedback, rate_scale=scale)
        dag[dag.scene_id()].output = layout
        composed = compose(dag, layout)
        elements = []
        for i, item in enumerate(composed["legend"]):
            elements.append({"id": f"e{i + 1}", "type": item["type"], "bbox": item["bbox"], "desc": item["desc"], "node": item["node"]})
        return {"svg": composed["svg"], "elements": elements, "canvas": list(dag.canvas), "dag": dag}
