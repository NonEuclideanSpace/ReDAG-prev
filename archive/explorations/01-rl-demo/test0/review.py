from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .compose import legend_text
from .dag import DAG, Node
from .llm import OpenAICompatible, parse_json
from .prompts import FINAL_REVIEW_SYSTEM, NODE_REVIEW_SYSTEM, SCENE_REVIEW_SYSTEM

ISSUE_TYPES = ("missing_object", "count_mismatch", "color_mismatch", "position_mismatch", "size_mismatch", "pose_mismatch", "identity_mismatch", "draft_drift", "other")


def normalize_review(data: Any, default_nodes: Optional[List[str]] = None, model: str = "") -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"passed": False, "issues": [{"type": "other", "severity": "major", "description": "unparseable review", "nodes": list(default_nodes or []), "fix_instruction": "retry"}], "model": model}
    issues: List[Dict[str, Any]] = []
    raw = data.get("issues")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type") or "other").strip().lower()
            if itype not in ISSUE_TYPES:
                itype = "other"
            nodes = item.get("nodes")
            if isinstance(nodes, str):
                nodes = [nodes]
            if not isinstance(nodes, list):
                nodes = []
            issues.append({
                "type": itype,
                "severity": "minor" if str(item.get("severity") or "major").lower().startswith("min") else "major",
                "description": str(item.get("description") or "").strip(),
                "nodes": [str(n) for n in nodes if str(n).strip()],
                "fix_instruction": str(item.get("fix_instruction") or "").strip(),
                "source": "vlm",
            })
    passed = bool(data.get("passed")) if "passed" in data else not any(i["severity"] == "major" for i in issues)
    if any(i["severity"] == "major" for i in issues):
        passed = False
    return {"passed": passed, "issues": issues, "model": model}


class LLMReviewer:
    def __init__(self, llm: OpenAICompatible, model_name: str = ""):
        self.llm = llm
        self.model_name = model_name

    def node(self, node: Node, fragment: str, png: Optional[str], dag: DAG) -> Dict[str, Any]:
        user = (
            f"SCENE PROMPT (context): {dag.prompt}\nNODE id={node.id} type={node.type}\nSPEC: {json.dumps(node.spec, ensure_ascii=False)}\n\n"
            f"SVG FRAGMENT:\n{fragment[:8000]}\n\nThe attached image is the rendered fragment. Return the JSON verdict."
        )
        text = self.llm.chat("review", NODE_REVIEW_SYSTEM, user, images=[png] if png else None, json_mode=True, node=node.id)
        try:
            data = parse_json(text)
        except ValueError:
            data = None
        return normalize_review(data, [node.id], self.model_name)

    def scene(self, dag: DAG, layout: Dict[str, Any], composed_svg: str, png: Optional[str], overlay: Optional[str]) -> Dict[str, Any]:
        legend = [{"node": p["node"], "type": dag[p["node"]].type, "desc": dag[p["node"]].desc(), "count": dag[p["node"]].count(), "bbox": p.get("bbox"), "participants": dag[p["node"]].participants()} for p in layout.get("placements", []) if isinstance(p, dict) and p.get("node") in dag]
        user = (
            f"USER PROMPT: {dag.prompt}\nCANVAS: {dag.canvas[0]}x{dag.canvas[1]}\n\nLEGEND (node ids you must use for attribution; use \"scene\" for layout problems):\n{legend_text(legend)}\n\n"
            "Image 1 is the clean draft, image 2 is the draft with node ids overlaid. Return the JSON verdict."
        )
        images = [p for p in (png, overlay) if p]
        text = self.llm.chat("review", SCENE_REVIEW_SYSTEM, user, images=images or None, json_mode=True, node=dag.scene_id())
        try:
            data = parse_json(text)
        except ValueError:
            data = None
        return normalize_review(data, [dag.scene_id()], self.model_name)

    def final(self, dag: DAG, layout: Dict[str, Any], composed_svg: str, draft_png: Optional[str], final_png: Optional[str], round_index: int) -> Dict[str, Any]:
        legend = [{"node": p["node"], "type": dag[p["node"]].type, "desc": dag[p["node"]].desc(), "count": dag[p["node"]].count(), "bbox": p.get("bbox"), "participants": dag[p["node"]].participants()} for p in layout.get("placements", []) if isinstance(p, dict) and p.get("node") in dag]
        user = (
            f"USER PROMPT: {dag.prompt}\n\nLEGEND of the structural draft:\n{legend_text(legend)}\n\n"
            "Image 1 is the structural draft, image 2 is the final generated image. Return the JSON verdict."
        )
        images = [p for p in (draft_png, final_png) if p]
        text = self.llm.chat("review", FINAL_REVIEW_SYSTEM, user, images=images or None, json_mode=True, node="final")
        try:
            data = parse_json(text)
        except ValueError:
            data = None
        return normalize_review(data, ["final"], self.model_name)
