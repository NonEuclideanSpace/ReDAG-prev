from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .dag import DAG, Node, build_dag
from .llm import OpenAICompatible, parse_json
from .prompts import PLANNER_SYSTEM, REVISE_SPEC_SYSTEM


def catalog_text(catalog: List[Dict[str, Any]], limit: int = 40) -> str:
    if not catalog:
        return "(empty)"
    lines = []
    for entry in catalog[:limit]:
        lines.append(f"- type={entry.get('type')} spec={json.dumps(entry.get('spec'), ensure_ascii=False)}")
    return "\n".join(lines)


class LLMPlanner:
    def __init__(self, llm: OpenAICompatible, retries: int = 2):
        self.llm = llm
        self.retries = int(retries)

    def plan(self, prompt: str, catalog: Optional[List[Dict[str, Any]]] = None, hint: Optional[Dict[str, Any]] = None) -> DAG:
        if hint:
            return build_dag(prompt, hint)
        user = f"USER PROMPT:\n{prompt}\n\nREUSE CATALOG (specs already drawn and cached):\n{catalog_text(catalog or [])}\n\nReturn the DAG JSON."
        last_error = ""
        for attempt in range(self.retries + 1):
            text = self.llm.chat("planner", PLANNER_SYSTEM, user + (f"\n\nYour previous answer was rejected: {last_error}. Fix it." if last_error else ""), json_mode=True, phase="plan", node="planner")
            try:
                data = parse_json(text)
                return build_dag(prompt, data)
            except (ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)[:500]
        raise RuntimeError(f"planner failed: {last_error}")

    def revise_spec(self, node: Node, issues: List[Dict[str, Any]], dag: DAG) -> Dict[str, Any]:
        user = (
            f"USER PROMPT:\n{dag.prompt}\n\nNODE id={node.id} type={node.type}\nCURRENT SPEC:\n{json.dumps(node.spec, ensure_ascii=False, indent=2)}\n\n"
            f"ISSUES ATTRIBUTED TO THIS NODE:\n{json.dumps(issues, ensure_ascii=False, indent=2)}\n\nReturn the revised spec JSON."
        )
        text = self.llm.chat("planner", REVISE_SPEC_SYSTEM, user, json_mode=True, node=node.id, phase="repair")
        try:
            data = parse_json(text)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            spec = json.loads(json.dumps(node.spec))
            notes = list(spec.get("notes") or [])
            notes.extend(str(i.get("fix_instruction") or i.get("description") or "") for i in issues)
            spec["notes"] = [n for n in notes if n][-6:]
            return spec
        spec = dict(data)
        for key in ("count", "colors", "participants", "poses"):
            if key in node.spec and key not in spec:
                spec[key] = node.spec[key]
        if "desc" not in spec:
            spec["desc"] = node.spec.get("desc", node.id)
        if node.type == "scene" and not isinstance(spec.get("relations"), list):
            spec["relations"] = node.spec.get("relations", [])
        return spec
