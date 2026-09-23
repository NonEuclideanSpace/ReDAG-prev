from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .llm import OpenAICompatible, parse_json
from .prompts import JUDGE_SYSTEM


def checklist_text(checks: List[Dict[str, Any]]) -> str:
    lines = []
    for chk in checks:
        lines.append(f"- id={chk.get('id')}: {chk.get('text')}")
    return "\n".join(lines) if lines else "(none)"


def score(results: List[Dict[str, Any]], checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {str(r.get("id")): bool(r.get("passed")) for r in results if isinstance(r, dict)}
    total = len(checks)
    passed = sum(1 for chk in checks if by_id.get(str(chk.get("id")), False))
    return {"passed": passed, "total": total, "pass_rate": (passed / total) if total else 0.0}


class LLMJudge:
    def __init__(self, llm: OpenAICompatible):
        self.llm = llm

    def judge(self, prompt: str, checks: List[Dict[str, Any]], image_path: Optional[str], structure: Dict[str, Any]) -> Dict[str, Any]:
        if not image_path:
            return {"results": [{"id": c.get("id"), "passed": False, "evidence": "no image"} for c in checks], "model": "none"}
        user = f"PROMPT: {prompt}\n\nCHECKLIST:\n{checklist_text(checks)}\n\nThe attached image is the generated result. Return the JSON results."
        text = self.llm.chat("judge", JUDGE_SYSTEM, user, images=[image_path], json_mode=True, node="judge")
        try:
            data = parse_json(text)
        except ValueError:
            data = {}
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            results = [{"id": c.get("id"), "passed": False, "evidence": "judge output unparseable"} for c in checks]
        return {"results": results, "model": "judge"}
