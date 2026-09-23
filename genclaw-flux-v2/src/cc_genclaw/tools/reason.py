from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..runtime.artifacts import record_artifacts_in_dir
from ..runtime.session_dir import tool_output_dir
from .base import BaseTool
from ._reason import run_reasoning


class ReasonTool(BaseTool):
    name = "reason"
    description = ""
    parameters_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "reference_image_paths": {"type": "array", "items": {"type": "string"}},
            "user_image_path": {"type": "string"},
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    def execute(self, args: Dict[str, Any], signal: Any = None):
        from cc_genclaw.config import load_genclaw_config
        load_genclaw_config()

        prompt = str(args.get("prompt", "")).strip()
        if not prompt:
            return self.err("reason requires non-empty 'prompt'")

        user_image = str(args.get("user_image_path", "") or "")
        refs = list(args.get("reference_image_paths") or [])

        try:
            raw = run_reasoning(
                prompt=prompt,
                user_image_path=user_image or None,
                reference_image_paths=refs or None,
            )
        except Exception as e:
            return self.err(f"reason: run_reasoning failed: {type(e).__name__}: {e}")

        if not isinstance(raw, dict):
            raw = {}

        def _as_list(v) -> List[str]:
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            if isinstance(v, str) and v.strip():
                return [v]
            return []

        reasoning = _as_list(raw.get("reasoning_knowledge"))

        out_dir = tool_output_dir("reason")
        try:
            Path(out_dir, "meta.json").write_text(json.dumps({
                "raw": raw, "prompt_preview": prompt[:1000]
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            record_artifacts_in_dir(out_dir, tool_name="reason")
        except Exception:
            pass

        payload: Dict[str, Any] = {
            "reasoning_output": reasoning,
            "status": "success" if reasoning else "failed",
        }
        if payload["status"] != "success":
            payload["error"] = "reason returned empty reasoning_output"

        return self.ok(json.dumps(payload, ensure_ascii=False)) if payload["status"] == "success" \
            else self.err(json.dumps(payload, ensure_ascii=False))

