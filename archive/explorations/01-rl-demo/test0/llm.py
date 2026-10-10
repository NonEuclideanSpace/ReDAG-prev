from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config
from .cost import CostMeter


def encode_image(path: str) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    data = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def parse_json(text: str) -> Any:
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start = s.find("{")
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no json object found")
    start = min(starts)
    opener = s[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1])
    raise ValueError("unterminated json")


class OpenAICompatible:
    def __init__(self, cfg: Config, cost: Optional[CostMeter] = None):
        self.cfg = cfg
        self.cost = cost
        self._clients: Dict[str, Any] = {}

    def _client(self, role: str):
        base, key, _model = self.cfg.role_endpoint(role)
        cache_key = f"{base}|{key}"
        if cache_key not in self._clients:
            from openai import OpenAI

            self._clients[cache_key] = OpenAI(api_key=key or "EMPTY", base_url=base, timeout=float(self.cfg.LLM_TIMEOUT))
        return self._clients[cache_key]

    def chat(
        self,
        role: str,
        system: str,
        user: str,
        images: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
        node: Optional[str] = None,
        phase: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> str:
        _base, _key, model = self.cfg.role_endpoint(role)
        content: Any = user
        if images:
            content = [{"type": "text", "text": user}]
            for path in images:
                content.append({"type": "image_url", "image_url": {"url": encode_image(path)}})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        limit = int(max_tokens or (self.cfg.VLM_MAX_TOKENS if images else self.cfg.LLM_MAX_TOKENS))
        if str(model).startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max(limit, 2048)
        else:
            kwargs["max_tokens"] = limit
            if temperature is not None:
                kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        client = self._client(role)
        last_error: Optional[Exception] = None
        for attempt in range(int(self.cfg.LLM_RETRIES) + 1):
            t0 = time.time()
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if json_mode and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                time.sleep(min(8.0, 1.5 * (attempt + 1)))
                continue
            usage = getattr(response, "usage", None)
            text = response.choices[0].message.content or ""
            if self.cost is not None:
                self.cost.record(
                    kind or ("vlm" if images else "llm"),
                    node=node,
                    seconds=time.time() - t0,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    phase=phase,
                    label=f"{role}:{model}",
                )
            return text
        raise RuntimeError(f"llm call failed for role {role}: {last_error}")
