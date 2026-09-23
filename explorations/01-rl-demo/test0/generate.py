from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from .config import Config
from .cost import CostMeter
from .dag import BACKGROUND, DAG
from .prompts import GENERATION_STYLE


def generation_prompt(dag: DAG, legend: List[Dict[str, Any]]) -> str:
    parts = [dag.prompt.strip()]
    details = []
    for item in legend:
        if item["type"] == BACKGROUND:
            details.append(f"background: {item['desc']}")
        else:
            details.append(item["desc"])
    if details:
        parts.append("Contents: " + "; ".join(details) + ".")
    parts.append(GENERATION_STYLE)
    return " ".join(parts)


def _budgeted_size(width: int, height: int, budget: int = 1024 * 1024) -> tuple:
    aspect = min(2.0, max(0.5, width / float(height)))
    side = budget ** 0.5
    w = int(round(side * aspect ** 0.5))
    h = int(round(side / aspect ** 0.5))
    w = min(1536, max(256, (w // 16) * 16))
    h = min(1536, max(256, (h // 16) * 16))
    return w, h


class RemoteGenerator:
    def __init__(self, cfg: Config, cost: Optional[CostMeter] = None):
        self.cfg = cfg
        self.cost = cost

    def generate(self, draft_png: str, prompt: str, params: Dict[str, Any], out_path: str) -> Dict[str, Any]:
        t0 = time.time()
        with Image.open(draft_png) as source:
            img = source.convert("RGB")
            width, height = _budgeted_size(img.width, img.height)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        body = {
            "model": str(self.cfg.GENERATE_MODEL),
            "prompt": prompt,
            "images": [encoded],
            "width": width,
            "height": height,
            "seed": int(params.get("seed") if params.get("seed") is not None else self.cfg.GENERATE_SEED),
            "steps": int(params.get("steps") or self.cfg.GENERATE_STEPS),
            "guidance": float(params.get("guidance") or self.cfg.GENERATE_GUIDANCE),
            "strength": float(params.get("strength") or self.cfg.GENERATE_STRENGTH),
        }
        if not str(self.cfg.GENERATE_MODEL).startswith("flux"):
            body["control_scale"] = float(params.get("control_scale") or self.cfg.GENERATE_CONTROL_SCALE)
        base_url = str(self.cfg.GENERATE_BASE_URL).rstrip("/")
        headers = {}
        token = str(self.cfg.GENERATE_API_KEY or "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.post(base_url + "/v1/generate", json=body, headers=headers, timeout=(10, float(self.cfg.GENERATE_TIMEOUT)))
        except requests.ConnectionError as exc:
            raise RuntimeError(f"cannot connect to generation server at {base_url}") from exc
        except requests.Timeout as exc:
            raise RuntimeError("generation request timed out") from exc
        if not response.ok:
            try:
                detail = response.json().get("detail", "request rejected")
            except ValueError:
                detail = response.text[:400]
            raise RuntimeError(f"generation server HTTP {response.status_code}: {str(detail)[:600]}")
        payload = response.json()
        raw = base64.b64decode(payload["image_base64"], validate=True)
        with Image.open(io.BytesIO(raw)) as generated:
            generated.load()
            image = generated.convert("RGB")
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG")
        meta = {k: v for k, v in payload.items() if k != "image_base64"}
        meta.update({"prompt": prompt, "params": body | {"images": f"[{len(body['images'])} image]"}})
        target.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.cost is not None:
            self.cost.record("diffusion", node="final", seconds=time.time() - t0, label=str(payload.get("backend") or self.cfg.GENERATE_MODEL))
        return meta
