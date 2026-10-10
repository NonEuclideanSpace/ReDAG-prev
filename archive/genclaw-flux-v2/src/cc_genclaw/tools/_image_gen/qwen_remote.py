from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image, ImageOps


def _number(name: str, default: Any, convert):
    value = os.environ.get(name, "").strip()
    return convert(value) if value else default


def _env(name: str, fallback: str = "") -> str:
    return os.environ.get(name) or os.environ.get(fallback, "")


def _number_any(names: List[str], default: Any, convert):
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return convert(value)
    return default


def _budgeted_size(src_width: int, src_height: int, budget: int):
    if src_width * src_height <= budget:
        width = min(1536, max(64, (src_width // 16) * 16))
        height = min(1536, max(64, (src_height // 16) * 16))
        return width, height
    aspect = min(2.0, max(0.5, src_width / float(src_height)))
    side = budget ** 0.5
    width = int(round(side * aspect ** 0.5))
    height = int(round(side / aspect ** 0.5))
    width = min(1536, max(64, (width // 16) * 16))
    height = min(1536, max(64, (height // 16) * 16))
    return width, height


def request_image(prompt: str, reference_images: Optional[List[str]], model: str,
                  cfg: Dict[str, Any], output_path: str) -> str:
    refs = reference_images or []
    if len(refs) > 8:
        raise RuntimeError("Qwen diffusion worker accepts at most 8 reference images; none were sent.")
    width = _number_any(["QWEN_IMAGE_WIDTH", "FLUX_WIDTH"], 768, int)
    height = _number_any(["QWEN_IMAGE_HEIGHT", "FLUX_HEIGHT"], 768, int)
    encoded = []
    for index, path in enumerate(refs):
        with Image.open(path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
        if index == 0:
            width, height = _budgeted_size(img.width, img.height, width * height)
        img.thumbnail((1536, 1536))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

    body = {"model": model, "prompt": prompt, "images": encoded,
            "width": width, "height": height,
            "seed": _number_any(["QWEN_IMAGE_SEED", "FLUX_SEED"], None, int),
            "steps": _number_any(["QWEN_IMAGE_STEPS", "FLUX_STEPS"], 28, int),
            "guidance": _number_any(["QWEN_IMAGE_GUIDANCE", "FLUX_GUIDANCE"],
                                    30.0 if model in {"qwen-image", "flux-fill"} else 3.5, float),
            "strength": _number_any(["QWEN_IMAGE_STRENGTH", "FLUX_STRENGTH"], 0.6, float)}
    base_url = str(cfg.get("base_url") or "http://127.0.0.1:8188").rstrip("/")
    headers = {}
    token = cfg.get("api_key") or ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    timeout = _number_any(["QWEN_IMAGE_TIMEOUT", "FLUX_TIMEOUT"], 900.0, float)
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(base_url + "/v1/generate", json=body,
                                    headers=headers, timeout=(10, timeout))
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Cannot connect to Qwen diffusion worker at {base_url}. Start the worker and tunnel.") from exc
    except requests.Timeout as exc:
        raise RuntimeError("Qwen diffusion request timed out. It may still be running remotely; it was not retried.") from exc
    if not response.ok:
        try:
            detail = response.json().get("detail", "request rejected")
        except ValueError:
            detail = response.text[:400]
        raise RuntimeError(f"Qwen diffusion HTTP {response.status_code}: {str(detail)[:600]}")
    try:
        payload = response.json()
        raw = base64.b64decode(payload["image_base64"], validate=True)
        with Image.open(io.BytesIO(raw)) as generated:
            generated.load()
            image = generated.convert("RGB")
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise RuntimeError("Qwen diffusion worker returned an invalid image response.") from exc
    if payload.get("backend") == "mock" and _env("QWEN_IMAGE_ALLOW_MOCK", "FLUX_ALLOW_MOCK") != "1":
        raise RuntimeError("Worker is in mock mode. Set QWEN_IMAGE_ALLOW_MOCK=1 only for an explicit transport test.")
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    metadata = {k: v for k, v in payload.items() if k != "image_base64"}
    metadata["reference_count"] = len(refs)
    target.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)
