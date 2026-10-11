"""Small adapter to the existing GenClaw Qwen worker; no model implementation."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

from PIL import Image


@contextmanager
def _environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def refine_layout(layout_path, prompt, output_path, base_url=None, seed=261011, steps=30):
    """Send exactly ONE composited layout PNG, not a list of original assets.

    Callers using the repo-local module need both repository root and src on
    PYTHONPATH (the example does this). This synchronous adapter changes worker
    options temporarily; concurrent callers should use separate processes.
    """
    from cc_genclaw.config import load_genclaw_config
    from cc_genclaw.tools._image_gen.qwen_remote import request_image

    load_genclaw_config()
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if steps <= 0:
        raise ValueError("Steps must be positive")
    layout_path = Path(layout_path).resolve()
    with Image.open(layout_path) as image:
        width, height = image.size
        if min(width, height) < 64 or max(width, height) > 1536 or width % 32 or height % 32:
            raise ValueError("Qwen demo canvas must be 64..1536 pixels, multiples of 32")
        if image.convert("RGBA").getchannel("A").getextrema()[0] < 255:
            raise ValueError("Qwen layout needs an opaque background; keep alpha in foreground.png")
    cfg = {"base_url": base_url or os.environ.get("QWEN_IMAGE_BASE_URL") or "http://127.0.0.1:8190",
           "api_key": os.environ.get("QWEN_IMAGE_API_KEY", "")}
    options = {"QWEN_IMAGE_WIDTH": width, "QWEN_IMAGE_HEIGHT": height,
               "QWEN_IMAGE_SEED": seed, "QWEN_IMAGE_STEPS": steps,
               "QWEN_IMAGE_GUIDANCE": 1.0, "QWEN_IMAGE_ALLOW_MOCK": "0"}
    with _environment(options):
        return request_image(prompt, [str(layout_path)], "qwen-image", cfg, str(output_path))
