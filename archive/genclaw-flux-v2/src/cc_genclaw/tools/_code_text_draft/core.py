from __future__ import annotations

from pathlib import Path

from .config import set_gen_backend
from .router import classify_text_render_type as classify
from .pipelines import run_single_case


def render_text_image(
    prompt: str,
    expected_texts: list[str],
    output_dir: str | Path,
    category: str = "",
    gen_backend: str = "gemini",
) -> dict:
    set_gen_backend(gen_backend)

    case = {
        "id": Path(output_dir).name,
        "prompt": prompt,
        "expected_texts": expected_texts,
        "category": category,
    }

    log = run_single_case(case, Path(output_dir).parent)

    final_path = Path(output_dir) / "final.png"
    return {
        "case_id": case["id"],
        "render_type": log.get("render_type", "unknown"),
        "final_path": str(final_path) if final_path.exists() else None,
        "error": log.get("error"),
        "log": log,
    }

