from __future__ import annotations

import shutil
from pathlib import Path


def _generate(prompt: str, output_path: str, images=None) -> str:
    from .._image_gen import generate_image
    generated = Path(generate_image(prompt, images)).resolve()
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if generated != target:
        shutil.copy2(generated, target)
        metadata = generated.with_suffix(".json")
        if metadata.is_file():
            shutil.copy2(metadata, target.with_suffix(".json"))
    return str(target)


def t2i_generate(prompt: str, output_path: str) -> str:
    return _generate(prompt, output_path)


def i2i_refine(image_path: str, prompt: str, description: str, output_path: str,
               custom_instruction: str = "") -> str:
    instruction = custom_instruction.strip() or (
        f"Transform this into a photorealistic image. Target: {prompt}. "
        f"Layout: {description}. Preserve spatial arrangement and all text content exactly as shown."
    )
    return _generate(instruction, output_path, [image_path])

