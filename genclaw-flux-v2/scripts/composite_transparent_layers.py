"""Compose RGBA PNG layers using ComfyUI-style normal source-over blending.

The JSON manifest specifies a canvas, an optional background, and layers with
path, x, y, width, height, z_index, opacity, and optional inverted mask.
RGB is blended in linear-light space; existing alpha is always respected.
This is a small standalone normal-blend compositor, not a ComfyUI runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92,
                    ((np.maximum(rgb, 0.0) + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.0031308, rgb * 12.92,
                    1.055 * np.maximum(rgb, 0.0) ** (1.0 / 2.4) - 0.055).astype(np.float32)


def resolve_path(manifest_path: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else manifest_path.parent / path


def image_to_linear(image: Image.Image) -> np.ndarray:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    return np.concatenate((srgb_to_linear(rgba[..., :3]), rgba[..., 3:4]), axis=-1)


def linear_to_image(rgba: np.ndarray) -> Image.Image:
    rgb = linear_to_srgb(np.clip(rgba[..., :3], 0, 1))
    out = np.concatenate((rgb, np.clip(rgba[..., 3:4], 0, 1)), axis=-1)
    return Image.fromarray(np.rint(out * 255).astype(np.uint8), "RGBA")


def source_over(canvas: np.ndarray, layer: np.ndarray, x: int, y: int,
                opacity: float = 1.0) -> None:
    """Modify the canvas in place, clipping the placed layer to its bounds."""
    if not 0 <= opacity <= 1:
        raise ValueError("Layer opacity must be between 0 and 1")
    height, width = canvas.shape[:2]
    layer_height, layer_width = layer.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + layer_width), min(height, y + layer_height)
    if x0 >= x1 or y0 >= y1:
        return
    source = layer[y0 - y:y1 - y, x0 - x:x1 - x]
    backdrop = canvas[y0:y1, x0:x1]
    source_alpha = source[..., 3:4] * opacity
    backdrop_alpha = backdrop[..., 3:4]
    out_alpha = source_alpha + backdrop_alpha * (1 - source_alpha)
    premultiplied = (source[..., :3] * source_alpha
                     + backdrop[..., :3] * backdrop_alpha * (1 - source_alpha))
    out_rgb = np.divide(premultiplied, out_alpha,
                        out=np.zeros_like(premultiplied), where=out_alpha > 0)
    backdrop[..., :3] = out_rgb
    backdrop[..., 3:4] = out_alpha


def prepare_layer(manifest_path: Path, spec: dict) -> Image.Image:
    if "path" not in spec:
        raise ValueError("Every layer needs a path")
    with Image.open(resolve_path(manifest_path, spec["path"])) as source:
        image = source.convert("RGBA")
    width = int(spec.get("width", image.width))
    height = int(spec.get("height", image.height))
    if width <= 0 or height <= 0:
        raise ValueError("Layer width and height must be positive")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    if "mask" in spec:
        with Image.open(resolve_path(manifest_path, spec["mask"])) as source_mask:
            mask = source_mask.convert("L").resize(image.size, Image.Resampling.BILINEAR)
        # ComfyUI's optional mask convention: 255 hides, 0 keeps.
        alpha = np.asarray(image.getchannel("A"), dtype=np.float32)
        inverse = 1 - np.asarray(mask, dtype=np.float32) / 255.0
        image.putalpha(Image.fromarray(np.rint(alpha * inverse).astype(np.uint8), "L"))
    return image


def render(manifest_path: Path, stage_dir: Path | None = None,
           background_override: Path | None = None,
           skip_background: bool = False) -> Image.Image:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    width, height = (int(v) for v in manifest["canvas"])
    if width <= 0 or height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    canvas = np.zeros((height, width, 4), dtype=np.float32)
    background = background_override or manifest.get("background")
    if background and not skip_background:
        background_path = (Path(background_override) if background_override is not None
                           else resolve_path(manifest_path, background))
        with Image.open(background_path) as source:
            if source.size != (width, height):
                raise ValueError("Background size must match the canvas exactly")
            source_over(canvas, image_to_linear(source), 0, 0)
    layers = manifest.get("layers", [])
    ordered = sorted(enumerate(layers), key=lambda entry: int(entry[1].get("z_index", 0)))
    for stage_number, (index, layer_spec) in enumerate(ordered, start=1):
        if layer_spec.get("visible", True) is False:
            continue
        image = prepare_layer(manifest_path, layer_spec)
        source_over(canvas, image_to_linear(image), int(layer_spec.get("x", 0)),
                    int(layer_spec.get("y", 0)), float(layer_spec.get("opacity", 1)))
        if stage_dir is not None:
            stage_dir.mkdir(parents=True, exist_ok=True)
            name = str(layer_spec.get("name", f"layer_{index}")).replace("/", "_")
            linear_to_image(canvas).save(stage_dir / f"{stage_number:02d}_{name}.png")
    return linear_to_image(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-stages", type=Path)
    parser.add_argument("--background-override", type=Path,
                        help="Use a generated background while preserving all manifest layers")
    parser.add_argument("--foreground-mask-output", type=Path,
                        help="Save the combined foreground alpha mask (white means protected)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}; use --overwrite to replace it")
    if args.foreground_mask_output and args.foreground_mask_output.exists() and not args.overwrite:
        parser.error(f"Mask already exists: {args.foreground_mask_output}; use --overwrite to replace it")
    result = render(args.manifest.resolve(), args.save_stages,
                    background_override=args.background_override)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    if args.foreground_mask_output:
        foreground = render(args.manifest.resolve(), skip_background=True)
        args.foreground_mask_output.parent.mkdir(parents=True, exist_ok=True)
        foreground.getchannel("A").save(args.foreground_mask_output)
    print(json.dumps({"output": str(args.output.resolve()), "size": result.size,
                      "mode": result.mode, "layers": len(json.loads(args.manifest.read_text())["layers"])},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
