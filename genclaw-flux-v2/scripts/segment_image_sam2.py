"""Extract a prompted object into an RGBA PNG with local SAM 2.1 weights.

Run with the project's GPU environment, e.g.:
  /home/lipz/miniconda3/envs/uno_env/bin/python scripts/segment_image_sam2.py \
    --image input.png --output object.png --box 100 80 500 420

Coordinates are original-image pixels. SAM2 is promptable, not a text detector:
provide at least one box or positive point to identify the desired object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter


DEFAULT_MODEL = Path("/home/hdd/lpz/model/sam2.1")


def validate_prompts(size: tuple[int, int], box, points, negative_points) -> None:
    width, height = size
    if box is None and not points:
        raise ValueError("Provide --box or at least one --point to select an object")
    if box is not None:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Box must lie inside the {width}x{height} image")
    for x, y in [*points, *negative_points]:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Point ({x}, {y}) is outside the {width}x{height} image")


def make_cutout(image: Image.Image, mask: Image.Image, *, feather: float = 0,
                crop: bool = False, margin: int = 0) -> Image.Image:
    """Preserve the source RGB pixels and apply a real alpha mask."""
    if mask.size != image.size:
        raise ValueError("Mask and source image must have the same dimensions")
    if feather < 0 or margin < 0:
        raise ValueError("Feather and crop margin must be nonnegative")
    alpha = mask.convert("L")
    if feather:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("SAM2 returned an empty mask")
    result = image.convert("RGBA")
    result.putalpha(alpha)
    if crop:
        x0, y0, x1, y1 = bounds
        width, height = image.size
        result = result.crop((max(0, x0 - margin), max(0, y0 - margin),
                              min(width, x1 + margin), min(height, y1 + margin)))
    return result


def segment(image: Image.Image, model_dir: Path, box, points, negative_points,
            device: str) -> tuple[Image.Image, float]:
    import torch
    from transformers import Sam2Model, Sam2Processor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"SAM2 weights not found: {model_dir / 'model.safetensors'}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    processor = Sam2Processor.from_pretrained(model_dir, local_files_only=True)
    model = Sam2Model.from_pretrained(model_dir, local_files_only=True).to(device).eval()

    kwargs = {}
    if box is not None:
        kwargs["input_boxes"] = [[list(box)]]
    if points or negative_points:
        all_points = [*points, *negative_points]
        kwargs["input_points"] = [[[list(point) for point in all_points]]]
        kwargs["input_labels"] = [[[1] * len(points) + [0] * len(negative_points)]]
    inputs = processor(images=image.convert("RGB"), return_tensors="pt", **kwargs).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    scores = outputs.iou_scores[0, 0].float().cpu()
    best = int(scores.argmax())
    masks = processor.post_process_masks(outputs.pred_masks.float().cpu(),
                                         inputs["original_sizes"])[0]
    mask = masks[0, best].to(torch.uint8).numpy() * 255
    return Image.fromarray(mask, mode="L"), float(scores[best])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="New transparent PNG path")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--point", type=float, nargs=2, action="append", default=[], metavar=("X", "Y"))
    parser.add_argument("--negative-point", type=float, nargs=2, action="append", default=[], metavar=("X", "Y"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--crop", action="store_true", help="Crop transparent padding around the object")
    parser.add_argument("--margin", type=int, default=8, help="Extra pixels around a cropped object")
    parser.add_argument("--feather", type=float, default=0, help="Optional mask edge blur radius in pixels")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output PNG")
    args = parser.parse_args()

    if args.image.resolve() == args.output.resolve():
        parser.error("Output must differ from the source image")
    if args.output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.output}; use --overwrite to replace it")
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    try:
        validate_prompts(image.size, args.box, args.point, args.negative_point)
        mask, score = segment(image, args.model_dir, args.box, args.point,
                              args.negative_point, args.device)
        cutout = make_cutout(image, mask, feather=args.feather,
                             crop=args.crop, margin=args.margin)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cutout.save(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "size": cutout.size,
                      "mode": cutout.mode, "iou_score": round(score, 4),
                      "mask_pixels": mask.histogram()[255],
                      "source_size": image.size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
