"""RGBA references -> explicit Scene IR/layout -> existing Qwen-Image worker."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from modules.compositor import ImageLayer, SceneComposition
from modules.compositor.generation import refine_layout
from modules.compositor.io import asset_record, write_json


def build_scene(reference_dir, layout=None, canvas=(1024, 576)):
    """Explicit layout takes priority. Auto mode is only a non-semantic row."""
    reference_dir = Path(reference_dir).resolve()
    if not reference_dir.is_dir():
        raise ValueError(f"Reference directory does not exist: {reference_dir}")
    layout = layout or {}
    canvas = tuple(layout.get("canvas", canvas))
    background = layout.get("background")
    if background:
        background = (reference_dir / background).resolve()
    scene = SceneComposition(canvas, tuple(layout.get("background_color", [128, 128, 128, 255])),
                             background, metadata={"layout_mode": "explicit" if layout else "auto_row",
                                                   **layout.get("metadata", {})},
                             relations=layout.get("relations", []))
    specs = layout.get("layers")
    if specs is None:
        specs = [{"name": p.stem, "source": p.name}
                 for p in sorted(reference_dir.iterdir()) if p.suffix.lower() == ".png"]
    if not specs:
        raise ValueError("No reference PNG layers found")
    for index, spec in enumerate(specs):
        source = (reference_dir / spec["source"]).resolve()
        if source.suffix.lower() != ".png":
            raise ValueError("References must be PNG files")
        layer = ImageLayer(name=spec.get("name", source.stem), image_path=str(source),
                           position=tuple(spec.get("position", [0, 0])),
                           scale=spec.get("scale", 1), rotation=spec.get("rotation", 0),
                           z_order=spec.get("z_order", index), opacity=spec.get("opacity", 1),
                           metadata=spec.get("metadata", {}), depends_on=spec.get("depends_on", []),
                           trim_alpha=spec.get("trim_alpha", True))
        # A solid PNG is not a transparent reference; never silently cut it out.
        from PIL import Image
        with Image.open(source) as original:
            if original.convert("RGBA").getchannel("A").getextrema()[0] == 255:
                raise ValueError(f"Reference lacks transparent pixels: {source}")
        if not layout:
            cell = canvas[0] / len(specs)
            layer.scale = min(cell * 0.82 / layer.image.width, canvas[1] * 0.65 / layer.image.height)
            w, h = layer.scaled_size
            layer.position = (round(cell * (index + 0.5) - w / 2), round(canvas[1] * 0.9 - h))
        scene.add_layer(layer)
    return scene


def run(args):
    layout = json.loads(args.layout.read_text(encoding="utf-8")) if args.layout else None
    scene = build_scene(args.reference_dir, layout, (args.width, args.height))
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not prompt or not prompt.strip():
        raise ValueError("Supply a nonempty --prompt or --prompt-file")
    output = args.output_dir or ROOT / "experiments" / "layer_composition_poc" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6])
    output = output.resolve()
    # Protect all previous experiments; choose a fresh directory for each run.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(prompt, encoding="utf-8")
    scene.metadata["prompt_file"] = "prompt.txt"
    scene.render(output / "scene_layout.png")
    foreground = scene.render_foreground()
    foreground.save(output / "foreground.png")
    foreground.getchannel("A").save(output / "foreground_mask.png")
    report = {
        "node_id": scene.composition_id, "node_type": "layer_composition",
        "depends_on": [layer.name for layer in scene.layers],
        "input_artifacts": [asset_record(layer.image_path, output) for layer in scene.layers],
        "output_artifacts": ["scene_ir.json", "scene_layout.png", "foreground.png",
                             "foreground_mask.png", "prompt.txt"],
        "generation": {"status": "not_requested" if args.layout_only else "pending",
                       "input_artifacts": ["scene_layout.png"],
                       "prompt_file": "prompt.txt", "seed": args.seed, "steps": args.steps,
                       "model": "qwen-image", "base_url": args.base_url,
                       "guidance": 1.0, "output_artifacts": []},
        "limitations": ["Auto layout is a row, not semantic relation or pose planning.",
                        "Qwen receives one flattened PNG; Scene IR is not a model control input.",
                        "Visual conditioning is soft: identity, pose and layout are not hard locked.",
                        "Current QwenImage21Pipeline has no mask/strength-based preservation here.",
                        "No efficacy claim without a controlled multi-person comparison."]}
    write_json(output / "artifacts.json", report)
    if not args.layout_only:
        print(f"Generating with one layout reference: {output / 'scene_layout.png'}", flush=True)
        try:
            refine_layout(output / "scene_layout.png", prompt, output / "final.png",
                          args.base_url, args.seed, args.steps)
            metadata = json.loads((output / "final.json").read_text(encoding="utf-8"))
            report["generation"].update(status="completed", worker_metadata=metadata,
                                         output_artifacts=["final.png", "final.json"])
            report["output_artifacts"] += ["final.png", "final.json"]
        except Exception as exc:
            report["generation"].update(status="failed", error=str(exc))
            raise
        finally:
            write_json(output / "artifacts.json", report)
    print(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", "--reference-dir", type=Path, required=True)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument("--layout", type=Path, help="Explicit placements JSON; sources relative to reference_dir")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--base-url", default=None, help="Existing Qwen worker URL; defaults to project config")
    parser.add_argument("--seed", type=int, default=261011)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--layout-only", action="store_true", help="Offline composition without GPU/API")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Demo failed: {exc}\n")


if __name__ == "__main__":
    main()
