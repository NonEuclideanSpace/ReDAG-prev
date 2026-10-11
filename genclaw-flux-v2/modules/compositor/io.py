"""Versioned JSON IR with source hashes, crop bounds and dependency hooks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .layer import ImageLayer
from .renderer import prepare_layer


def write_json(path, payload):
    path = Path(path)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")
    return path


def asset_record(path, base):
    path = Path(path).resolve()
    return {"source": os.path.relpath(path, base),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save_scene(scene, path):
    path = Path(path).resolve()
    layers = []
    for layer in scene.layers:
        source = asset_record(layer.image_path, path.parent)
        if source["sha256"] != layer.source_sha256:
            raise ValueError(f"Source changed after loading: {layer.name}")
        transformed, (x, y) = prepare_layer(layer)
        layers.append({"node_id": layer.name, "name": layer.name, **source,
                       "source_size": list(layer.source_size),
                       "alpha_crop_bbox": list(layer.crop_bbox), "trim_alpha": layer.trim_alpha,
                       "position": list(layer.position), "scale": layer.scale,
                       "rotation": layer.rotation, "z_order": layer.z_order,
                       "opacity": layer.opacity,
                       "raster_bbox": [x, y, x + transformed.width, y + transformed.height],
                       "depends_on": layer.depends_on, "metadata": layer.metadata})
    background = asset_record(scene.background_path, path.parent) if scene.background_path else None
    return write_json(path, {
        "schema_version": 1, "composition_id": scene.composition_id,
        "coordinate_system": "pixels; top-left origin; position before clockwise center rotation",
        "alpha_convention": "0 transparent, 255 opaque", "blend_mode": "linear_rgb_source_over",
        "canvas": {"size": list(scene.canvas_size), "background_color": list(scene.background_color),
                   "background": background},
        "layers": layers, "relations": scene.relations, "metadata": scene.metadata})


def load_scene(path, verify_sources=True):
    from .composition import SceneComposition
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported scene IR version")

    def resolve(record):
        source = (path.parent / record["source"]).resolve()
        if verify_sources and hashlib.sha256(source.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")
        return source

    canvas = data["canvas"]
    background = resolve(canvas["background"]) if canvas.get("background") else None
    scene = SceneComposition(tuple(canvas["size"]), tuple(canvas["background_color"]), background,
                             data.get("metadata"), data.get("relations"), data["composition_id"])
    for spec in data["layers"]:
        scene.add_layer(ImageLayer(
            name=spec["name"], image_path=str(resolve(spec)), position=tuple(spec["position"]),
            scale=spec["scale"], rotation=spec.get("rotation", 0), z_order=spec["z_order"],
            opacity=spec.get("opacity", 1), metadata=spec.get("metadata", {}),
            depends_on=spec.get("depends_on", []), trim_alpha=spec.get("trim_alpha", True)))
    return scene
