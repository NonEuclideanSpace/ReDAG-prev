"""An editable scene intermediate representation, not just a flattened PNG."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from .renderer import render_scene


class SceneComposition:
    def __init__(self, canvas_size=(1024, 576), background_color=(128, 128, 128, 255),
                 background_path=None, metadata=None, relations=None, composition_id=None):
        if len(canvas_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in canvas_size):
            raise ValueError("Canvas dimensions must be positive integers")
        if len(background_color) != 4 or any(not isinstance(v, int) or not 0 <= v <= 255
                                             for v in background_color):
            raise ValueError("Background color must be four 8-bit RGBA integers")
        self.canvas_size = tuple(canvas_size)
        self.background_color = tuple(background_color)
        self.background_path = str(Path(background_path).resolve()) if background_path else None
        self.metadata = metadata or {}
        self.relations = relations or []
        self.composition_id = composition_id or uuid4().hex
        self.layers = []

    def add_layer(self, layer):
        if any(existing.name == layer.name for existing in self.layers):
            raise ValueError(f"Duplicate layer name: {layer.name}")
        self.layers.append(layer)
        return self

    def ordered_layers(self):
        # Python sort is stable: same z_order retains insertion order.
        return sorted(self.layers, key=lambda layer: layer.z_order)

    def render(self, output=None, ir_path=None):
        image = render_scene(self)
        if output is not None:
            from .io import save_scene
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_scene(self, ir_path or path.parent / "scene_ir.json")
            image.save(path, format="PNG")
        elif ir_path is not None:
            self.save_ir(ir_path)
        return image

    def render_foreground(self):
        return render_scene(self, foreground_only=True)

    def save_ir(self, path):
        from .io import save_scene
        return save_scene(self, path)

    @classmethod
    def from_ir(cls, path, verify_sources=True):
        from .io import load_scene
        return load_scene(path, verify_sources=verify_sources)
