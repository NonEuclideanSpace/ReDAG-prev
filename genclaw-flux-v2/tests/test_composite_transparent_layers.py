"""Deterministic checks for the ComfyUI-inspired normal compositor."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from composite_transparent_layers import (image_to_linear, linear_to_image,
                                          prepare_layer, render, source_over)


def test_linear_light_source_over_and_alpha():
    canvas = image_to_linear(Image.new("RGBA", (1, 1), (0, 0, 0, 255)))
    white = image_to_linear(Image.new("RGBA", (1, 1), (255, 255, 255, 128)))
    source_over(canvas, white, 0, 0)
    pixel = linear_to_image(canvas).getpixel((0, 0))
    assert 185 <= pixel[0] <= 190  # Linear-light 50% white over black, not sRGB 128.
    assert pixel[3] == 255
    transparent = np.zeros((1, 1, 4), dtype=np.float32)
    source_over(transparent, white, 0, 0)
    assert 126 <= linear_to_image(transparent).getpixel((0, 0))[3] <= 129


def test_z_order_is_stable_and_layers_clip_to_canvas(tmp_path):
    base = Image.new("RGBA", (3, 3), (0, 0, 0, 0))
    red = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
    blue = Image.new("RGBA", (2, 2), (0, 0, 255, 255))
    base.save(tmp_path / "base.png")
    red.save(tmp_path / "red.png")
    blue.save(tmp_path / "blue.png")
    manifest = {"canvas": [3, 3], "background": "base.png", "layers": [
        {"name": "red", "path": "red.png", "x": 2, "y": 2, "z_index": 3},
        {"name": "blue", "path": "blue.png", "x": 2, "y": 2, "z_index": 3},
    ]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = render(path, tmp_path / "stages")
    assert result.getpixel((2, 2)) == (0, 0, 255, 255)
    assert result.getpixel((0, 0))[3] == 0
    assert sorted(p.name for p in (tmp_path / "stages").iterdir()) == ["01_red.png", "02_blue.png"]


def test_optional_mask_hides_existing_alpha(tmp_path):
    Image.new("RGBA", (1, 1), (255, 20, 10, 128)).save(tmp_path / "layer.png")
    Image.new("L", (1, 1), 255).save(tmp_path / "mask.png")
    result = prepare_layer(tmp_path / "manifest.json", {"path": "layer.png", "mask": "mask.png"})
    assert result.getpixel((0, 0))[3] == 0


def test_background_override_locks_opaque_foreground_pixels(tmp_path):
    Image.new("RGB", (3, 3), (20, 20, 20)).save(tmp_path / "old.png")
    Image.new("RGB", (3, 3), (200, 200, 200)).save(tmp_path / "generated.png")
    Image.new("RGBA", (1, 1), (240, 30, 10, 255)).save(tmp_path / "subject.png")
    manifest = {"canvas": [3, 3], "background": "old.png", "layers": [
        {"path": "subject.png", "x": 1, "y": 1}
    ]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original = render(path)
    locked = render(path, background_override=tmp_path / "generated.png")
    mask = render(path, skip_background=True).getchannel("A")
    assert mask.getpixel((1, 1)) == 255
    assert mask.getpixel((0, 0)) == 0
    assert locked.getpixel((1, 1)) == original.getpixel((1, 1))
    assert locked.getpixel((0, 0)) == (200, 200, 200, 255)
