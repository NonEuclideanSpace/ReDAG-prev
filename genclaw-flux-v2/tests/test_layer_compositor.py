"""Offline checks for editable IR, visual composition and actual reference routing."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules.compositor import ImageLayer, SceneComposition
from modules.compositor.generation import refine_layout
from examples.multi_reference_compositor_demo import build_scene


def png(tmp_path, name, color=(255, 0, 0, 255), size=(2, 2)):
    path = tmp_path / name
    Image.new("RGBA", size, color).save(path)
    return str(path)


def test_linear_light_alpha_and_stable_order(tmp_path):
    white = png(tmp_path, "white.png", (255, 255, 255, 128), (1, 1))
    scene = SceneComposition((1, 1), (0, 0, 0, 255))
    scene.add_layer(ImageLayer("white", white))
    assert 185 <= scene.render().getpixel((0, 0))[0] <= 190
    assert scene.render_foreground().getpixel((0, 0))[3] == 128
    red = png(tmp_path, "red.png")
    blue = png(tmp_path, "blue.png", (0, 0, 255, 255))
    scene = SceneComposition((3, 3), (0, 0, 0, 0))
    scene.add_layer(ImageLayer("red", red, (2, 2), z_order=3))
    scene.add_layer(ImageLayer("blue", blue, (2, 2), z_order=3))
    assert scene.render().getpixel((2, 2)) == (0, 0, 255, 255)
    assert scene.render().getpixel((0, 0))[3] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        scene.add_layer(ImageLayer("blue", blue))


def test_minimum_bbox_negative_position_scale_rotation(tmp_path):
    source = Image.new("RGBA", (8, 8))
    source.paste((255, 0, 0, 255), (2, 3, 6, 5))
    path = tmp_path / "padded.png"
    source.save(path)
    layer = ImageLayer("padded", str(path), (-1, 1), scale=2, rotation=90)
    assert layer.image.size == (4, 2)
    assert layer.crop_bbox == (2, 3, 6, 5)
    assert layer.scaled_size == (8, 4)
    scene = SceneComposition((10, 10), (0, 0, 0, 0)).add_layer(layer)
    image = scene.render(tmp_path / "scene_layout.png")
    record = json.loads((tmp_path / "scene_ir.json").read_text())["layers"][0]
    assert record["raster_bbox"] == [1, -1, 5, 7]
    assert image.getpixel((1, 0)) == (255, 0, 0, 255)
    assert image.getpixel((0, 0))[3] == 0


def test_ir_roundtrip_metadata_and_tamper_detection(tmp_path):
    red = png(tmp_path, "red.png")
    scene = SceneComposition((8, 8), metadata={"branch": "characters"},
                             relations=[{"subject": "red", "predicate": "near", "object": "car"}])
    scene.add_layer(ImageLayer("red", red, (1, 2), scale=1.5, rotation=20,
                               metadata={"reference_node": "partial_A"}, depends_on=["draft_A"]))
    ir = tmp_path / "run" / "scene_ir.json"
    scene.save_ir(ir)
    replay = SceneComposition.from_ir(ir)
    assert np.array_equal(np.asarray(scene.render()), np.asarray(replay.render()))
    assert replay.layers[0].depends_on == ["draft_A"]
    assert replay.relations == scene.relations
    record = json.loads(ir.read_text())
    assert record["layers"][0]["source"] == "../red.png"
    png(tmp_path, "red.png", (0, 0, 255, 255))
    with pytest.raises(ValueError, match="hash mismatch"):
        SceneComposition.from_ir(ir)
    with pytest.raises(ValueError, match="Source changed"):
        scene.save_ir(ir)


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": float("nan")},
                                    {"rotation": float("inf")}, {"opacity": 2}])
def test_invalid_transforms(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ImageLayer("bad", png(tmp_path, "bad.png"), **kwargs)


def test_transparent_assets_and_auto_layout(tmp_path):
    source = Image.new("RGBA", (20, 30))
    source.paste((200, 20, 10, 255), (2, 2, 18, 28))
    source.save(tmp_path / "a.png")
    source.save(tmp_path / "b.png")
    scene = build_scene(tmp_path, canvas=(320, 192))
    assert len(scene.layers) == 2
    assert scene.layers[0].x < scene.layers[1].x
    assert scene.layers[0].image.size == (16, 26)
    png(tmp_path, "opaque.png")
    with pytest.raises(ValueError, match="transparent pixels"):
        build_scene(tmp_path)


def test_adapter_only_sends_layout_and_restores_options(tmp_path, monkeypatch):
    import cc_genclaw.config
    import cc_genclaw.tools._image_gen.qwen_remote as provider
    monkeypatch.setattr(cc_genclaw.config, "load_genclaw_config", lambda: None)
    layout = png(tmp_path, "layout.png", (128, 128, 128, 255), (320, 192))
    calls = []
    def request(prompt, refs, model, cfg, output):
        calls.append((prompt, refs, model, cfg, output))
        assert os.environ["QWEN_IMAGE_ALLOW_MOCK"] == "0"
        assert os.environ["QWEN_IMAGE_WIDTH"] == "320"
        return output
    monkeypatch.setattr(provider, "request_image", request)
    monkeypatch.setenv("QWEN_IMAGE_WIDTH", "777")
    refine_layout(layout, "exact prompt\n", tmp_path / "final.png", "http://localhost:8190")
    assert calls[0][:3] == ("exact prompt\n", [layout], "qwen-image")
    assert os.environ["QWEN_IMAGE_WIDTH"] == "777"
    with pytest.raises(ValueError, match="multiples of 32"):
        refine_layout(png(tmp_path, "bad_size.png", size=(65, 65)), "test", tmp_path / "final.png")


def test_demo_layout_only_and_output_protection(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    source = Image.new("RGBA", (20, 30))
    source.paste((200, 20, 10, 255), (2, 2, 18, 28))
    source.save(refs / "person.png")
    output = tmp_path / "run"
    cmd = [sys.executable, "examples/multi_reference_compositor_demo.py", "--reference_dir", str(refs),
           "--prompt", "Exact prompt.", "--layout-only", "--output-dir", str(output)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (output / "prompt.txt").read_text() == "Exact prompt."
    assert Image.open(output / "scene_layout.png").size == (1024, 576)
    assert not (output / "final.png").exists()
    assert json.loads((output / "artifacts.json").read_text())["generation"]["status"] == "not_requested"
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 1
