"""Normal source-over in linear RGB, stable z-order and canvas clipping.

Inspired by ComfyUI's compositor design; uses only NumPy and Pillow, not
ComfyUI imports, nodes, workflows, or custom operators.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_linear(image):
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255
    rgb = rgba[..., :3]
    rgba[..., :3] = np.where(rgb <= 0.04045, rgb / 12.92,
                            ((rgb + 0.055) / 1.055) ** 2.4)
    return rgba


def to_image(rgba):
    rgb = np.clip(rgba[..., :3], 0, 1)
    rgb = np.where(rgb <= 0.0031308, rgb * 12.92,
                   1.055 * rgb ** (1 / 2.4) - 0.055)
    encoded = np.concatenate((rgb, np.clip(rgba[..., 3:4], 0, 1)), axis=-1)
    return Image.fromarray(np.rint(encoded * 255).astype(np.uint8))


def source_over(canvas, source, x, y, opacity=1.0):
    """RGBA arrays are straight-alpha, linear-light floats. Modify canvas."""
    h, w = canvas.shape[:2]
    sh, sw = source.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + sw), min(h, y + sh)
    if x0 >= x1 or y0 >= y1:
        return
    src = source[y0 - y:y1 - y, x0 - x:x1 - x]
    dst = canvas[y0:y1, x0:x1]
    sa, da = src[..., 3:4] * opacity, dst[..., 3:4]
    alpha = sa + da * (1 - sa)
    premul = src[..., :3] * sa + dst[..., :3] * da * (1 - sa)
    dst[..., :3] = np.divide(premul, alpha, out=np.zeros_like(premul), where=alpha > 0)
    dst[..., 3:4] = alpha


def prepare_layer(layer):
    image = layer.image.resize(layer.scaled_size, Image.Resampling.LANCZOS)
    w, h = image.size
    if layer.rotation % 360:
        image = image.rotate(-layer.rotation, resample=Image.Resampling.BICUBIC, expand=True)
    # Preserve the pre-rotation center; raster placement rounds down at half pixels.
    x = layer.x + (w - image.width) // 2
    y = layer.y + (h - image.height) // 2
    return image, (x, y)


def render_scene(scene, foreground_only=False):
    color = (0, 0, 0, 0) if foreground_only else scene.background_color
    canvas = to_linear(Image.new("RGBA", scene.canvas_size, color))
    if scene.background_path and not foreground_only:
        with Image.open(scene.background_path) as background:
            if background.size != scene.canvas_size:
                raise ValueError("Background dimensions must match canvas; no silent stretching")
            source_over(canvas, to_linear(background), 0, 0)
    for layer in scene.ordered_layers():
        image, (x, y) = prepare_layer(layer)
        source_over(canvas, to_linear(image), x, y, layer.opacity)
    return to_image(canvas)
