from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from . import svg as S
from .cost import CostMeter


def _svg_size(svg_text: str, default: Tuple[int, int] = (1024, 1024)) -> Tuple[int, int, Tuple[float, float, float, float]]:
    tag = re.search(r"<svg\b[^>]*>", svg_text, re.IGNORECASE)
    head = tag.group(0) if tag else ""

    def num(attr: str) -> Optional[float]:
        m = re.search(rf'{attr}\s*=\s*["\']\s*([0-9]+(?:\.[0-9]+)?)', head, re.IGNORECASE)
        return float(m.group(1)) if m else None

    w, h = num("width"), num("height")
    vb = None
    m = re.search(r'viewBox\s*=\s*["\']\s*([-0-9.]+)[\s,]+([-0-9.]+)[\s,]+([0-9.]+)[\s,]+([0-9.]+)', head, re.IGNORECASE)
    if m:
        vb = tuple(float(v) for v in m.groups())
    if (w is None or h is None) and vb is not None:
        w = w or vb[2]
        h = h or vb[3]
    if w is None or h is None:
        w, h = float(default[0]), float(default[1])
    if vb is None:
        vb = (0.0, 0.0, float(w), float(h))
    return int(round(w)), int(round(h)), vb


def _fit(width: int, height: int, max_side: int) -> Tuple[int, int]:
    scale = min(1.0, max_side / float(max(width, height)))
    return max(16, int(round(width * scale))), max(16, int(round(height * scale)))


class PilRasterizer:
    def render(self, svg_text: str, out_path: str, max_side: int = 1024) -> str:
        root = S.parse_svg_text(svg_text)
        w, h, vb = _svg_size(svg_text)
        tw, th = _fit(w, h, max_side)
        img = Image.new("RGBA", (tw, th), (255, 255, 255, 255))
        sx = tw / vb[2] if vb[2] else 1.0
        sy = th / vb[3] if vb[3] else 1.0
        base: S.Matrix = (sx, 0.0, 0.0, sy, -vb[0] * sx, -vb[1] * sy)
        parent_map = {child: parent for parent in root.iter() for child in parent}
        for elem, m in S.walk_shapes(root, base):
            self._draw(img, elem, m, root, parent_map)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(out_path, format="PNG")
        return out_path

    def _style(self, elem, root, parent_map, attr: str, default: Optional[str]) -> Optional[str]:
        node = elem
        while node is not None:
            value = node.get(attr)
            if value is None:
                style = node.get("style") or ""
                m = re.search(rf"(?:^|;)\s*{attr}\s*:\s*([^;]+)", style)
                value = m.group(1).strip() if m else None
            if value is not None:
                return value
            node = parent_map.get(node)
        return default

    def _opacity(self, elem, root, parent_map, attr: str) -> float:
        total = 1.0
        node = elem
        while node is not None:
            for key in (attr, "opacity"):
                raw = node.get(key)
                if raw is not None:
                    try:
                        total *= max(0.0, min(1.0, float(raw)))
                    except ValueError:
                        pass
            node = parent_map.get(node)
        return total

    def _draw(self, img: Image.Image, elem, m: S.Matrix, root, parent_map) -> None:
        polys = S.shape_polylines(elem)
        if not polys:
            return
        fill_raw = self._style(elem, root, parent_map, "fill", "black")
        stroke_raw = self._style(elem, root, parent_map, "stroke", None)
        fill = S.parse_color(fill_raw)
        stroke = S.parse_color(stroke_raw)
        sw_raw = self._style(elem, root, parent_map, "stroke-width", "1")
        try:
            sw = float(re.match(r"[-+]?\d*\.?\d+", str(sw_raw)).group(0))
        except (AttributeError, ValueError):
            sw = 1.0
        scale = (abs(m[0]) + abs(m[3])) / 2.0
        sw_px = max(1, int(round(sw * scale))) if stroke else 0
        fill_alpha = int(round(255 * self._opacity(elem, root, parent_map, "fill-opacity")))
        stroke_alpha = int(round(255 * self._opacity(elem, root, parent_map, "stroke-opacity")))
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        for pts, closed in polys:
            dev = [S.matrix_apply(m, x, y) for x, y in pts]
            if len(dev) < 2:
                continue
            if closed and fill is not None and len(dev) >= 3:
                draw.polygon(dev, fill=(fill[0], fill[1], fill[2], fill_alpha))
            if stroke is not None and sw_px > 0:
                seq = dev + [dev[0]] if closed else dev
                draw.line(seq, fill=(stroke[0], stroke[1], stroke[2], stroke_alpha), width=sw_px, joint="curve")
        img.alpha_composite(layer)


class PlaywrightRasterizer:
    def __init__(self, timeout_ms: int = 30000):
        self.timeout_ms = timeout_ms
        self._pw = None
        self._browser = None
        self._lock = threading.Lock()

    def _ensure(self):
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            timeout=self.timeout_ms,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )

    def render(self, svg_text: str, out_path: str, max_side: int = 1024) -> str:
        w, h, _vb = _svg_size(svg_text)
        tw, th = _fit(w, h, max_side)
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'><style>"
            "html,body{margin:0;padding:0;background:white;overflow:hidden}"
            f"svg{{display:block;width:{tw}px;height:{th}px}}"
            "</style></head><body>" + svg_text + "</body></html>"
        )
        with self._lock:
            self._ensure()
            page = self._browser.new_page(viewport={"width": tw, "height": th})
            try:
                page.set_default_timeout(self.timeout_ms)
                page.set_content(html, wait_until="load")
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=out_path, full_page=False, clip={"x": 0, "y": 0, "width": tw, "height": th})
            finally:
                page.close()
        return out_path

    def close(self) -> None:
        with self._lock:
            try:
                if self._browser is not None:
                    self._browser.close()
                if self._pw is not None:
                    self._pw.stop()
            finally:
                self._browser = None
                self._pw = None


class Renderer:
    def __init__(self, backend: str = "auto", cost: Optional[CostMeter] = None, max_side: int = 1024):
        self.requested = str(backend or "auto").lower()
        self.cost = cost
        self.max_side = int(max_side)
        self._pil = PilRasterizer()
        self._pw: Optional[PlaywrightRasterizer] = None
        self.backend = self._select()
        self.calls = 0

    def _select(self) -> str:
        if self.requested == "pil":
            return "pil"
        try:
            import playwright.sync_api
        except ImportError:
            if self.requested == "playwright":
                raise
            return "pil"
        return "playwright"

    def render(self, svg_text: str, out_path: str, node: Optional[str] = None, phase: Optional[str] = None) -> str:
        t0 = time.time()
        used = self.backend
        if self.backend == "playwright":
            try:
                if self._pw is None:
                    self._pw = PlaywrightRasterizer()
                self._pw.render(svg_text, out_path, self.max_side)
            except Exception:
                if self.requested == "playwright":
                    raise
                self.backend = "pil"
                used = "pil"
                self._pil.render(svg_text, out_path, self.max_side)
        else:
            self._pil.render(svg_text, out_path, self.max_side)
        self.calls += 1
        if self.cost is not None:
            self.cost.record("render", node=node, seconds=time.time() - t0, phase=phase, label=used)
        return out_path

    def close(self) -> None:
        if self._pw is not None:
            self._pw.close()
            self._pw = None


def overlay_boxes(png_path: str, out_path: str, boxes: List[Dict[str, Any]], canvas: Tuple[int, int]) -> str:
    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    sx = img.width / float(canvas[0])
    sy = img.height / float(canvas[1])
    palette = [(220, 30, 30), (30, 90, 220), (30, 160, 60), (200, 120, 0), (140, 40, 180), (0, 150, 160)]
    for i, box in enumerate(boxes):
        b = box["bbox"]
        color = palette[i % len(palette)]
        rect = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        draw.rectangle(rect, outline=color, width=3)
        label = str(box.get("label") or box.get("node") or i)
        tx, ty = rect[0] + 4, max(0, rect[1] - 14)
        draw.rectangle([tx - 2, ty - 1, tx + 7 * len(label) + 2, ty + 12], fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=color)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path
