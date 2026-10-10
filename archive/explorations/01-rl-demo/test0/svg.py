from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 30, 30),
    "crimson": (220, 20, 60),
    "orange": (255, 140, 0),
    "yellow": (250, 220, 40),
    "gold": (255, 200, 0),
    "green": (40, 170, 60),
    "lime": (120, 220, 60),
    "darkgreen": (0, 100, 0),
    "blue": (40, 90, 220),
    "navy": (20, 30, 110),
    "skyblue": (135, 206, 235),
    "lightblue": (170, 210, 240),
    "cyan": (0, 200, 220),
    "teal": (0, 128, 128),
    "purple": (140, 60, 180),
    "violet": (160, 100, 220),
    "pink": (255, 150, 190),
    "magenta": (230, 40, 200),
    "brown": (140, 80, 40),
    "tan": (210, 180, 140),
    "beige": (235, 220, 190),
    "black": (20, 20, 20),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (190, 190, 195),
    "white": (250, 250, 250),
    "wooden": (160, 110, 60),
}

_ALIASES = {"grey": "gray", "wood": "wooden", "golden": "gold", "sky-blue": "skyblue", "light-blue": "lightblue", "dark-green": "darkgreen"}

SHAPE_TAGS = ("rect", "circle", "ellipse", "line", "polyline", "polygon", "path")
CONTAINER_TAGS = ("g", "svg")
ALLOWED_TAGS = SHAPE_TAGS + CONTAINER_TAGS + ("title", "desc")
FORBIDDEN_TAGS = ("text", "tspan", "textPath", "image", "linearGradient", "radialGradient", "pattern", "filter", "mask", "clipPath", "animate", "animateTransform", "animateMotion", "set", "script", "style", "foreignObject", "use", "symbol", "marker", "defs", "a", "switch", "video", "audio", "iframe")

Matrix = Tuple[float, float, float, float, float, float]
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def normalize_color_name(name: str) -> str:
    key = str(name or "").strip().lower().replace("_", "-")
    key = _ALIASES.get(key, key)
    key = key.replace("-", "")
    return key


def parse_color(value: Any) -> Optional[Tuple[int, int, int]]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s or s in {"none", "transparent", "currentcolor", "inherit"}:
        return None
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            except ValueError:
                return None
        return None
    m = re.match(r"rgba?\(\s*([0-9.]+%?)\s*,\s*([0-9.]+%?)\s*,\s*([0-9.]+%?)", s)
    if m:
        vals = []
        for part in m.groups():
            if part.endswith("%"):
                vals.append(int(round(float(part[:-1]) * 2.55)))
            else:
                vals.append(int(round(float(part))))
        return tuple(max(0, min(255, v)) for v in vals)
    key = normalize_color_name(s)
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    return _CSS_EXTRA.get(key)


_CSS_EXTRA: Dict[str, Tuple[int, int, int]] = {
    "darkred": (139, 0, 0), "firebrick": (178, 34, 34), "tomato": (255, 99, 71), "coral": (255, 127, 80),
    "darkorange": (255, 140, 0), "peru": (205, 133, 63), "chocolate": (210, 105, 30), "sienna": (160, 82, 45),
    "saddlebrown": (139, 69, 19), "burlywood": (222, 184, 135), "wheat": (245, 222, 179), "khaki": (240, 230, 140),
    "olive": (128, 128, 0), "forestgreen": (34, 139, 34), "seagreen": (46, 139, 87), "limegreen": (50, 205, 50),
    "lightgreen": (144, 238, 144), "yellowgreen": (154, 205, 50), "springgreen": (0, 255, 127), "aqua": (0, 255, 255),
    "turquoise": (64, 224, 208), "steelblue": (70, 130, 180), "royalblue": (65, 105, 225), "dodgerblue": (30, 144, 255),
    "deepskyblue": (0, 191, 255), "cornflowerblue": (100, 149, 237), "midnightblue": (25, 25, 112), "indigo": (75, 0, 130),
    "darkviolet": (148, 0, 211), "orchid": (218, 112, 214), "plum": (221, 160, 221), "hotpink": (255, 105, 180),
    "deeppink": (255, 20, 147), "salmon": (250, 128, 114), "lightgray": (211, 211, 211), "lightgrey": (211, 211, 211),
    "darkgray": (169, 169, 169), "darkgrey": (169, 169, 169), "dimgray": (105, 105, 105), "whitesmoke": (245, 245, 245),
    "ivory": (255, 255, 240), "snow": (255, 250, 250), "lavender": (230, 230, 250), "maroon": (128, 0, 0),
    "sandybrown": (244, 164, 96), "goldenrod": (218, 165, 32), "lightyellow": (255, 255, 224), "mintcream": (245, 255, 250),
}


def color_distance(a: Sequence[int], b: Sequence[int]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def nearest_color_name(rgb: Sequence[int]) -> str:
    best, best_d = "gray", float("inf")
    for name, ref in NAMED_COLORS.items():
        if name in {"grey", "wooden"}:
            continue
        d = color_distance(rgb, ref)
        if d < best_d:
            best, best_d = name, d
    return best


def color_matches(rgb: Sequence[int], name: str, tolerance: float = 95.0) -> bool:
    ref = parse_color(name)
    if ref is None:
        return False
    if color_distance(rgb, ref) <= tolerance:
        return True
    return nearest_color_name(rgb) == normalize_color_name(name)


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    a0, a1, a2, a3, a4, a5 = a
    b0, b1, b2, b3, b4, b5 = b
    return (
        a0 * b0 + a2 * b1,
        a1 * b0 + a3 * b1,
        a0 * b2 + a2 * b3,
        a1 * b2 + a3 * b3,
        a0 * b4 + a2 * b5 + a4,
        a1 * b4 + a3 * b5 + a5,
    )


def matrix_apply(m: Matrix, x: float, y: float) -> Tuple[float, float]:
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def parse_transform(value: Optional[str]) -> Matrix:
    m: Matrix = IDENTITY
    if not value:
        return m
    for name, args in re.findall(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", value):
        nums = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", args)]
        if name == "translate":
            tx = nums[0] if nums else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            t: Matrix = (1, 0, 0, 1, tx, ty)
        elif name == "scale":
            sx = nums[0] if nums else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            t = (sx, 0, 0, sy, 0, 0)
        elif name == "rotate":
            ang = math.radians(nums[0] if nums else 0.0)
            c, s = math.cos(ang), math.sin(ang)
            t = (c, s, -s, c, 0, 0)
            if len(nums) >= 3:
                cx, cy = nums[1], nums[2]
                t = matrix_multiply(matrix_multiply((1, 0, 0, 1, cx, cy), t), (1, 0, 0, 1, -cx, -cy))
        elif name == "skewX":
            t = (1, 0, math.tan(math.radians(nums[0] if nums else 0.0)), 1, 0, 0)
        elif name == "skewY":
            t = (1, math.tan(math.radians(nums[0] if nums else 0.0)), 0, 1, 0, 0)
        else:
            if len(nums) < 6:
                continue
            t = tuple(nums[:6])
        m = matrix_multiply(m, t)
    return m


def local_tag(elem: ET.Element) -> str:
    tag = elem.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return str(tag)


def strip_namespaces(root: ET.Element) -> ET.Element:
    for elem in root.iter():
        if isinstance(elem.tag, str) and "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
        for key in list(elem.attrib.keys()):
            if "}" in key:
                new_key = key.split("}", 1)[1]
                elem.attrib[new_key] = elem.attrib.pop(key)
            elif key.startswith("xmlns"):
                elem.attrib.pop(key)
    return root


def extract_code(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^```(?:svg|xml|html)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    fence = re.search(r"```(?:svg|xml|html)?\s*([\s\S]*?)```", s)
    if fence and "<" in fence.group(1):
        s = fence.group(1).strip()
    start = None
    for pat in (r"<svg[\s>]", r"<g[\s>]"):
        m = re.search(pat, s, re.IGNORECASE)
        if m and (start is None or m.start() < start):
            start = m.start()
    if start is not None:
        s = s[start:]
    end = None
    for pat in (r"</svg\s*>", r"</g\s*>"):
        matches = list(re.finditer(pat, s, re.IGNORECASE))
        if matches:
            pos = matches[-1].end()
            if end is None or pos > end:
                end = pos
    if end is not None:
        s = s[:end]
    s = re.sub(r"<\?xml[^>]*\?>", "", s)
    s = re.sub(r"<!DOCTYPE[^>]*>", "", s, flags=re.IGNORECASE)
    return s.strip()


def parse_svg_text(text: str) -> ET.Element:
    code = extract_code(text)
    if not code:
        raise ValueError("empty svg code")
    try:
        root = ET.fromstring(code)
    except ET.ParseError:
        wrapped = f"<svg xmlns='http://www.w3.org/2000/svg'>{code}</svg>"
        root = ET.fromstring(wrapped)
    return strip_namespaces(root)


def parse_fragment(text: str, node_id: Optional[str] = None) -> ET.Element:
    root = parse_svg_text(text)
    if local_tag(root) == "svg":
        groups = [c for c in list(root) if local_tag(c) == "g"]
        if node_id is not None:
            named = [g for g in groups if g.get("data-node") == node_id]
            if named:
                root = named[0]
            elif len(groups) == 1:
                root = groups[0]
            else:
                g = ET.Element("g")
                for child in list(root):
                    g.append(child)
                if root.get("viewBox"):
                    parts = root.get("viewBox").split()
                    if len(parts) == 4:
                        g.set("data-bbox", " ".join(parts))
                root = g
        elif len(groups) == 1:
            root = groups[0]
        else:
            g = ET.Element("g")
            for child in list(root):
                g.append(child)
            root = g
    if local_tag(root) != "g":
        g = ET.Element("g")
        g.append(root)
        root = g
    if node_id is not None and root.get("data-node") != node_id:
        inner = root.find(f".//g[@data-node='{node_id}']")
        if inner is not None and inner is not root:
            root = inner
        else:
            root.set("data-node", node_id)
    return root


def sanitize(root: ET.Element) -> Tuple[ET.Element, List[str]]:
    removed: List[str] = []
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for elem in list(root.iter()):
        tag = local_tag(elem)
        if tag in FORBIDDEN_TAGS:
            parent = parent_map.get(elem)
            if parent is not None:
                parent.remove(elem)
                removed.append(tag)
    for elem in root.iter():
        for attr in list(elem.attrib.keys()):
            if attr.startswith("on") or attr in {"href", "xlink:href", "filter", "mask", "clip-path"}:
                elem.attrib.pop(attr)
                removed.append(attr)
            elif attr in {"fill", "stroke"} and "url(" in elem.attrib[attr]:
                elem.attrib[attr] = "#888888"
                removed.append("url-paint")
    return root, removed


def serialize(elem: ET.Element) -> str:
    return ET.tostring(elem, encoding="unicode")


def declared_bbox(root: ET.Element) -> Optional[Tuple[float, float, float, float]]:
    raw = root.get("data-bbox") or root.get("viewBox")
    if not raw:
        return None
    parts = re.split(r"[\s,]+", raw.strip())
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _num(elem: ET.Element, attr: str, default: float = 0.0) -> float:
    raw = elem.get(attr)
    if raw is None:
        return default
    m = re.match(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", raw)
    return float(m.group(1)) if m else default


def _bezier(points: List[Tuple[float, float]], steps: int = 10) -> List[Tuple[float, float]]:
    out = []
    n = len(points) - 1
    for i in range(1, steps + 1):
        t = i / steps
        x = y = 0.0
        for k, (px, py) in enumerate(points):
            coef = math.comb(n, k) * (1 - t) ** (n - k) * t ** k
            x += coef * px
            y += coef * py
        out.append((x, y))
    return out


def _arc(x1, y1, rx, ry, phi_deg, large, sweep, x2, y2, steps: int = 16) -> List[Tuple[float, float]]:
    if rx == 0 or ry == 0:
        return [(x2, y2)]
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_p * dx + sin_p * dy
    y1p = -sin_p * dx + cos_p * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1p ** 2) / (rx ** 2) + (y1p ** 2) / (ry ** 2)
    if lam > 1:
        rx *= math.sqrt(lam)
        ry *= math.sqrt(lam)
    num = rx ** 2 * ry ** 2 - rx ** 2 * y1p ** 2 - ry ** 2 * x1p ** 2
    den = rx ** 2 * y1p ** 2 + ry ** 2 * x1p ** 2
    coef = 0.0 if den == 0 else math.sqrt(max(0.0, num / den))
    if large == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (x1 + x2) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (y1 + y2) / 2.0

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        if length == 0:
            return 0.0
        a = math.acos(max(-1.0, min(1.0, dot / length)))
        return -a if ux * vy - uy * vx < 0 else a

    theta1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi
    out = []
    for i in range(1, steps + 1):
        t = theta1 + dtheta * i / steps
        px = cx + rx * math.cos(t) * cos_p - ry * math.sin(t) * sin_p
        py = cy + rx * math.cos(t) * sin_p + ry * math.sin(t) * cos_p
        out.append((px, py))
    return out


_PATH_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def path_to_polylines(d: str) -> List[Tuple[List[Tuple[float, float]], bool]]:
    tokens = _PATH_TOKEN.findall(d or "")
    polys: List[Tuple[List[Tuple[float, float]], bool]] = []
    current: List[Tuple[float, float]] = []
    cmd = ""
    i = 0
    x = y = 0.0
    sx = sy = 0.0
    last_ctrl: Optional[Tuple[float, float]] = None

    def flush(closed: bool):
        nonlocal current
        if len(current) >= 2:
            polys.append((current, closed))
        current = []

    def read(n: int) -> List[float]:
        nonlocal i
        vals = []
        for _ in range(n):
            if i >= len(tokens):
                raise ValueError("path data truncated")
            vals.append(float(tokens[i]))
            i += 1
        return vals

    while i < len(tokens):
        tok = tokens[i]
        if re.match(r"[A-Za-z]", tok):
            cmd = tok
            i += 1
            if cmd in "Zz":
                if current:
                    current.append((sx, sy))
                flush(True)
                x, y = sx, sy
                continue
        elif not cmd:
            raise ValueError("path data without command")
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            px, py = read(2)
            if rel:
                px, py = x + px, y + py
            flush(False)
            x, y = px, py
            sx, sy = x, y
            current = [(x, y)]
            cmd = "l" if rel else "L"
            last_ctrl = None
        elif c == "L":
            px, py = read(2)
            if rel:
                px, py = x + px, y + py
            x, y = px, py
            current.append((x, y))
            last_ctrl = None
        elif c == "H":
            (px,) = read(1)
            x = x + px if rel else px
            current.append((x, y))
            last_ctrl = None
        elif c == "V":
            (py,) = read(1)
            y = y + py if rel else py
            current.append((x, y))
            last_ctrl = None
        elif c == "C":
            x1, y1, x2, y2, px, py = read(6)
            if rel:
                x1, y1, x2, y2, px, py = x + x1, y + y1, x + x2, y + y2, x + px, y + py
            current.extend(_bezier([(x, y), (x1, y1), (x2, y2), (px, py)]))
            last_ctrl = (x2, y2)
            x, y = px, py
        elif c == "S":
            x2, y2, px, py = read(4)
            if rel:
                x2, y2, px, py = x + x2, y + y2, x + px, y + py
            x1, y1 = (2 * x - last_ctrl[0], 2 * y - last_ctrl[1]) if last_ctrl else (x, y)
            current.extend(_bezier([(x, y), (x1, y1), (x2, y2), (px, py)]))
            last_ctrl = (x2, y2)
            x, y = px, py
        elif c == "Q":
            x1, y1, px, py = read(4)
            if rel:
                x1, y1, px, py = x + x1, y + y1, x + px, y + py
            current.extend(_bezier([(x, y), (x1, y1), (px, py)]))
            last_ctrl = (x1, y1)
            x, y = px, py
        elif c == "T":
            px, py = read(2)
            if rel:
                px, py = x + px, y + py
            x1, y1 = (2 * x - last_ctrl[0], 2 * y - last_ctrl[1]) if last_ctrl else (x, y)
            current.extend(_bezier([(x, y), (x1, y1), (px, py)]))
            last_ctrl = (x1, y1)
            x, y = px, py
        elif c == "A":
            rx, ry, rot, large, sweep, px, py = read(7)
            if rel:
                px, py = x + px, y + py
            current.extend(_arc(x, y, rx, ry, rot, bool(int(large)), bool(int(sweep)), px, py))
            x, y = px, py
            last_ctrl = None
        else:
            raise ValueError(f"unsupported path command {cmd}")
    flush(False)
    return polys


def shape_polylines(elem: ET.Element) -> List[Tuple[List[Tuple[float, float]], bool]]:
    tag = local_tag(elem)
    if tag == "rect":
        x, y = _num(elem, "x"), _num(elem, "y")
        w, h = _num(elem, "width"), _num(elem, "height")
        if w <= 0 or h <= 0:
            return []
        rx = min(_num(elem, "rx", _num(elem, "ry", 0.0)), w / 2)
        ry = min(_num(elem, "ry", rx), h / 2)
        if rx > 0 and ry > 0:
            pts: List[Tuple[float, float]] = []
            corners = [(x + w - rx, y + ry, -90, 0), (x + w - rx, y + h - ry, 0, 90), (x + rx, y + h - ry, 90, 180), (x + rx, y + ry, 180, 270)]
            for cx, cy, a0, a1 in corners:
                for k in range(0, 7):
                    a = math.radians(a0 + (a1 - a0) * k / 6)
                    pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
            return [(pts, True)]
        return [([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], True)]
    if tag == "circle":
        cx, cy, r = _num(elem, "cx"), _num(elem, "cy"), _num(elem, "r")
        if r <= 0:
            return []
        return [([(cx + r * math.cos(2 * math.pi * k / 48), cy + r * math.sin(2 * math.pi * k / 48)) for k in range(48)], True)]
    if tag == "ellipse":
        cx, cy, rx, ry = _num(elem, "cx"), _num(elem, "cy"), _num(elem, "rx"), _num(elem, "ry")
        if rx <= 0 or ry <= 0:
            return []
        return [([(cx + rx * math.cos(2 * math.pi * k / 48), cy + ry * math.sin(2 * math.pi * k / 48)) for k in range(48)], True)]
    if tag == "line":
        return [([(_num(elem, "x1"), _num(elem, "y1")), (_num(elem, "x2"), _num(elem, "y2"))], False)]
    if tag in ("polyline", "polygon"):
        nums = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", elem.get("points", ""))]
        pts = list(zip(nums[0::2], nums[1::2]))
        if len(pts) < 2:
            return []
        return [(pts, tag == "polygon")]
    if tag == "path":
        try:
            return path_to_polylines(elem.get("d", ""))
        except ValueError:
            return []
    return []


def walk_shapes(root: ET.Element, matrix: Matrix = IDENTITY) -> Iterator[Tuple[ET.Element, Matrix]]:
    m = matrix_multiply(matrix, parse_transform(root.get("transform")))
    tag = local_tag(root)
    if tag in SHAPE_TAGS:
        yield root, m
        return
    if tag in ("defs", "clipPath", "mask", "symbol", "marker"):
        return
    for child in list(root):
        yield from walk_shapes(child, m)


def shape_bounds(elem: ET.Element, matrix: Matrix) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for pts, _closed in shape_polylines(elem):
        for px, py in pts:
            tx, ty = matrix_apply(matrix, px, py)
            xs.append(tx)
            ys.append(ty)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def bounds(root: ET.Element, matrix: Matrix = IDENTITY) -> Optional[Tuple[float, float, float, float]]:
    acc: Optional[List[float]] = None
    for elem, m in walk_shapes(root, matrix):
        b = shape_bounds(elem, m)
        if b is None:
            continue
        if acc is None:
            acc = list(b)
        else:
            acc[0] = min(acc[0], b[0])
            acc[1] = min(acc[1], b[1])
            acc[2] = max(acc[2], b[2])
            acc[3] = max(acc[3], b[3])
    return tuple(acc) if acc else None


def shape_count(root: ET.Element) -> int:
    return sum(1 for _ in walk_shapes(root))


def fills(root: ET.Element) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for elem, _m in walk_shapes(root):
        fill = elem.get("fill")
        if fill is None:
            style = elem.get("style") or ""
            m = re.search(r"fill\s*:\s*([^;]+)", style)
            fill = m.group(1) if m else None
        if fill is None:
            parent_fill = _inherited(root, elem, "fill")
            fill = parent_fill
        rgb = parse_color(fill if fill is not None else "black")
        if rgb is not None:
            out.append(rgb)
        stroke = elem.get("stroke")
        srgb = parse_color(stroke)
        if srgb is not None and _num(elem, "stroke-width", 1.0) >= 6:
            out.append(srgb)
    return out


def _inherited(root: ET.Element, target: ET.Element, attr: str) -> Optional[str]:
    parent_map = {child: parent for parent in root.iter() for child in parent}
    node = parent_map.get(target)
    while node is not None:
        if node.get(attr) is not None:
            return node.get(attr)
        node = parent_map.get(node)
    return root.get(attr)


def instances(root: ET.Element) -> List[ET.Element]:
    return [g for g in root.iter("g") if g.get("data-instance") is not None]


def coverage(root: ET.Element, bbox: Tuple[float, float, float, float]) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    total = 0.0
    for elem, m in walk_shapes(root):
        b = shape_bounds(elem, m)
        if b is None:
            continue
        cx1, cy1 = max(b[0], x), max(b[1], y)
        cx2, cy2 = min(b[2], x + w), min(b[3], y + h)
        if cx2 > cx1 and cy2 > cy1:
            total += (cx2 - cx1) * (cy2 - cy1)
    return min(1.0, total / (w * h))


def svg_document(width: float, height: float, body: str, background: str = "white") -> str:
    w, h = int(round(width)), int(round(height))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="{background}"/>'
        f"{body}</svg>"
    )


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_overlap_fraction(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = min(max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]), max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))
    return inter / area if area > 0 else 0.0
