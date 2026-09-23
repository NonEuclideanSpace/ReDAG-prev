from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import svg as S

ENTITY_WORDS = {
    "person", "people", "man", "woman", "girl", "boy", "child", "kid", "baby", "farmer", "chef", "teacher", "student", "worker",
    "cat", "kitten", "dog", "puppy", "horse", "cow", "sheep", "pig", "goat", "rabbit", "bunny", "fox", "bear", "deer", "elephant",
    "giraffe", "lion", "tiger", "zebra", "monkey", "mouse", "squirrel", "wolf", "panda",
    "bird", "duck", "chicken", "hen", "rooster", "owl", "eagle", "parrot", "penguin", "swan", "goose",
    "fish", "shark", "dolphin", "whale", "turtle", "frog", "butterfly", "bee", "snake",
}
QUADRUPEDS = {"cat", "kitten", "dog", "puppy", "horse", "cow", "sheep", "pig", "goat", "rabbit", "bunny", "fox", "bear", "deer", "elephant", "giraffe", "lion", "tiger", "zebra", "monkey", "mouse", "squirrel", "wolf", "panda", "turtle", "frog"}
BIRDS = {"bird", "duck", "chicken", "hen", "rooster", "owl", "eagle", "parrot", "penguin", "swan", "goose", "butterfly", "bee"}
FISHES = {"fish", "shark", "dolphin", "whale"}
BACKGROUND_WORDS = {
    "sky", "grass", "ground", "field", "meadow", "lawn", "sea", "ocean", "lake", "river", "pond", "water", "beach", "sand", "desert",
    "road", "street", "path", "mountain", "mountains", "hill", "hills", "sun", "moon", "cloud", "clouds", "wall", "floor", "room",
    "forest", "snow", "night", "sunset", "park", "garden", "farm", "stars", "kitchen", "classroom", "carpet", "rug", "sky", "horizon",
}
DEFAULT_COLORS = {
    "person": "blue", "cat": "orange", "dog": "brown", "horse": "brown", "cow": "black", "sheep": "white", "pig": "pink", "goat": "gray",
    "rabbit": "white", "fox": "orange", "bear": "brown", "deer": "brown", "elephant": "gray", "giraffe": "yellow", "lion": "gold",
    "tiger": "orange", "zebra": "white", "monkey": "brown", "mouse": "gray", "squirrel": "brown", "wolf": "gray", "panda": "white",
    "bird": "blue", "duck": "yellow", "chicken": "white", "owl": "brown", "eagle": "brown", "parrot": "green", "penguin": "black",
    "fish": "orange", "tree": "green", "house": "beige", "car": "red", "truck": "blue", "bus": "yellow", "bicycle": "blue",
    "apple": "red", "orange": "orange", "banana": "yellow", "ball": "red", "balloon": "red", "cup": "white", "mug": "white",
    "bottle": "green", "vase": "blue", "table": "wooden", "desk": "wooden", "chair": "wooden", "bench": "wooden", "umbrella": "red",
    "sun": "yellow", "moon": "silver", "cloud": "white", "flower": "pink", "box": "brown", "book": "blue", "gift": "red", "boat": "white",
    "star": "gold", "lamp": "yellow", "hat": "black", "kite": "red", "cake": "pink", "basket": "tan", "bucket": "gray", "tent": "orange",
    "generic": "gray",
}
PLURALS = {"people": "person", "children": "child", "kids": "kid", "women": "woman", "men": "man", "mice": "mouse", "geese": "goose", "sheep": "sheep", "fish": "fish", "boxes": "box", "buses": "bus", "benches": "bench", "bunnies": "bunny", "puppies": "puppy", "kitties": "kitten", "wolves": "wolf", "leaves": "leaf", "foxes": "fox", "glasses": "glass", "dresses": "dress", "vases": "vase", "houses": "house", "horses": "horse"}
NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "single": 1, "pair": 2, "couple": 2, "dozen": 12}


def singular(word: str) -> str:
    w = word.lower().strip()
    if w in PLURALS:
        return PLURALS[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ches") or w.endswith("shes") or w.endswith("xes") or w.endswith("sses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def hexcolor(name: str) -> str:
    rgb = S.parse_color(name) or S.NAMED_COLORS["gray"]
    return "#%02x%02x%02x" % rgb


def darker(name: str, factor: float = 0.65) -> str:
    rgb = S.parse_color(name) or S.NAMED_COLORS["gray"]
    return "#%02x%02x%02x" % tuple(int(c * factor) for c in rgb)


def lighter(name: str, factor: float = 0.5) -> str:
    rgb = S.parse_color(name) or S.NAMED_COLORS["gray"]
    return "#%02x%02x%02x" % tuple(int(c + (255 - c) * factor) for c in rgb)


def _sec(color: str, secondary: Optional[str]) -> str:
    return hexcolor(secondary) if secondary else darker(color)


def head_x(facing: str, w: float, inset: float = 0.24) -> float:
    if facing == "left":
        return w * inset
    if facing == "right":
        return w * (1 - inset)
    return w / 2.0


def noun_category(noun: str) -> str:
    n = singular(noun)
    if n in {"person", "man", "woman", "girl", "boy", "child", "kid", "baby", "farmer", "chef", "teacher", "student", "worker"}:
        return "person"
    if n in QUADRUPEDS:
        return "quadruped"
    if n in BIRDS:
        return "bird"
    if n in FISHES:
        return "fish"
    if n in {"tree", "pine", "oak", "palm", "bush"}:
        return "tree"
    if n in {"house", "cottage", "barn", "hut", "cabin", "building", "tower", "castle"}:
        return "house"
    if n in {"car", "truck", "bus", "van", "taxi", "tractor", "cart", "wagon"}:
        return "car"
    if n in {"bicycle", "bike", "motorcycle", "scooter"}:
        return "bicycle"
    if n in {"apple", "orange", "ball", "balloon", "tomato", "peach", "plum", "cherry", "melon", "pumpkin", "egg", "lemon", "coin", "stone", "rock", "globe", "planet"}:
        return "round"
    if n in {"banana"}:
        return "banana"
    if n in {"cup", "mug", "bottle", "vase", "glass", "jar", "bucket", "can", "pot"}:
        return "cup"
    if n in {"table", "desk", "counter"}:
        return "table"
    if n in {"chair", "bench", "stool", "sofa", "couch", "seat"}:
        return "chair"
    if n in {"umbrella", "parasol"}:
        return "umbrella"
    if n in {"sun", "moon"}:
        return "disk"
    if n in {"cloud"}:
        return "cloud"
    if n in {"flower", "rose", "tulip", "daisy", "sunflower"}:
        return "flower"
    if n in {"box", "book", "gift", "present", "crate", "suitcase", "bag", "backpack", "laptop", "phone", "screen", "sign", "board", "pillow", "brick", "block"}:
        return "box"
    if n in {"boat", "ship", "sailboat", "canoe"}:
        return "boat"
    if n in {"star"}:
        return "star"
    if n in {"lamp", "candle", "torch", "streetlight"}:
        return "lamp"
    if n in {"hat", "cap", "helmet"}:
        return "hat"
    if n in {"kite", "flag", "banner"}:
        return "kite"
    if n in {"cake", "pie", "pizza", "bread", "sandwich", "burger"}:
        return "cake"
    if n in {"basket", "bowl", "plate", "dish"}:
        return "basket"
    if n in {"tent", "mountain", "pyramid", "roof"}:
        return "tent"
    return "generic"


def default_color(noun: str) -> str:
    n = singular(noun)
    if n in DEFAULT_COLORS:
        return DEFAULT_COLORS[n]
    cat = noun_category(noun)
    return {"person": "blue", "quadruped": "brown", "bird": "blue", "fish": "orange", "round": "red", "cup": "white", "table": "wooden", "chair": "wooden"}.get(cat, "gray")


def default_aspect(noun: str, pose: str) -> Tuple[int, int]:
    cat = noun_category(noun)
    if cat == "person":
        if pose in ("lying", "swimming"):
            return 400, 180
        if pose in ("sitting", "crouching", "kneeling"):
            return 260, 300
        return 200, 400
    if cat == "quadruped":
        if pose in ("sitting",):
            return 240, 300
        return 340, 240
    if cat in ("bird", "fish", "boat", "car", "bicycle", "cloud", "banana"):
        return 320, 200
    if cat in ("tree", "lamp", "bottle", "flower"):
        return 220, 380
    if cat in ("house", "tent", "table", "chair", "cake", "basket"):
        return 320, 260
    if cat in ("umbrella", "kite", "hat"):
        return 300, 300
    return 260, 260


def _rect(x, y, w, h, fill, extra: str = "", rx: float = 0) -> str:
    rx_attr = f' rx="{rx:.1f}"' if rx else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}"{rx_attr}{extra}/>'


def _circle(cx, cy, r, fill, extra: str = "") -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{extra}/>'


def _ellipse(cx, cy, rx, ry, fill, extra: str = "") -> str:
    return f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="{fill}"{extra}/>'


def _poly(points: Sequence[Tuple[float, float]], fill: str, extra: str = "") -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}"{extra}/>'


def draw_person(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    skin = "#f1c27d"
    dark = _sec(color, secondary)
    out: List[str] = []
    if pose in ("lying", "swimming"):
        hx = head_x(facing, w, 0.12)
        body_x = hx + w * 0.1 if facing != "right" else hx - w * 0.6
        out.append(_rect(body_x, h * 0.35, w * 0.5, h * 0.3, hexcolor(color), rx=h * 0.08))
        out.append(_rect(body_x + (w * 0.5 if facing != "right" else -w * 0.28), h * 0.4, w * 0.28, h * 0.2, dark))
        out.append(_circle(hx, h * 0.5, h * 0.22, skin))
        out.append(_circle(hx + (h * 0.08 if facing == "right" else -h * 0.08 if facing == "left" else 0), h * 0.45, h * 0.05, "#222222", ' data-anchor="face"'))
        return out
    head_r = w * 0.19
    hx = w / 2.0
    hy = h * 0.14
    if pose in ("sitting", "crouching", "kneeling"):
        out.append(_rect(w * 0.3, h * 0.3, w * 0.4, h * 0.32, hexcolor(color), rx=w * 0.05))
        leg_dir = -1 if facing == "left" else 1
        lx = w * 0.5 if leg_dir > 0 else w * 0.1
        out.append(_rect(lx, h * 0.58, w * 0.4, h * 0.14, dark, rx=w * 0.03))
        out.append(_rect(lx + (w * 0.3 if leg_dir > 0 else 0), h * 0.7, w * 0.1, h * 0.24, dark))
        out.append(_rect(w * 0.18, h * 0.34, w * 0.1, h * 0.24, hexcolor(color)))
        out.append(_rect(w * 0.72, h * 0.34, w * 0.1, h * 0.24, hexcolor(color)))
    else:
        lean = w * 0.06 if pose == "running" else 0
        out.append(_poly([(w * 0.3 + lean, h * 0.3), (w * 0.7 + lean, h * 0.3), (w * 0.72, h * 0.62), (w * 0.28, h * 0.62)], hexcolor(color)))
        if pose in ("running", "walking"):
            out.append(_poly([(w * 0.3, h * 0.6), (w * 0.44, h * 0.6), (w * 0.2, h * 0.94), (w * 0.08, h * 0.9)], dark))
            out.append(_poly([(w * 0.56, h * 0.6), (w * 0.7, h * 0.6), (w * 0.9, h * 0.92), (w * 0.78, h * 0.96)], dark))
        elif pose == "jumping":
            out.append(_poly([(w * 0.3, h * 0.6), (w * 0.44, h * 0.6), (w * 0.34, h * 0.82), (w * 0.2, h * 0.78)], dark))
            out.append(_poly([(w * 0.56, h * 0.6), (w * 0.7, h * 0.6), (w * 0.8, h * 0.78), (w * 0.66, h * 0.82)], dark))
        else:
            out.append(_rect(w * 0.32, h * 0.6, w * 0.14, h * 0.36, dark))
            out.append(_rect(w * 0.54, h * 0.6, w * 0.14, h * 0.36, dark))
        if pose in ("holding", "reaching"):
            out.append(_poly([(w * 0.66, h * 0.32), (w * 0.76, h * 0.3), (w * 0.94, h * 0.08), (w * 0.99, h * 0.14)], hexcolor(color)))
            out.append(_rect(w * 0.12, h * 0.32, w * 0.12, h * 0.3, hexcolor(color)))
        elif pose == "jumping":
            out.append(_poly([(w * 0.3, h * 0.32), (w * 0.2, h * 0.3), (w * 0.02, h * 0.1), (w * 0.08, h * 0.05)], hexcolor(color)))
            out.append(_poly([(w * 0.7, h * 0.32), (w * 0.8, h * 0.3), (w * 0.98, h * 0.1), (w * 0.92, h * 0.05)], hexcolor(color)))
        else:
            out.append(_rect(w * 0.14, h * 0.32, w * 0.12, h * 0.3, hexcolor(color)))
            out.append(_rect(w * 0.74, h * 0.32, w * 0.12, h * 0.3, hexcolor(color)))
    out.append(_circle(hx, hy + head_r * 0.2, head_r, skin))
    out.append(_rect(hx - head_r, hy - head_r * 0.9, head_r * 2, head_r * 0.9, "#4a2f1a", rx=head_r * 0.4))
    eye_dx = 0.0
    if facing == "left":
        eye_dx = -head_r * 0.65
    elif facing == "right":
        eye_dx = head_r * 0.65
    out.append(_circle(hx + eye_dx, hy + head_r * 0.15, head_r * 0.16, "#222222", ' data-anchor="face"'))
    return out


def draw_quadruped(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    dark = _sec(color, secondary)
    out: List[str] = []
    if pose == "sitting":
        out.append(_ellipse(w * 0.5, h * 0.66, w * 0.3, h * 0.3, hexcolor(color)))
        out.append(_rect(w * 0.3, h * 0.78, w * 0.12, h * 0.2, dark, rx=w * 0.04))
        out.append(_rect(w * 0.58, h * 0.78, w * 0.12, h * 0.2, dark, rx=w * 0.04))
        hx = head_x(facing, w, 0.32)
        hy = h * 0.3
        hr = w * 0.2
    elif pose == "lying":
        out.append(_ellipse(w * 0.5, h * 0.66, w * 0.42, h * 0.22, hexcolor(color)))
        hx = head_x(facing, w, 0.16)
        hy = h * 0.5
        hr = h * 0.2
    else:
        out.append(_ellipse(w * 0.5, h * 0.5, w * 0.34, h * 0.22, hexcolor(color)))
        spread = w * 0.08 if pose in ("running", "walking") else 0.0
        for i, lx in enumerate((w * 0.24, w * 0.38, w * 0.58, w * 0.72)):
            dx = spread if i % 2 == 0 else -spread
            out.append(_poly([(lx, h * 0.6), (lx + w * 0.08, h * 0.6), (lx + w * 0.08 + dx, h * 0.94), (lx + dx, h * 0.94)], dark))
        hx = head_x(facing, w, 0.16)
        hy = h * 0.32
        hr = h * 0.17
    tail_x = w - hx
    out.append(_poly([(tail_x, hy + hr * 0.6), (tail_x + (hr * 1.2 if tail_x > w / 2 else -hr * 1.2), hy - hr * 1.2), (tail_x + (hr * 0.5 if tail_x > w / 2 else -hr * 0.5), hy + hr * 0.4)], dark))
    out.append(_circle(hx, hy, hr, hexcolor(color), ' data-anchor="face"'))
    out.append(_poly([(hx - hr * 0.8, hy - hr * 0.5), (hx - hr * 0.5, hy - hr * 1.5), (hx - hr * 0.2, hy - hr * 0.6)], dark))
    out.append(_poly([(hx + hr * 0.8, hy - hr * 0.5), (hx + hr * 0.5, hy - hr * 1.5), (hx + hr * 0.2, hy - hr * 0.6)], dark))
    out.append(_circle(hx - hr * 0.3, hy - hr * 0.1, hr * 0.12, "#111111"))
    out.append(_circle(hx + hr * 0.3, hy - hr * 0.1, hr * 0.12, "#111111"))
    return out


def draw_bird(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    dark = _sec(color, secondary)
    out = [_ellipse(w * 0.5, h * 0.58, w * 0.3, h * 0.24, hexcolor(color))]
    hx = head_x(facing, w, 0.2)
    out.append(_circle(hx, h * 0.3, h * 0.16, hexcolor(color), ' data-anchor="face"'))
    beak_dir = -1 if hx < w / 2 else 1
    out.append(_poly([(hx + beak_dir * h * 0.14, h * 0.28), (hx + beak_dir * h * 0.3, h * 0.33), (hx + beak_dir * h * 0.14, h * 0.38)], "#f5a623"))
    wing_y = h * 0.3 if pose == "flying" else h * 0.55
    out.append(_ellipse(w * 0.5, wing_y, w * 0.22, h * 0.1, dark))
    if pose != "flying":
        out.append(_rect(w * 0.42, h * 0.8, w * 0.04, h * 0.16, "#f5a623"))
        out.append(_rect(w * 0.54, h * 0.8, w * 0.04, h * 0.16, "#f5a623"))
    return out


def draw_fish(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    dark = _sec(color, secondary)
    hx = head_x(facing, w, 0.2)
    tail_x = w - hx
    out = [_ellipse(w * 0.5, h * 0.5, w * 0.32, h * 0.3, hexcolor(color))]
    out.append(_poly([(tail_x, h * 0.5), (tail_x + (w * 0.18 if tail_x > w / 2 else -w * 0.18), h * 0.2), (tail_x + (w * 0.18 if tail_x > w / 2 else -w * 0.18), h * 0.8)], dark))
    out.append(_circle(hx, h * 0.45, h * 0.06, "#111111", ' data-anchor="face"'))
    return out


def draw_tree(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.42, h * 0.55, w * 0.16, h * 0.45, "#7a4a1e"),
        _circle(w * 0.5, h * 0.36, w * 0.36, hexcolor(color)),
        _circle(w * 0.3, h * 0.5, w * 0.24, hexcolor(color)),
        _circle(w * 0.7, h * 0.5, w * 0.24, hexcolor(color)),
    ]


def draw_house(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.1, h * 0.45, w * 0.8, h * 0.55, hexcolor(color)),
        _poly([(w * 0.02, h * 0.47), (w * 0.5, h * 0.05), (w * 0.98, h * 0.47)], "#a03a2a"),
        _rect(w * 0.42, h * 0.68, w * 0.16, h * 0.32, "#5a3a1a"),
        _rect(w * 0.16, h * 0.55, w * 0.16, h * 0.16, "#8fd3ff"),
        _rect(w * 0.68, h * 0.55, w * 0.16, h * 0.16, "#8fd3ff"),
    ]


def draw_car(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.05, h * 0.45, w * 0.9, h * 0.3, hexcolor(color), rx=h * 0.06),
        _poly([(w * 0.22, h * 0.46), (w * 0.34, h * 0.18), (w * 0.7, h * 0.18), (w * 0.82, h * 0.46)], _sec(color, secondary)),
        _rect(w * 0.36, h * 0.22, w * 0.13, h * 0.2, "#bfe6ff"),
        _rect(w * 0.53, h * 0.22, w * 0.13, h * 0.2, "#bfe6ff"),
        _circle(w * 0.25, h * 0.8, h * 0.14, "#222222"),
        _circle(w * 0.75, h * 0.8, h * 0.14, "#222222"),
    ]


def draw_bicycle(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _circle(w * 0.22, h * 0.7, h * 0.26, "#333333"),
        _circle(w * 0.78, h * 0.7, h * 0.26, "#333333"),
        _circle(w * 0.22, h * 0.7, h * 0.18, "#dddddd"),
        _circle(w * 0.78, h * 0.7, h * 0.18, "#dddddd"),
        _poly([(w * 0.22, h * 0.7), (w * 0.45, h * 0.3), (w * 0.72, h * 0.3), (w * 0.78, h * 0.7), (w * 0.5, h * 0.7)], hexcolor(color)),
        _rect(w * 0.42, h * 0.16, w * 0.14, h * 0.06, "#222222"),
    ]


def draw_round(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    r = min(w, h) * 0.42
    return [
        _circle(w * 0.5, h * 0.55, r, hexcolor(color)),
        _rect(w * 0.48, h * 0.55 - r - h * 0.06, w * 0.04, h * 0.1, "#5a3a1a"),
    ]


def draw_banana(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [f'<path d="M {w*0.08:.1f} {h*0.45:.1f} Q {w*0.5:.1f} {h*0.95:.1f} {w*0.92:.1f} {h*0.45:.1f} Q {w*0.5:.1f} {h*0.7:.1f} {w*0.08:.1f} {h*0.45:.1f} Z" fill="{hexcolor(color)}"/>']


def draw_cup(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.22, h * 0.25, w * 0.5, h * 0.7, hexcolor(color), rx=w * 0.06),
        _rect(w * 0.7, h * 0.4, w * 0.16, h * 0.32, _sec(color, secondary), rx=w * 0.06),
        _ellipse(w * 0.47, h * 0.25, w * 0.25, h * 0.06, _sec(color, secondary)),
    ]


def draw_table(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.04, h * 0.3, w * 0.92, h * 0.14, hexcolor(color)),
        _rect(w * 0.1, h * 0.44, w * 0.08, h * 0.54, _sec(color, secondary)),
        _rect(w * 0.82, h * 0.44, w * 0.08, h * 0.54, _sec(color, secondary)),
    ]


def draw_chair(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.15, h * 0.5, w * 0.7, h * 0.12, hexcolor(color)),
        _rect(w * 0.15, h * 0.05, w * 0.12, h * 0.5, hexcolor(color)),
        _rect(w * 0.18, h * 0.62, w * 0.08, h * 0.36, _sec(color, secondary)),
        _rect(w * 0.74, h * 0.62, w * 0.08, h * 0.36, _sec(color, secondary)),
    ]


def draw_umbrella(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        f'<path d="M {w*0.05:.1f} {h*0.5:.1f} A {w*0.45:.1f} {h*0.42:.1f} 0 0 1 {w*0.95:.1f} {h*0.5:.1f} Z" fill="{hexcolor(color)}"/>',
        _rect(w * 0.48, h * 0.5, w * 0.04, h * 0.45, "#333333"),
    ]


def draw_disk(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [_circle(w * 0.5, h * 0.5, min(w, h) * 0.42, hexcolor(color))]


def draw_cloud(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _ellipse(w * 0.5, h * 0.6, w * 0.45, h * 0.28, hexcolor(color)),
        _circle(w * 0.35, h * 0.45, h * 0.28, hexcolor(color)),
        _circle(w * 0.6, h * 0.4, h * 0.32, hexcolor(color)),
    ]


def draw_flower(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    out = [_rect(w * 0.47, h * 0.45, w * 0.06, h * 0.55, "#3a9a3a")]
    cx, cy = w * 0.5, h * 0.3
    for k in range(6):
        a = 2 * math.pi * k / 6
        out.append(_ellipse(cx + math.cos(a) * w * 0.2, cy + math.sin(a) * h * 0.14, w * 0.14, h * 0.09, hexcolor(color)))
    out.append(_circle(cx, cy, w * 0.1, "#f5c400"))
    return out


def draw_box(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.08, h * 0.15, w * 0.84, h * 0.75, hexcolor(color), rx=w * 0.03),
        _rect(w * 0.08, h * 0.15, w * 0.84, h * 0.14, _sec(color, secondary)),
    ]


def draw_boat(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _poly([(w * 0.02, h * 0.62), (w * 0.98, h * 0.62), (w * 0.8, h * 0.95), (w * 0.2, h * 0.95)], hexcolor(color)),
        _rect(w * 0.48, h * 0.05, w * 0.04, h * 0.57, "#333333"),
        _poly([(w * 0.52, h * 0.08), (w * 0.9, h * 0.58), (w * 0.52, h * 0.58)], "#f5f5f5"),
    ]


def draw_star(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    cx, cy, R, r = w * 0.5, h * 0.5, min(w, h) * 0.46, min(w, h) * 0.2
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rad = R if k % 2 == 0 else r
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return [_poly(pts, hexcolor(color))]


def draw_lamp(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.46, h * 0.3, w * 0.08, h * 0.62, "#444444"),
        _rect(w * 0.25, h * 0.9, w * 0.5, h * 0.08, "#444444"),
        _poly([(w * 0.2, h * 0.32), (w * 0.8, h * 0.32), (w * 0.65, h * 0.04), (w * 0.35, h * 0.04)], hexcolor(color)),
    ]


def draw_hat(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.05, h * 0.62, w * 0.9, h * 0.12, hexcolor(color), rx=h * 0.05),
        _rect(w * 0.28, h * 0.15, w * 0.44, h * 0.5, hexcolor(color), rx=w * 0.04),
    ]


def draw_kite(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _poly([(w * 0.5, h * 0.02), (w * 0.9, h * 0.4), (w * 0.5, h * 0.8), (w * 0.1, h * 0.4)], hexcolor(color)),
        _rect(w * 0.49, h * 0.8, w * 0.02, h * 0.18, "#333333"),
    ]


def draw_cake(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.1, h * 0.5, w * 0.8, h * 0.4, hexcolor(color), rx=w * 0.03),
        _rect(w * 0.2, h * 0.25, w * 0.6, h * 0.28, lighter(color, 0.35), rx=w * 0.03),
        _rect(w * 0.48, h * 0.05, w * 0.04, h * 0.2, "#ffcc00"),
    ]


def draw_basket(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _poly([(w * 0.08, h * 0.4), (w * 0.92, h * 0.4), (w * 0.8, h * 0.92), (w * 0.2, h * 0.92)], hexcolor(color)),
        f'<path d="M {w*0.2:.1f} {h*0.4:.1f} A {w*0.3:.1f} {h*0.35:.1f} 0 0 1 {w*0.8:.1f} {h*0.4:.1f}" fill="none" stroke="{_sec(color, secondary)}" stroke-width="{w*0.05:.1f}"/>',
    ]


def draw_tent(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _poly([(w * 0.05, h * 0.92), (w * 0.5, h * 0.08), (w * 0.95, h * 0.92)], hexcolor(color)),
        _poly([(w * 0.38, h * 0.92), (w * 0.5, h * 0.5), (w * 0.62, h * 0.92)], _sec(color, secondary)),
    ]


def draw_generic(color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return [
        _rect(w * 0.15, h * 0.3, w * 0.7, h * 0.6, hexcolor(color), rx=w * 0.08),
        _circle(w * 0.5, h * 0.28, w * 0.2, _sec(color, secondary)),
    ]


TEMPLATES: Dict[str, Callable[..., List[str]]] = {
    "person": draw_person,
    "quadruped": draw_quadruped,
    "bird": draw_bird,
    "fish": draw_fish,
    "tree": draw_tree,
    "house": draw_house,
    "car": draw_car,
    "bicycle": draw_bicycle,
    "round": draw_round,
    "banana": draw_banana,
    "cup": draw_cup,
    "table": draw_table,
    "chair": draw_chair,
    "umbrella": draw_umbrella,
    "disk": draw_disk,
    "cloud": draw_cloud,
    "flower": draw_flower,
    "box": draw_box,
    "boat": draw_boat,
    "star": draw_star,
    "lamp": draw_lamp,
    "hat": draw_hat,
    "kite": draw_kite,
    "cake": draw_cake,
    "basket": draw_basket,
    "tent": draw_tent,
    "generic": draw_generic,
}


def draw_noun(noun: str, color: str, w: float, h: float, pose: str, facing: str, secondary: Optional[str] = None) -> List[str]:
    return TEMPLATES[noun_category(noun)](color, w, h, pose, facing, secondary)


SLOT_ORDER = ["sky", "wall", "mountain", "sun", "moon", "stars", "cloud", "forest", "sea", "ground", "sand", "snow", "floor", "road"]
SLOT_DEFAULT = {"sky": "#8ecbff", "wall": "#e8dcc8", "mountain": "#8a7f9a", "sun": "#ffd23f", "moon": "#f4f1de", "stars": "#ffffff", "cloud": "#ffffff", "forest": "#2e8b57", "sea": "#2f7fd6", "ground": "#5cb85c", "sand": "#e8d29a", "snow": "#f4f7fb", "floor": "#c9a97a", "road": "#7d7d7d"}
COLOR_SLOTS = {
    "blue": ["sky", "sea", "wall"], "skyblue": ["sky", "sea"], "lightblue": ["sky", "sea"], "navy": ["sky", "sea"], "cyan": ["sea", "sky"], "teal": ["sea", "wall"],
    "green": ["ground", "forest", "wall"], "darkgreen": ["forest", "ground"], "lime": ["ground"],
    "yellow": ["sand", "sun", "wall"], "gold": ["sun", "sand"], "tan": ["sand", "floor", "wall"], "beige": ["wall", "sand", "floor"], "wooden": ["floor", "wall"], "brown": ["ground", "floor", "wall"],
    "gray": ["road", "sky", "wall", "floor"], "grey": ["road", "sky", "wall", "floor"], "silver": ["road", "sky"], "white": ["snow", "wall", "cloud", "floor"],
    "orange": ["sky", "sand"], "pink": ["sky", "wall"], "purple": ["sky", "wall"], "violet": ["sky"], "red": ["wall", "ground", "floor"], "black": ["sky", "road", "ground"],
}


def _slot_of(name: str) -> Optional[str]:
    n = singular(name)
    if n in {"sky", "horizon", "night", "sunset", "dusk", "dawn", "sunrise"}:
        return "sky"
    if n in {"wall", "room", "kitchen", "classroom", "interior"}:
        return "wall"
    if n in {"mountain", "hill", "cliff"}:
        return "mountain"
    if n in {"sun"}:
        return "sun"
    if n in {"moon"}:
        return "moon"
    if n in {"star"}:
        return "stars"
    if n in {"cloud"}:
        return "cloud"
    if n in {"forest", "tree", "wood", "pine"}:
        return "forest"
    if n in {"sea", "ocean", "lake", "river", "pond", "water", "wave"}:
        return "sea"
    if n in {"grass", "field", "meadow", "lawn", "park", "garden", "farm", "ground", "yard", "dirt"}:
        return "ground"
    if n in {"sand", "beach", "desert", "dune"}:
        return "sand"
    if n in {"snow", "ice"}:
        return "snow"
    if n in {"floor", "carpet", "rug", "tile"}:
        return "floor"
    if n in {"road", "street", "path", "pavement", "sidewalk"}:
        return "road"
    return None


def background_elements(elements: Sequence[str], colors: Sequence[str], w: float, h: float) -> List[str]:
    names = [str(e).strip().lower() for e in elements]
    slots: List[str] = []
    for n in names:
        slot = _slot_of(n)
        if slot and slot not in slots:
            slots.append(slot)
    night = any(singular(n) in {"night", "star", "moon"} for n in names)
    sunset = any(singular(n) in {"sunset", "dusk", "dawn", "sunrise"} for n in names)
    indoor = "wall" in slots
    if not slots:
        slots = ["ground"]
    if not any(s in slots for s in ("sky", "wall")):
        slots.insert(0, "wall" if any(s in slots for s in ("floor",)) else "sky")
    if not any(s in slots for s in ("ground", "sand", "snow", "floor", "road", "sea")):
        slots.append("floor" if indoor else "ground")
    fills: Dict[str, str] = {s: SLOT_DEFAULT[s] for s in slots}
    if night:
        fills["sky"] = "#1a2350"
    elif sunset:
        fills["sky"] = "#ff9a4a"
    used: List[str] = []
    for c in colors:
        key = S.normalize_color_name(c)
        if S.parse_color(c) is None:
            continue
        target = None
        for candidate in COLOR_SLOTS.get(key, []):
            if candidate in slots and candidate not in used:
                target = candidate
                break
        if target is None:
            for candidate in ["ground", "sand", "snow", "floor", "road", "sea", "sky", "wall"]:
                if candidate in slots and candidate not in used:
                    target = candidate
                    break
        if target is None:
            continue
        fills[target] = hexcolor(c)
        used.append(target)
    lower_slots = [s for s in ("sea", "ground", "sand", "snow", "floor", "road") if s in slots]
    out: List[str] = []
    for slot in SLOT_ORDER:
        if slot not in slots:
            continue
        fill = fills[slot]
        if slot == "sky":
            out.append(_rect(0, 0, w, h * 0.62, fill))
        elif slot == "wall":
            out.append(_rect(0, 0, w, h * 0.7, fill))
        elif slot == "mountain":
            out.append(f'<g>{_poly([(0, h * 0.62), (w * 0.25, h * 0.28), (w * 0.5, h * 0.62)], fill)}{_poly([(w * 0.4, h * 0.62), (w * 0.7, h * 0.22), (w, h * 0.62)], darker(fill, 0.85))}</g>')
        elif slot == "sun":
            out.append(_circle(w * 0.82, h * 0.16, min(w, h) * 0.08, fill))
        elif slot == "moon":
            out.append(_circle(w * 0.8, h * 0.16, min(w, h) * 0.07, fill))
        elif slot == "stars":
            out.append("<g>" + "".join(_circle(w * (0.1 + 0.15 * k), h * (0.08 + 0.04 * (k % 3)), min(w, h) * 0.01, fill) for k in range(6)) + "</g>")
        elif slot == "cloud":
            parts = []
            for k in range(2):
                cx = w * (0.25 + 0.45 * k)
                parts.append(_ellipse(cx, h * 0.18, w * 0.09, h * 0.05, fill))
                parts.append(_circle(cx - w * 0.04, h * 0.15, h * 0.05, fill))
            out.append("<g>" + "".join(parts) + "</g>")
        elif slot == "forest":
            parts = []
            for k in range(5):
                cx = w * (0.1 + 0.2 * k)
                parts.append(_rect(cx - w * 0.01, h * 0.5, w * 0.02, h * 0.14, "#5a3a1a"))
                parts.append(_circle(cx, h * 0.48, h * 0.08, fill))
            out.append("<g>" + "".join(parts) + "</g>")
        elif slot in ("sea", "ground", "sand", "snow", "floor", "road"):
            idx = lower_slots.index(slot)
            n = len(lower_slots)
            top = h * (0.62 + 0.38 * idx / n)
            height = h * 0.38 / n
            if slot == "road" and n > 1:
                out.append(_rect(0, top + height * 0.2, w, height * 0.6, fill))
            else:
                out.append(_rect(0, top, w, height, fill))
    return out


def grid_layout(count: int, w: float, h: float, cell_aspect: float) -> List[Tuple[float, float, float, float]]:
    best = None
    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        cw, ch = w / cols, h / rows
        s = min(cw / cell_aspect, ch)
        if best is None or s > best[0]:
            best = (s, cols, rows)
    _s, cols, rows = best
    cw, ch = w / cols, h / rows
    cells = []
    for i in range(count):
        r, c = divmod(i, cols)
        cells.append((c * cw + cw * 0.04, r * ch + ch * 0.04, cw * 0.92, ch * 0.92))
    return cells
