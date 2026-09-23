from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import svg as S
from .dag import BACKGROUND, DAG


def placement_transform(bbox_local: Tuple[float, float, float, float], target: List[float], stretch: bool) -> Tuple[str, List[float]]:
    lx, ly, lw, lh = bbox_local
    tx1, ty1, tx2, ty2 = target
    tw, th = tx2 - tx1, ty2 - ty1
    if stretch:
        sx, sy = tw / lw, th / lh
        ox, oy = tx1, ty1
        actual = [tx1, ty1, tx2, ty2]
    else:
        s = min(tw / lw, th / lh)
        sx = sy = s
        ox = tx1 + (tw - lw * s) / 2.0
        oy = ty1 + (th - lh * s) / 2.0
        actual = [ox, oy, ox + lw * s, oy + lh * s]
    transform = f"translate({ox - lx * sx:.2f} {oy - ly * sy:.2f}) scale({sx:.5f} {sy:.5f})"
    return transform, actual


def compose(dag: DAG, layout: Dict[str, Any]) -> Dict[str, Any]:
    W, H = dag.canvas
    placements = [p for p in (layout.get("placements") or []) if isinstance(p, dict) and p.get("node") in dag]
    ordered = sorted(placements, key=lambda p: (int(p.get("z", 1)), 0))
    body_parts: List[str] = []
    legend: List[Dict[str, Any]] = []
    for p in ordered:
        nid = str(p["node"])
        node = dag[nid]
        if not node.output or not isinstance(node.output, str):
            continue
        try:
            root = S.parse_fragment(node.output, nid)
        except (ValueError, Exception):
            continue
        bbox_local = S.declared_bbox(root)
        if bbox_local is None:
            b = S.bounds(root)
            if b is None:
                continue
            bbox_local = (b[0], b[1], max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1]))
        try:
            target = [float(v) for v in p.get("bbox")]
        except (TypeError, ValueError):
            continue
        if len(target) != 4 or target[2] <= target[0] or target[3] <= target[1]:
            continue
        transform, actual = placement_transform(bbox_local, target, stretch=(node.type == BACKGROUND))
        body_parts.append(f'<g data-place="{nid}" transform="{transform}">{S.serialize(root)}</g>')
        legend.append({
            "node": nid,
            "type": node.type,
            "desc": node.desc(),
            "count": node.count() if node.type in ("entity", "object") else 1,
            "bbox": [round(v, 1) for v in actual],
            "z": int(p.get("z", 1)),
            "participants": node.participants() if node.type == "action" else [],
        })
    svg = S.svg_document(W, H, "".join(body_parts))
    return {"svg": svg, "legend": legend, "canvas": [W, H]}


def legend_text(legend: List[Dict[str, Any]]) -> str:
    lines = []
    for item in legend:
        extra = f" participants={item['participants']}" if item.get("participants") else ""
        lines.append(f"- {item['node']} [{item['type']}]: {item['desc']} | count={item.get('count', 1)} | bbox={item['bbox']}{extra}")
    return "\n".join(lines) if lines else "(empty)"
