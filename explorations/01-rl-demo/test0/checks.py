from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import svg as S
from .dag import ACTION, BACKGROUND, DAG, ENTITY, OBJECT, Node

FRAGMENT_ITEMS = ("syntax", "no_forbidden", "bbox", "count", "color", "pose", "elements", "participants")
LAYOUT_ITEMS = ("complete", "in_canvas", "sizes", "overlap", "relations")


def _item(name: str, passed: bool, weight: float, detail: str = "", skipped: bool = False) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "weight": float(weight), "detail": detail, "skipped": bool(skipped)}


def _finish(items: List[Dict[str, Any]], fatal: bool) -> Dict[str, Any]:
    active = [i for i in items if not i["skipped"]]
    total = sum(i["weight"] for i in active) or 1.0
    got = sum(i["weight"] for i in active if i["passed"])
    return {
        "passed": all(i["passed"] for i in active),
        "score": round(got / total, 4),
        "fatal": bool(fatal),
        "items": items,
        "failures": [i for i in active if not i["passed"]],
    }


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _pose_ok(declared: str, required: str) -> bool:
    d, r = _norm(declared), _norm(required)
    if not r or r == "none":
        return True
    if not d:
        return False
    return d == r or r in d or d in r


def _facing_side(root: ET.Element, bbox: Tuple[float, float, float, float]) -> Optional[str]:
    anchor = None
    anchor_matrix: S.Matrix = S.IDENTITY
    for elem, m in S.walk_shapes(root):
        if elem.get("data-anchor") == "face":
            anchor, anchor_matrix = elem, m
            break
    if anchor is None:
        return None
    ab = S.shape_bounds(anchor, anchor_matrix)
    if ab is None:
        return None
    cx = (ab[0] + ab[2]) / 2.0
    parent_map = {child: parent for parent in root.iter() for child in parent}
    container = parent_map.get(anchor)
    instance = None
    while container is not None:
        if container.get("data-instance") is not None:
            instance = container
            break
        container = parent_map.get(container)
    if instance is not None:
        chain = []
        node = instance
        while node is not None and node is not root:
            chain.append(node)
            node = parent_map.get(node)
        m: S.Matrix = S.parse_transform(root.get("transform"))
        for elem in reversed(chain[1:]):
            m = S.matrix_multiply(m, S.parse_transform(elem.get("transform")))
        ib = S.bounds(instance, m)
        if ib is None:
            return None
        mid = (ib[0] + ib[2]) / 2.0
        width = ib[2] - ib[0]
    else:
        m = S.parse_transform(root.get("transform"))
        mid = S.matrix_apply(m, bbox[0] + bbox[2] / 2.0, 0)[0]
        width = bbox[2] * abs(m[0]) if m[0] else bbox[2]
    if abs(cx - mid) < max(1e-6, width) * 0.08:
        return "front"
    return "right" if cx > mid else "left"


def check_fragment(fragment: str, node: Node, dag: Optional[DAG] = None) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    try:
        root = S.parse_fragment(fragment, node.id)
    except (ET.ParseError, ValueError) as exc:
        items.append(_item("syntax", False, 3.0, f"unparseable svg: {exc}"))
        for name in FRAGMENT_ITEMS[1:]:
            items.append(_item(name, False, 1.0, "not evaluated", skipped=True))
        return _finish(items, fatal=True)
    bad_tags = sorted({S.local_tag(e) for e in root.iter() if S.local_tag(e) not in S.ALLOWED_TAGS})
    forbidden = [t for t in bad_tags if t in S.FORBIDDEN_TAGS]
    unknown = [t for t in bad_tags if t not in S.FORBIDDEN_TAGS]
    shapes = S.shape_count(root)
    if shapes == 0:
        items.append(_item("syntax", False, 3.0, "fragment draws no shapes"))
        for name in FRAGMENT_ITEMS[1:]:
            items.append(_item(name, False, 1.0, "not evaluated", skipped=True))
        return _finish(items, fatal=True)
    items.append(_item("syntax", not unknown, 3.0, f"unknown tags {unknown}" if unknown else f"{shapes} shapes"))
    items.append(_item("no_forbidden", not forbidden, 1.0, f"forbidden tags {forbidden}" if forbidden else "ok"))
    bbox = S.declared_bbox(root)
    bounds = S.bounds(root)
    if bbox is None:
        items.append(_item("bbox", False, 1.0, "missing data-bbox"))
    elif bounds is None:
        items.append(_item("bbox", False, 1.0, "no drawable bounds"))
    else:
        x, y, w, h = bbox
        tol_x, tol_y = w * 0.05, h * 0.05
        inside = bounds[0] >= x - tol_x and bounds[1] >= y - tol_y and bounds[2] <= x + w + tol_x and bounds[3] <= y + h + tol_y
        cov = S.coverage(root, bbox)
        big_enough = cov >= 0.12
        detail = f"bounds={tuple(round(v, 1) for v in bounds)} bbox={bbox} coverage={cov:.2f}"
        items.append(_item("bbox", inside and big_enough, 1.0, detail if (inside and big_enough) else ("content leaves declared bbox; " if not inside else "content too small; ") + detail))
    if node.type in (ENTITY, OBJECT):
        want = node.count()
        inst = len(S.instances(root))
        if want > 1:
            items.append(_item("count", inst == want, 2.0, f"instances={inst} expected={want}"))
        else:
            items.append(_item("count", inst <= 1, 2.0, f"instances={inst} expected<=1"))
    elif node.type == ACTION:
        items.append(_item("count", True, 2.0, "n/a", skipped=True))
    else:
        items.append(_item("count", True, 2.0, "n/a", skipped=True))
    colors = node.colors()
    if colors:
        found = S.fills(root)
        missing = [c for c in colors if not any(S.color_matches(rgb, c) for rgb in found)]
        items.append(_item("color", not missing, 2.0, f"missing colors {missing}" if missing else f"colors present {colors}"))
    else:
        items.append(_item("color", True, 2.0, "no color constraint", skipped=True))
    if node.type == ENTITY:
        pose_req = str(node.spec.get("pose") or "")
        face_req = _norm(node.spec.get("facing") or "")
        pose_declared = root.get("data-pose") or ""
        face_declared = _norm(root.get("data-facing") or "")
        ok = _pose_ok(pose_declared, pose_req)
        detail = f"pose={pose_declared!r} required={pose_req!r}"
        if face_req and face_req != "none":
            ok = ok and face_declared == face_req
            detail += f" facing={face_declared!r} required={face_req!r}"
            if bbox is not None:
                side = _facing_side(root, bbox)
                if side is not None and face_req in ("left", "right") and side != face_req:
                    ok = False
                    detail += f" face-anchor-on={side}"
        items.append(_item("pose", ok, 1.0, detail))
    elif node.type == ACTION:
        poses = node.spec.get("poses") if isinstance(node.spec.get("poses"), dict) else {}
        problems = []
        for pid, required in poses.items():
            g = root.find(f".//g[@data-node='{pid}']")
            if g is None:
                continue
            if not _pose_ok(g.get("data-pose") or "", str(required)):
                problems.append(f"{pid}: pose={g.get('data-pose')!r} required={required!r}")
        items.append(_item("pose", not problems, 1.0, "; ".join(problems) if problems else "poses consistent"))
    else:
        items.append(_item("pose", True, 1.0, "n/a", skipped=True))
    if node.type == BACKGROUND:
        elements = node.spec.get("elements") or []
        need = len(elements) if isinstance(elements, list) else 0
        top_level = len([c for c in list(root) if S.local_tag(c) in S.SHAPE_TAGS or S.local_tag(c) == "g"])
        required = max(1, (need + 1) // 2)
        items.append(_item("elements", top_level >= required, 1.0, f"top-level parts={top_level} elements={need} required>={required}"))
    else:
        items.append(_item("elements", True, 1.0, "n/a", skipped=True))
    if node.type == ACTION:
        problems = []
        for pid in node.participants():
            groups = root.findall(f".//g[@data-node='{pid}']")
            if len(groups) != 1:
                problems.append(f"{pid} appears {len(groups)} times")
            elif dag is not None and pid in dag and S.shape_count(groups[0]) == 0:
                problems.append(f"{pid} is empty")
        items.append(_item("participants", not problems, 2.0, "; ".join(problems) if problems else "all participants embedded once"))
    else:
        items.append(_item("participants", True, 2.0, "n/a", skipped=True))
    return _finish(items, fatal=False)


def relation_holds(rel: str, a: Sequence[float], b: Sequence[float]) -> bool:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    r = _norm(rel).replace(" ", "_")
    if r in ("left_of", "left"):
        return ax < bx and a[2] <= b[0] + 0.25 * min(aw, bw)
    if r in ("right_of", "right"):
        return ax > bx and a[0] >= b[2] - 0.25 * min(aw, bw)
    if r == "above":
        return ay < by and a[3] <= b[1] + 0.25 * min(ah, bh)
    if r == "below":
        return ay > by and a[1] >= b[3] - 0.25 * min(ah, bh)
    if r == "on":
        horizontal = ax >= b[0] - 0.1 * bw and ax <= b[2] + 0.1 * bw
        return horizontal and abs(a[3] - b[1]) <= 0.35 * max(ah, bh) and ay < by
    if r == "under":
        horizontal = ax >= b[0] - 0.1 * bw and ax <= b[2] + 0.1 * bw
        return horizontal and ay > by
    if r == "inside":
        return a[0] >= b[0] - 0.05 * bw and a[1] >= b[1] - 0.05 * bh and a[2] <= b[2] + 0.05 * bw and a[3] <= b[3] + 0.05 * bh
    if r == "near":
        return abs(ax - bx) <= 1.2 * (aw + bw) and abs(ay - by) <= 1.2 * (ah + bh)
    if r == "far_from":
        return abs(ax - bx) > 1.0 * (aw + bw) or abs(ay - by) > 1.0 * (ah + bh)
    if r == "larger_than":
        return aw * ah > 1.2 * bw * bh
    if r == "smaller_than":
        return aw * ah * 1.2 < bw * bh
    if r == "behind":
        return ay <= by
    if r == "in_front_of":
        return ay >= by
    return True


def check_layout(layout: Dict[str, Any], dag: DAG) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    W, H = dag.canvas
    placements = layout.get("placements") if isinstance(layout, dict) else None
    if not isinstance(placements, list):
        items.append(_item("complete", False, 3.0, "layout has no placements"))
        for name in LAYOUT_ITEMS[1:]:
            items.append(_item(name, False, 1.0, "not evaluated", skipped=True))
        return _finish(items, fatal=True)
    expected = {n.id for n in dag.placeables()}
    seen: Dict[str, List[float]] = {}
    unknown: List[str] = []
    duplicates: List[str] = []
    invalid: List[str] = []
    for p in placements:
        if not isinstance(p, dict):
            invalid.append(str(p))
            continue
        nid = str(p.get("node") or "")
        bbox = p.get("bbox")
        if nid not in expected:
            unknown.append(nid)
            continue
        if nid in seen:
            duplicates.append(nid)
            continue
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            invalid.append(nid)
            continue
        try:
            b = [float(v) for v in bbox]
        except (TypeError, ValueError):
            invalid.append(nid)
            continue
        if b[2] <= b[0] or b[3] <= b[1]:
            invalid.append(nid)
            continue
        seen[nid] = b
    missing = sorted(expected - set(seen))
    detail = []
    if missing:
        detail.append(f"missing {missing}")
    if unknown:
        detail.append(f"unknown {unknown}")
    if duplicates:
        detail.append(f"duplicate {duplicates}")
    if invalid:
        detail.append(f"invalid {invalid}")
    complete = not (missing or unknown or duplicates or invalid)
    items.append(_item("complete", complete, 3.0, "; ".join(detail) if detail else f"{len(seen)} placements"))
    if not seen:
        for name in LAYOUT_ITEMS[1:]:
            items.append(_item(name, False, 1.0, "not evaluated", skipped=True))
        return _finish(items, fatal=True)
    outside = []
    for nid, b in seen.items():
        if b[0] < -0.02 * W or b[1] < -0.02 * H or b[2] > 1.02 * W or b[3] > 1.02 * H:
            outside.append(nid)
    items.append(_item("in_canvas", not outside, 2.0, f"outside canvas {outside}" if outside else "all inside"))
    subjects = {nid: b for nid, b in seen.items() if dag[nid].type != BACKGROUND}
    tiny = [nid for nid, b in subjects.items() if (b[2] - b[0]) * (b[3] - b[1]) < 0.005 * W * H]
    union = _union_area(list(subjects.values()), W, H) / float(W * H) if subjects else 1.0
    ok_sizes = not tiny and union >= 0.2
    items.append(_item("sizes", ok_sizes, 1.0, f"tiny {tiny}; subject coverage={union:.2f}"))
    scene = dag[dag.scene_id()]
    rels = scene.spec.get("relations") or []
    consumed = dag.consumed()
    contained = set()
    for rel in rels:
        if isinstance(rel, dict) and str(rel.get("rel")) == "inside":
            a_ref = consumed.get(str(rel.get("a")), str(rel.get("a")))
            b_ref = consumed.get(str(rel.get("b")), str(rel.get("b")))
            contained.add((a_ref, b_ref))
            contained.add((b_ref, a_ref))
    overlaps = []
    ids = list(subjects.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if (ids[i], ids[j]) in contained:
                continue
            frac = S.bbox_overlap_fraction(subjects[ids[i]], subjects[ids[j]])
            if frac > 0.4:
                overlaps.append((ids[i], ids[j], round(frac, 2)))
    items.append(_item("overlap", not overlaps, 1.0, f"heavy overlaps {overlaps}" if overlaps else "ok"))
    failed = []
    checked = 0
    for rel in rels:
        if not isinstance(rel, dict):
            continue
        a, b, r = str(rel.get("a")), str(rel.get("b")), str(rel.get("rel") or "near")
        a_ref = consumed.get(a, a)
        b_ref = consumed.get(b, b)
        if a_ref == b_ref:
            continue
        if a_ref not in seen or b_ref not in seen:
            continue
        checked += 1
        if not relation_holds(r, seen[a_ref], seen[b_ref]):
            failed.append(f"{a} {r} {b}")
    if checked == 0:
        items.append(_item("relations", True, 2.0, "no relations", skipped=True))
    else:
        items.append(_item("relations", not failed, 2.0, f"violated {failed}" if failed else f"{checked} relations hold"))
    return _finish(items, fatal=False)


def _union_area(boxes: List[List[float]], W: int, H: int, grid: int = 64) -> float:
    if not boxes:
        return 0.0
    cell_w, cell_h = W / grid, H / grid
    covered = 0
    for gy in range(grid):
        cy = (gy + 0.5) * cell_h
        for gx in range(grid):
            cx = (gx + 0.5) * cell_w
            for b in boxes:
                if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                    covered += 1
                    break
    return covered * cell_w * cell_h


def issues_from_checks(report: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
    mapping = {
        "syntax": "other",
        "no_forbidden": "other",
        "bbox": "size_mismatch",
        "count": "count_mismatch",
        "color": "color_mismatch",
        "pose": "pose_mismatch",
        "elements": "missing_object",
        "participants": "missing_object",
        "complete": "missing_object",
        "in_canvas": "position_mismatch",
        "sizes": "size_mismatch",
        "overlap": "position_mismatch",
        "relations": "position_mismatch",
    }
    out = []
    for f in report.get("failures", []):
        out.append({
            "type": mapping.get(f["name"], "other"),
            "severity": "major" if f["weight"] >= 2 else "minor",
            "description": f"{f['name']}: {f['detail']}",
            "fix_instruction": f"fix {f['name']} ({f['detail']})",
            "nodes": [node_id],
            "source": "check",
        })
    return out
