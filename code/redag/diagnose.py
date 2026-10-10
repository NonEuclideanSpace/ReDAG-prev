import json
import re

from .conditions import region_of
from .schema import CHILD_TYPES, MULTI_TYPES, NUM, box_iou, union_box
from .views import load_image, resize_min_side, to_rgb

KEEP_TAIL = " Keep everything else in the image unchanged."
JSON_TAIL = ' Reply with one JSON object and nothing else: {{"instruction": "<one imperative sentence>", "bbox_2d": [x1, y1, x2, y2]}}.'

P_ADD = ("This image must contain: {desc}. It is not in the image yet. Write ONE imperative sentence for an image editing model that adds it at a realistic size, "
         "placed relative to things that are visible in the image, without covering important people or objects, and changes nothing else. "
         "The bounding box is the free place where it should go." + JSON_TAIL)
P_FIX_ENTITY = ("This image should contain: {desc}. An element of this kind is present but does not match the description: it looks different, has a different pose, or is in another place. "
                "Find that element. Write ONE imperative sentence for an image editing model that changes that existing element so that the description becomes true; "
                "name it by what is visible of it; do not add a second one; change nothing else. The bounding box is that element." + JSON_TAIL)
P_FIX_CHILD = ("The image contains {parent}. This must also be true about it: {desc}. A checker asked \"{question}\" and the answer was no. "
               "Write ONE imperative sentence for an image editing model that changes this element so that the statement becomes true; name the element by what is visible of it; "
               "keep it where it is; do not add another one; change nothing else. The bounding box is the element." + JSON_TAIL)
P_FIX_MULTI = ("The image contains {parts}. This must be true: {desc}. A checker asked \"{question}\" and the answer was no. "
               "Decide which participant has to change (pose, orientation, hands, position) and write ONE imperative sentence for an image editing model that makes the statement true "
               "while changing as little as possible; name the participants by what is visible of them; do not add anyone or anything; change nothing else. "
               "The bounding box is the area that has to change." + JSON_TAIL)
P_REMOVE_COUNT = ("This image should show exactly {need}, but it shows more. The ones that belong in the picture are: {keep}. Find the extra one that does not belong. "
                  "Write ONE imperative sentence for an image editing model that removes the extra one, named by what is visible of it and where it is, and fills its place with the surrounding background, changing nothing else. "
                  "The bounding box is the extra element." + JSON_TAIL)
P_REMOVE_ABSENT = ("This image must not contain {desc}, but it does. Write ONE imperative sentence for an image editing model that removes it, named by what is visible of it and where it is, "
                   "and fills its place with the surrounding background, changing nothing else. The bounding box is the element to remove." + JSON_TAIL)
P_SPLIT = ("The image should contain two different individuals: (1) {a}; (2) {b}. Right now they appear as one individual, or only one of them is present. "
           "Write ONE imperative sentence for an image editing model that makes both present as two separate individuals: keep the existing one where it is and add or separate the other right next to it, changing nothing else. "
           "The bounding box is the area where the second individual should be." + JSON_TAIL)


def diagnose(plan, conds, col, vlm=None, image=None):
    """Type every confirmed failure by the shape of the table. Returns a list of target dicts."""
    out = []
    by_cid = {c.cid: c for c in conds}
    for c in conds:
        if not col.fails(c.cid):
            continue
        t = c.ntype
        key = c.cid
        if t == "background":
            out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="skip", reason="background failed; whole-image redraw is not an edit", participants=[], region=None))
            continue
        if t == "entity":
            cls = plan.nodes[c.node].get("class")
            n_cls = plan.class_count(cls) if cls else 1
            present_all = False
            if vlm is not None and image is not None and cls:
                ccid = "c:cnt_" + re.sub(r"[^a-z0-9]+", "_", cls.lower()).strip("_")
                cc = by_cid.get(ccid)
                q = cc.question_alt if cc is not None and cc.question_alt else f"Are there at least {NUM.get(n_cls, str(n_cls))} {cls if n_cls == 1 else cls + 's'} in this image?"
                im = to_rgb(load_image(image))
                p, _ = vlm.yesno_one(resize_min_side(im, 1024) if min(im.size) > 1024 else im, q, "a", False)
                present_all = p >= 0.5
            if present_all:
                out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="fix", reason="all instances of the class present; this one does not match", participants=[c.node], region=region_of(plan, c, col.boxes)))
            else:
                out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="add", reason="entity absent after two-stage check", participants=[c.node], region=region_of(plan, c, col.boxes)))
            continue
        if t in CHILD_TYPES:
            out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="fix", reason="parent entity present; detail wrong", participants=list(c.participants), region=region_of(plan, c, col.boxes)))
            continue
        if t in MULTI_TYPES:
            side = None
            for u in c.participants:
                for w in plan.children(u):
                    if plan.typ(w) in CHILD_TYPES and col.fails("n:" + w):
                        side = u
                        break
                if side:
                    break
            out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="fix", reason="participants present; interaction wrong", participants=list(c.participants), region=region_of(plan, c, col.boxes), side=side))
            continue
        if t == "absent":
            out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="remove", reason="forbidden element present", participants=[], region=None))
            continue
        if c.kind == "count":
            out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="remove", reason="more instances than planned", participants=[u for u in plan.entities() if plan.nodes[u].get("class") == c.cls], region=None, cls=c.cls))
            continue
        if c.kind == "pair":
            out.append(dict(key=key, cid=c.cid, node=c.node, ntype=t, op="split", reason="two planned individuals appear as one", participants=list(c.participants), region=region_of(plan, c, col.boxes)))
            continue
    return out


def _clean_instruction(text):
    t = (text or "").strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    if not 4 <= len(t.split()) <= 70:
        return None
    if re.search(r"\b\d{2,4}\s*,\s*\d{2,4}\b|bbox|bounding box|coordinate", t, re.I):
        return None
    if t[-1] not in ".!":
        t += "."
    return t


def parse_instruction(vlm, text, sent_size, orig_size):
    instr, box = None, None
    try:
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group(0)) if m else {}
        instr = _clean_instruction(str(d.get("instruction", "")))
    except Exception:
        m = re.search(r'"instruction"\s*:\s*"(.*?)"\s*[,}]', text, re.S)
        instr = _clean_instruction(m.group(1)) if m else None
    box = vlm.parse_bbox(text, sent_size, orig_size, min_side=12)
    return instr, box


def _desc(plan, v):
    return plan.nodes[v]["desc"].strip().rstrip(".")


def write_instruction(vlm, plan, image, target):
    """Ask the evaluator to turn a typed diagnosis into one editing instruction plus the region to change."""
    op, t = target["op"], target["ntype"]
    v = target["node"]
    n = plan.nodes[v]
    if op == "add":
        prompt = P_ADD.format(desc=_desc(plan, v))
    elif op == "fix" and t == "entity":
        prompt = P_FIX_ENTITY.format(desc=_desc(plan, v))
    elif op == "fix" and t in CHILD_TYPES:
        prompt = P_FIX_CHILD.format(parent=_desc(plan, target["participants"][0]), desc=_desc(plan, v), question=n["question"].strip())
    elif op == "fix" and t in MULTI_TYPES:
        parts = " and ".join(f"({i + 1}) {_desc(plan, u)}" for i, u in enumerate(target["participants"]))
        prompt = P_FIX_MULTI.format(parts=parts, desc=_desc(plan, v), question=n["question"].strip())
        if target.get("side"):
            prompt = prompt.replace("Decide which participant has to change", f"The participant that most likely has to change is {_desc(plan, target['side'])}. Decide which participant has to change")
    elif op == "remove" and t == "count":
        keep = "; ".join(_desc(plan, u) for u in target["participants"])
        prompt = P_REMOVE_COUNT.format(need=n["desc"], keep=keep)
    elif op == "remove" and t == "absent":
        prompt = P_REMOVE_ABSENT.format(desc=_desc(plan, v))
    elif op == "split":
        a, b = target["participants"]
        prompt = P_SPLIT.format(a=_desc(plan, a), b=_desc(plan, b))
    else:
        return None
    im = to_rgb(load_image(image))
    sent = resize_min_side(im, vlm.max_side) if min(im.size) > vlm.max_side else im
    txt = vlm.generate_one([sent], prompt, max_tokens=200)
    instr, box = parse_instruction(vlm, txt, sent.size, im.size)
    if instr is None:
        return None
    region = target.get("region")
    if op in ("add", "remove") or region is None:
        region = box or region
    elif box is not None and (region is None or box_iou(box, region) < 0.3):
        region = union_box([box, region]) if op in ("fix",) and t in MULTI_TYPES else box
    if region is None:
        region = plan.planned_box(v) if t != "count" and t != "absent" else None
    return dict(target, instruction=instr + KEEP_TAIL, raw=txt[:400], bbox=box, region=region)
