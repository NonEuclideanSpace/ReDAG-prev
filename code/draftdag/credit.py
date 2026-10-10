"""Region credit for a group of G candidates that solved the same target from the same canvas.

Each latent token belongs to exactly one region: the target's change region, one region per other established
entity, or the background. A region's reward is the sum of the scores of the conditions anchored on it; the global
reward adds the whole-image conditions. Advantages are standardised within the group per region and mixed with the
global advantage; region weights follow area and the reliability-weighted number of conditions on the region.
"""
import numpy as np

from .schema import CHILD_TYPES, MULTI_TYPES


def box_to_grid(box, W, H, gh, gw):
    m = np.zeros((gh, gw), dtype=bool)
    if box is None:
        return m
    x0 = int(np.clip(np.floor(box[0] * gw / W), 0, gw - 1))
    y0 = int(np.clip(np.floor(box[1] * gh / H), 0, gh - 1))
    x1 = int(np.clip(np.ceil(box[2] * gw / W), x0 + 1, gw))
    y1 = int(np.clip(np.ceil(box[3] * gh / H), y0 + 1, gh))
    m[y0:y1, x0:x1] = True
    return m


def partition(plan, canvas_boxes, target_nodes, target_masks, gh, gw):
    """Per candidate: region id map (gh, gw) and region names. id 0 = target change region, 1..k = other established
    entities (smaller boxes take precedence where boxes overlap), last = background."""
    others = [v for v in plan.entities() if v not in target_nodes and v in canvas_boxes]
    others.sort(key=lambda v: (canvas_boxes[v][2] - canvas_boxes[v][0]) * (canvas_boxes[v][3] - canvas_boxes[v][1]))
    names = ["target"] + others + ["bg"]
    out = []
    for tm in target_masks:
        rid = np.full((gh, gw), len(names) - 1, dtype=np.int32)
        for j, v in reversed(list(enumerate(others, start=1))):
            rid[box_to_grid(canvas_boxes[v], plan.W, plan.H, gh, gw)] = j
        rid[tm] = 0
        out.append(rid)
    return out, names


def anchors(plan, conds):
    """Condition -> list of (region name, share). Entity, attribute, state -> that entity; action, relation -> each
    participant equally; others -> global."""
    out = {}
    for c in conds:
        t = c.ntype
        if t == "entity":
            out[c.cid] = [(c.node, 1.0)]
        elif t in CHILD_TYPES:
            out[c.cid] = [(c.participants[0], 1.0)]
        elif t in MULTI_TYPES:
            k = len(c.participants)
            out[c.cid] = [(u, 1.0 / k) for u in c.participants]
        else:
            out[c.cid] = [("global", 1.0)]
    return out


def standardize(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    sd = float(x.std())
    return (x - x.mean()) / (sd + eps), sd


def region_credit(plan, conds, cols, target_nodes, names, rid_maps, rel=None, alpha=0.5, beta=0.7, a0=1e-2, min_std=0.02, lam_q=0.5, adv_clip=5.0):
    """cols: G verified Columns of the candidates. Returns per-token advantages [G, gh*gw], region weights [G, R],
    region advantages [G, R], global advantage [G] and the region rewards [G, R]."""
    G = len(cols)
    R = len(names)
    anc = anchors(plan, conds)
    rel = rel or {}
    name_idx = {n: i for i, n in enumerate(names)}
    Rw = np.zeros((G, R))
    psi_sum = np.zeros(R)
    glob = np.zeros(G)
    for c in conds:
        r_c = float(rel.get(c.cid, rel.get(c.ntype, 1.0)))
        for i, col in enumerate(cols):
            st = col.status(c.cid)
            s = col.score(c.cid, 0.0) if st in ("pass", "fail") else 0.0
            if c.kind == "quality":
                glob[i] += lam_q * c.psi * s / 2.0
                continue
            for (u, share) in anc[c.cid]:
                if u == "global":
                    glob[i] += c.psi * s * share
                else:
                    j = 0 if u in target_nodes else name_idx.get(u, R - 1)
                    Rw[i, j] += c.psi * s * share
                    if i == 0:
                        psi_sum[j] += c.psi * share * r_c
    phi = Rw.sum(1) + glob
    adv_g, sd_g = standardize(phi)
    if sd_g < 1e-8:
        adv_g = np.zeros(G)
    adv = np.zeros((G, R))
    fallback = np.zeros(R, dtype=bool)
    for j in range(R):
        a, sd = standardize(Rw[:, j])
        if psi_sum[j] <= 0 or sd < min_std:
            adv[:, j] = adv_g
            fallback[j] = True
        else:
            adv[:, j] = (1 - alpha) * adv_g + alpha * a
    adv = np.clip(adv, -adv_clip, adv_clip)
    adv_g = np.clip(adv_g, -adv_clip, adv_clip)
    L = rid_maps[0].size
    area = np.zeros((G, R))
    for i in range(G):
        for j in range(R):
            area[i, j] = float((rid_maps[i] == j).sum()) / L
    w = (area + a0) ** beta * np.maximum(psi_sum, 1e-6)[None]
    w = np.where(area > 0, w, 0.0)
    wsum = w.sum(1, keepdims=True)
    w = np.where(wsum > 0, w / np.maximum(wsum, 1e-12), 1.0 / R)
    tok = np.zeros((G, L))
    for i in range(G):
        tok[i] = adv[i][rid_maps[i].reshape(-1)]
    return dict(tok_adv=tok, w=w, adv=adv, adv_g=adv_g, Rw=Rw, phi=phi, fallback=fallback, area=area, names=names, psi_sum=psi_sum)
