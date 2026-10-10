import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

from .views import to_rgb


def _grid_slice(box, cell, gh, gw):
    x0 = int(np.clip(np.floor(box[0] / cell), 0, gw - 1))
    y0 = int(np.clip(np.floor(box[1] / cell), 0, gh - 1))
    x1 = int(np.clip(np.ceil(box[2] / cell), x0 + 1, gw))
    y1 = int(np.clip(np.ceil(box[3] / cell), y0 + 1, gh))
    return slice(y0, y1), slice(x0, x1)


def tone_fit(a, b, cell=8, keep=0.5):
    gh, gw = a.shape[0] // cell, a.shape[1] // cell
    ca = a[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell, 3).mean((1, 3)).reshape(-1, 3)
    cb = b[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell, 3).mean((1, 3)).reshape(-1, 3)
    n = max(16, int(keep * len(ca)))
    idx = np.argsort(np.abs(cb - ca).mean(-1))[:n]
    gain, bias = np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    for _ in range(2):
        for c in range(3):
            x, y = cb[idx, c], ca[idx, c]
            vx = float(x.var())
            g = float(np.clip(((x - x.mean()) * (y - y.mean())).mean() / vx, 0.8, 1.25)) if vx > 1e-5 else 1.0
            gain[c], bias[c] = g, float(np.clip(y.mean() - g * x.mean(), -0.15, 0.15))
        idx = np.argsort(np.abs(cb * gain + bias - ca).mean(-1))[:n]
    return gain, bias, float(np.abs(cb[idx] * gain + bias - ca[idx]).mean())


def change_grid(canvas, cand, cell=8, thr=0.05, tone=True, res_max=0.04):
    base = to_rgb(canvas)
    W, H = base.size
    a = np.asarray(base, dtype=np.float32) / 255.0
    b = np.asarray(to_rgb(cand).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    gain, bias = np.ones(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    if tone:
        g2, b2, res = tone_fit(a, b, cell)
        if res <= res_max:
            gain, bias = g2, b2
    b2 = np.clip(b * gain + bias, 0.0, 1.0)
    gh, gw = H // cell, W // cell
    m0 = np.abs(a - b2).mean(-1)[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean((1, 3)) > thr
    return a, b2, m0, (gh, gw)


def mask_boxes(m, cell):
    """Bounding boxes (pixel coords) of connected components of a grid mask."""
    if not m.any():
        return []
    lab, n = ndi.label(m, structure=np.ones((3, 3), dtype=bool))
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        out.append([float(xs.min() * cell), float(ys.min() * cell), float((xs.max() + 1) * cell), float((ys.max() + 1) * cell)])
    return out


def region_commit(canvas, cand, scope=(), protect=(), cell=8, thr=0.05, grow=2, feather=6, speck=3, rel=0.3, tone=True):
    """Paste only the changed region of a candidate back onto the canvas.

    scope: boxes (pixels) where the edit was asked; changed blobs near them are kept, far small blobs are dropped.
    protect: boxes (pixels) that must not change; they are removed from the commit mask.
    Returns (composite PIL, info) where info['grid'] is the commit mask at cell resolution and info['boxes'] its blobs.
    """
    a, b2, m0, (gh, gw) = change_grid(canvas, cand, cell, thr, tone)
    W, H = int(gw * cell), int(gh * cell)
    m = np.zeros_like(m0)
    kept = dropped = 0
    if m0.any():
        lab, n = ndi.label(m0, structure=np.ones((3, 3), dtype=bool))
        sizes = np.asarray(ndi.sum(m0, lab, range(1, n + 1)))
        inside = np.zeros_like(m0)
        for bx in scope:
            if bx is not None:
                ys, xs = _grid_slice(bx, cell, gh, gw)
                inside[ys, xs] = True
        near = ndi.binary_dilation(inside, iterations=int(grow)) if inside.any() else inside
        hit = np.asarray(ndi.sum(near, lab, range(1, n + 1))) > 0
        big = sizes[hit].max() if hit.any() else sizes.max()
        ok = (sizes >= speck) & (hit | (sizes >= rel * big))
        m = np.isin(lab, 1 + np.nonzero(ok)[0])
        kept, dropped = int(ok.sum()), int((~ok).sum())
    share_raw = float(m.mean())
    if m.any() and grow > 0:
        m = ndi.binary_dilation(m, iterations=int(grow))
        m = ndi.binary_fill_holes(m)
    for bx in protect:
        if bx is not None:
            ys, xs = _grid_slice(bx, cell, gh, gw)
            m[ys, xs] = False
    mask = Image.fromarray((m * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    if feather > 0 and m.any():
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    al = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    for bx in protect:
        if bx is not None:
            x0, y0 = max(0, int(bx[0])), max(0, int(bx[1]))
            x1, y1 = min(W, int(np.ceil(bx[2]))), min(H, int(np.ceil(bx[3])))
            if x1 > x0 and y1 > y0:
                al[y0:y1, x0:x1] = 0.0
    aa, bb = a[:H, :W], b2[:H, :W]
    out = np.where(al > 0, al * bb + (1.0 - al) * aa, aa)
    comp = Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))
    if comp.size != to_rgb(canvas).size:
        full = to_rgb(canvas).copy()
        full.paste(comp, (0, 0))
        comp = full
    info = dict(share=float(m.mean()), share_raw=share_raw, raw=float(m0.mean()), kept=kept, dropped=dropped, boxes=mask_boxes(m, cell), cell=cell, grid=m)
    return comp, info


def grids_overlap(ga, gb, dilate=1):
    a = ndi.binary_dilation(ga, iterations=dilate) if dilate > 0 else ga
    b = ndi.binary_dilation(gb, iterations=dilate) if dilate > 0 else gb
    return bool((a & b).any())


def merge_commits(canvas, committed):
    """committed: list of (composite PIL, grid mask). Masks must be pairwise disjoint (checked by caller).
    Pixels inside each mask come from that composite, the rest from the canvas."""
    base = np.asarray(to_rgb(canvas), dtype=np.float32)
    H, W = base.shape[:2]
    out = base.copy()
    for comp, grid in committed:
        cell = H // grid.shape[0]
        mask = Image.fromarray((grid * 255).astype(np.uint8)).resize((grid.shape[1] * cell, grid.shape[0] * cell), Image.NEAREST)
        mask = mask.filter(ImageFilter.GaussianBlur(6))
        al = np.zeros((H, W, 1), dtype=np.float32)
        am = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
        al[:am.shape[0], :am.shape[1]] = am
        c = np.asarray(to_rgb(comp).resize((W, H), Image.BILINEAR), dtype=np.float32)
        out = al * c + (1.0 - al) * out
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def touched_fraction(grid, box, cell):
    gh, gw = grid.shape
    ys, xs = _grid_slice(box, cell, gh, gw)
    sub = grid[ys, xs]
    return float(sub.mean()) if sub.size else 0.0
