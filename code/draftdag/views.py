import base64
import hashlib
import io
import os

from PIL import Image, ImageDraw

from .schema import dilate_box


def load_image(x):
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, str) and not os.path.isabs(x) and not os.path.exists(x):
        from .paths import P0

        x = os.path.join(P0, x)  # label items store image paths relative to the project root
    im = Image.open(x)
    im.load()
    return im


def carry(src, out):
    """Keep side-channel tags (used by the mock environment) on derived images."""
    for k in ("truth", "truth_pair"):
        if k in getattr(src, "info", {}):
            out.info[k] = src.info[k]
    return out


def save_png(im, path):
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    for k in ("truth",):
        if k in getattr(im, "info", {}):
            meta.add_text(k, im.info[k])
    to_rgb(im).save(path, format="PNG", pnginfo=meta, compress_level=3)
    return path


def to_rgb(im):
    return carry(im, im.convert("RGB")) if im.mode != "RGB" else im


def resize_min_side(im, side):
    w, h = im.size
    m = min(w, h)
    if m == side:
        return im
    s = side / float(m)
    return carry(im, im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS))


def fit_side(im, lo, hi):
    w, h = im.size
    m = min(w, h)
    if m < lo:
        return resize_min_side(im, lo)
    if m > hi:
        return resize_min_side(im, hi)
    return im


def crop_box(im, box, margin, min_side=448, max_side=1024, square=False):
    W, H = im.size
    x0, y0, x1, y1 = dilate_box([float(v) for v in box], margin, W, H)
    if square:
        s = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1 = cx - s / 2, cx + s / 2
        y0, y1 = cy - s / 2, cy + s / 2
        dx = min(0, x0) * -1 if x0 < 0 else (W - x1 if x1 > W else 0)
        dy = min(0, y0) * -1 if y0 < 0 else (H - y1 if y1 > H else 0)
        x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    win = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    if win[2] - win[0] < 8 or win[3] - win[1] < 8:
        win = (0, 0, W, H)
    out = carry(im, to_rgb(im).crop(win))
    return fit_side(out, min_side, max_side), win


def boxed_full(im, box, side=896, width=6):
    im2 = carry(im, to_rgb(im).copy())
    d = ImageDraw.Draw(im2)
    x0, y0, x1, y1 = [float(v) for v in box]
    d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=width)
    return resize_min_side(im2, side)


def pair_image(before, after, box=None, side=512, gap=16, ctx=2.2, min_frac=0.45):
    before, after = to_rgb(before), to_rgb(after)
    W, H = before.size
    if after.size != before.size:
        after = after.resize(before.size, Image.LANCZOS)
    if box is None:
        a, b = before, after
    else:
        x0, y0, x1, y1 = [float(q) for q in box]
        sd = min(max((x1 - x0) * ctx, (y1 - y0) * ctx, min_frac * min(W, H)), float(min(W, H)))
        wx = int(round(min(max(0.0, (x0 + x1) / 2 - sd / 2), W - sd)))
        wy = int(round(min(max(0.0, (y0 + y1) / 2 - sd / 2), H - sd)))
        win = (wx, wy, int(wx + sd), int(wy + sd))
        a, b = before.crop(win), after.crop(win)
    a, b = a.resize((side, side), Image.LANCZOS), b.resize((side, side), Image.LANCZOS)
    c = Image.new("RGB", (2 * side + gap, side), (255, 255, 255))
    c.paste(a, (0, 0))
    c.paste(b, (side + gap, 0))
    if "truth" in before.info or "truth" in after.info:
        c.info["truth_pair"] = (before.info.get("truth"), after.info.get("truth"))
    return c


def png_bytes(im):
    buf = io.BytesIO()
    to_rgb(im).save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def data_url(im):
    return "data:image/png;base64," + base64.b64encode(png_bytes(im)).decode("ascii")


def image_key(im):
    rgb = to_rgb(im)
    return hashlib.sha1(rgb.tobytes()).hexdigest()[:20] + f"-{rgb.size[0]}x{rgb.size[1]}"


def make_view(im, kind, box=None, full_side=896, crop_margin=0.25, crop_min=448, crop_max=1024):
    im = load_image(im)
    if kind == "full":
        return resize_min_side(to_rgb(im), full_side), None
    if kind == "native":
        return to_rgb(im), None
    if kind == "crop":
        return crop_box(im, box, crop_margin, crop_min, crop_max)
    if kind == "boxed":
        return boxed_full(im, box, full_side), None
    raise ValueError(f"unknown view {kind}")
