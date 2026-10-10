import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from draftdag.commit import grids_overlap, merge_commits, region_commit, touched_fraction


def _canvas(seed=0, size=512):
    rng = np.random.default_rng(seed)
    a = rng.integers(40, 220, size=(size, size, 3), dtype=np.uint8)
    a = np.asarray(Image.fromarray(a).filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(3)))
    return Image.fromarray(a)


def test_region_commit_keeps_outside_and_takes_inside():
    canvas = _canvas(1)
    a = np.asarray(canvas).copy()
    b = a.copy()
    b[100:200, 300:420] = [255, 0, 0]
    cand = Image.fromarray(b)
    comp, info = region_commit(canvas, cand, scope=[[300, 100, 420, 200]])
    c = np.asarray(comp).astype(int)
    assert np.abs(c[120:180, 320:400] - np.array([255, 0, 0])).max() < 3
    assert np.abs(c[:60, :] - a[:60, :].astype(int)).max() == 0
    assert np.abs(c[:, :250] - a[:, :250].astype(int)).max() == 0
    assert len(info["boxes"]) == 1
    bx = info["boxes"][0]
    assert bx[0] <= 300 and bx[2] >= 420 and bx[1] <= 100 and bx[3] >= 200
    assert 0.03 < info["share"] < 0.12


def test_tone_shift_is_not_committed():
    canvas = _canvas(2)
    a = np.asarray(canvas).astype(np.float32)
    b = np.clip(a * 1.06 + 4, 0, 255)
    b[200:260, 60:160] = [0, 255, 0]
    cand = Image.fromarray(b.astype(np.uint8))
    comp, info = region_commit(canvas, cand, scope=[[60, 200, 160, 260]])
    assert info["raw"] < 0.15, info["raw"]
    c = np.asarray(comp).astype(int)
    assert np.abs(c[400:, 300:] - a[400:, 300:].astype(int)).max() == 0
    patch = c[215:245, 80:140]
    assert patch[..., 1].min() > 200 and patch[..., 0].max() < 30 and patch[..., 2].max() < 30


def test_far_specks_dropped_and_protect_respected():
    canvas = _canvas(3)
    a = np.asarray(canvas).copy()
    b = a.copy()
    b[50:150, 50:150] = [0, 0, 255]
    b[400:412, 400:412] = [255, 255, 0]
    cand = Image.fromarray(b)
    comp, info = region_commit(canvas, cand, scope=[[50, 50, 150, 150]], protect=[[90, 90, 150, 150]])
    c = np.asarray(comp).astype(int)
    assert np.abs(c[400:412, 400:412] - a[400:412, 400:412].astype(int)).max() == 0
    assert np.abs(c[100:148, 100:148] - a[100:148, 100:148].astype(int)).max() == 0
    assert np.abs(c[60:80, 60:80] - np.array([0, 0, 255])).max() < 3


def test_merge_disjoint_commits():
    canvas = _canvas(4)
    a = np.asarray(canvas).copy()
    b1, b2 = a.copy(), a.copy()
    b1[40:120, 40:120] = [255, 0, 0]
    b2[300:400, 300:420] = [0, 0, 255]
    c1, i1 = region_commit(canvas, Image.fromarray(b1), scope=[[40, 40, 120, 120]])
    c2, i2 = region_commit(canvas, Image.fromarray(b2), scope=[[300, 300, 420, 400]])
    assert not grids_overlap(i1["grid"], i2["grid"], 1)
    m = np.asarray(merge_commits(canvas, [(c1, i1["grid"]), (c2, i2["grid"])])).astype(int)
    assert np.abs(m[60:100, 60:100] - np.array([255, 0, 0])).max() < 3
    assert np.abs(m[320:380, 320:400] - np.array([0, 0, 255])).max() < 3
    assert np.abs(m[200:260, 200:260] - a[200:260, 200:260].astype(int)).max() == 0
    assert touched_fraction(i1["grid"], [40, 40, 120, 120], i1["cell"]) > 0.9
    assert touched_fraction(i1["grid"], [300, 300, 420, 400], i1["cell"]) == 0.0


if __name__ == "__main__":
    test_region_commit_keeps_outside_and_takes_inside()
    test_tone_shift_is_not_committed()
    test_far_specks_dropped_and_protect_respected()
    test_merge_disjoint_commits()
    print("commit tests ok")
