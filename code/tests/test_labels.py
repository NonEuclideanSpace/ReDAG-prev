"""Label tooling without any model: parsing of model labels, agreement, metrics on a tiny synthetic set."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from draftdag.labels import acceptance_metrics, agreement, eval_metrics, label_items_with_model


class Fake:
    def generate(self, jobs, max_tokens=80, json_mode=False):
        out = []
        for j in jobs:
            assert all(hasattr(im, "size") for im in j["images"])
            if "fixed" in j["text"]:
                out.append('{"fixed": "y", "damaged": "n", "duplicate": "n", "note": "ok"}')
            elif "missing" in j["text"]:
                out.append('garbage {"label": "wrong", "note": "x"}')
            else:
                out.append('{"label": "U", "note": "partly"}')
        return out


def test_labels():
    d = tempfile.mkdtemp()
    for n in ("a.png", "b.png"):
        Image.new("RGB", (800, 600), (200, 100, 50)).save(os.path.join(d, n))
    items = [dict(id="c1", kind="cond", image=os.path.join(d, "a.png"), question="Is there a cat?", region=[10, 10, 200, 200], ntype="entity", image_size=[800, 600]),
             dict(id="c2", kind="cond", image=os.path.join(d, "a.png"), question="Is the cat black?", region=None, ntype="attribute", image_size=[800, 600]),
             dict(id="e1", kind="edit", before=os.path.join(d, "a.png"), after=os.path.join(d, "b.png"), instruction="Add a cat.", accept=True, rules=dict(r1=True, r2=True)),
             dict(id="d1", kind="diag", image=os.path.join(d, "a.png"), question="Is there a cat?", desc="a cat")]
    rows = label_items_with_model(items, Fake())
    by = {r["id"]: r for r in rows}
    assert by["c1"]["label"] == "u" and by["e1"]["fixed"] == "y" and by["e1"]["duplicate"] == "n" and by["d1"]["label"] == "wrong"
    human = [dict(id="c1", label="u"), dict(id="c2", label="y"), dict(id="e1", fixed="y", damaged="y", duplicate="n")]
    ag = agreement(human, rows, "cond")
    assert ag["n"] == 2 and abs(ag["exact"] - 0.5) < 1e-9
    ag2 = agreement(human, rows, "edit")
    assert ag2["fixed"] == 1.0 and ag2["damaged"] == 0.0
    em = eval_metrics(items, human, {"c1": 0.2, "c2": 0.9})
    assert em is not None
    am = acceptance_metrics(items, human)
    assert am is not None


def test_effective_label_flips_inverted():
    from draftdag.labels import effective_label, eval_metrics

    items = [dict(id="a", kind="cond", ntype="count", inv=True), dict(id="b", kind="cond", ntype="entity", inv=False)]
    assert effective_label(items[0], "y") == "n" and effective_label(items[0], "n") == "y" and effective_label(items[0], "u") == "u"
    assert effective_label(items[1], "y") == "y"
    m = eval_metrics(items, [dict(id="a", label="y"), dict(id="b", label="y")], {"a": 0.2, "b": 0.9})
    assert m["overall"]["detect_n"] == 1.0 and m["overall"]["false_fail"] == 0.0


if __name__ == "__main__":
    test_labels()
    test_effective_label_flips_inverted()
    print("label tests ok")
