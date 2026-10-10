import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redag.dedupe import dedupe_children, dedupe_plan, lost_by_merge, merge_entities, para_key  # noqa: E402
from redag.schema import Plan  # noqa: E402


def plan():
    return dict(id="t", prompt="A kiosk. A boy sits on the floor. A cat on the mat. Birds. A bulb hanging from the ceiling.", canvas=[1024, 1024], nodes=[
        dict(id="bg", type="background", desc="a hall", question="Does the image show a hall?", box=[0, 0, 1024, 1024]),
        dict(id="kiosk", type="entity", class_="", desc="an information kiosk", question="Is there an information kiosk?", box=[100, 100, 300, 400], count=1, count_exact=True),
        dict(id="boy", type="entity", desc="a boy", question="Is there a boy?", box=[400, 100, 600, 500], count=1, count_exact=True),
        dict(id="cat", type="entity", desc="a cat", question="Is there a cat?", box=[600, 600, 700, 700], count=1, count_exact=True),
        dict(id="mat", type="entity", desc="a mat", question="Is there a mat?", box=[550, 650, 750, 750], count=1, count_exact=True),
        dict(id="kiosk_entity", type="entity", desc="an information kiosk", question="Is there an information kiosk?", box=[700, 100, 900, 400], count=1, count_exact=True),
        dict(id="birds", type="entity", desc="birds", question="Are there birds?", box=[0, 0, 300, 200], count=2, count_exact=False),
        dict(id="birds_entity", type="entity", desc="birds", question="Are there birds?", box=[0, 0, 200, 200], count=2, count_exact=False),
        dict(id="bulb", type="entity", desc="a bulb", question="Is there a bare light bulb?", box=[480, 0, 560, 100], count=1, count_exact=True),
        dict(id="bulb_bg", type="entity", desc="a bulb", question="Is there a bare light bulb hanging from the ceiling?", box=[480, 0, 544, 100], count=1, count_exact=True),
        dict(id="kiosk_lit", type="attribute", of="kiosk_entity", desc="the kiosk is lit", question="Is the information kiosk lit?"),
        dict(id="boy_sitting", type="state", of="boy", desc="the boy sits on the floor", question="Is the boy sitting on the floor?"),
        dict(id="boy_on_floor", type="state", of="boy", desc="the boy sits on the floor", question="Does the boy sit on the floor?"),
        dict(id="cat_on_mat", type="relation", of=["cat", "mat"], desc="the cat is on the mat", question="Is the cat on the mat?"),
        dict(id="cat_under_mat", type="relation", of=["cat", "mat"], desc="the cat is under the mat", question="Is the cat under the mat?"),
        dict(id="cat_looks_boy", type="action", of=["cat", "boy"], desc="the cat looks at the boy", question="Is the cat looking at the boy?"),
        dict(id="boy_looks_cat", type="action", of=["boy", "cat"], desc="the boy looks at the cat", question="Is the boy looking at the cat?"),
        dict(id="birds_fog", type="relation", of=["birds_entity", "birds"], desc="fog trails from the birds", question="Do wisps of fog trail from the birds?"),
        dict(id="no_dog", type="absent", desc="dog", question="Are there any dogs?"),
        dict(id="no_dog_2", type="absent", desc="dog", question="Are there dogs?"),
    ], edges=[dict({"from": "bg", "to": "kiosk_entity", "kind": "support"}), dict({"from": "bg", "to": "kiosk", "kind": "support"}), dict({"from": "kiosk_entity", "to": "boy", "kind": "support"})], meta={})


def fix(d):
    for n in d["nodes"]:
        if n["type"] == "entity":
            n["class"] = n["desc"].split()[-1]
            n.pop("class_", None)
    return d


def test_para_key():
    assert para_key("Is the boy sitting on the floor?") == para_key("Does the boy sit on the floor?")
    assert para_key("Is the cat on the mat?") != para_key("Is the cat under the mat?")
    assert para_key("Is the cat looking at the boy?") != para_key("Is the boy looking at the cat?")
    assert para_key("Is the pigeon perched on the truck's hood?") == para_key("Is the pigeon perched on the truck’s hood?")
    assert para_key("Is the light on?") != para_key("Is the light off?")


def test_children():
    d = fix(plan())
    dropped = dedupe_children(d)
    ids = [n["id"] for n in d["nodes"]]
    assert [x["id"] for x in dropped] == ["boy_on_floor", "no_dog_2"], dropped
    assert "boy_on_floor" not in ids and "cat_under_mat" in ids and "boy_looks_cat" in ids and "no_dog" in ids


def test_merge():
    d = fix(plan())
    done, refused = merge_entities(d, [["kiosk", "kiosk_entity"], ["birds", "birds_entity"], ["bulb", "bulb_bg"], ["boy", "cat"]])
    byid = {n["id"]: n for n in d["nodes"]}
    assert [(m["drop"], m["keep"]) for m in done] == [("kiosk_entity", "kiosk"), ("birds_entity", "birds"), ("bulb", "bulb_bg")], done
    assert byid["kiosk_lit"]["of"] == "kiosk"  # the child moved
    assert byid["birds_fog"]["type"] == "state" and byid["birds_fog"]["of"] == "birds"  # one participant left
    assert "bulb" not in byid and "bulb_bg" in byid  # the later twin covers the earlier one, so it is the one kept
    assert byid["bulb_bg"]["box"] == [480, 0, 560, 100]  # with the earlier twin's box
    assert len(refused) == 1 and refused[0]["ids"] == ["boy", "cat"] and "would lose" in refused[0]["why"]
    assert sorted((e["from"], e["to"]) for e in d["edges"]) == [("bg", "kiosk"), ("kiosk", "boy")]


def test_lost():
    d = fix(plan())
    byid = {n["id"]: n for n in d["nodes"]}
    assert lost_by_merge(d, byid["bulb_bg"], byid["bulb"]) == ["ceil", "hang"]
    assert lost_by_merge(d, byid["bulb"], byid["bulb_bg"]) == []


def test_plan():
    d = fix(plan())
    before = json.dumps(d, sort_keys=True)
    out, info = dedupe_plan(d, [["kiosk", "kiosk_entity"], ["birds", "birds_entity"], ["bulb", "bulb_bg"]])
    assert json.dumps(d, sort_keys=True) == before  # the input is untouched
    assert out is not d and info["error"] is None
    assert info["n_before"] == 20 and info["n_after"] == 15, info
    errs, _w = Plan(out).validate(min_side=48, max_nodes=128)
    assert not errs, errs
    out2, info2 = dedupe_plan(out, [["kiosk", "kiosk_entity"]])
    assert out2 is out and not info2["merged"] and not info2["dropped"] and len(info2["refused"]) == 1


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("OK test_dedupe")
