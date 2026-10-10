"""Planner-side scripts without any API: set import, plan summary on existing plans, audit parsing with a fake chat."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from fixtures import lab_plan  # noqa: E402

from draftdag.planner import _grow_box, atomize_pass, audit_plan, facts_pass  # noqa: E402
from draftdag.schema import strip_generated  # noqa: E402


def test_planner_offline():
    root = tempfile.mkdtemp()
    try:
        env = dict(os.environ, P0=root, PYTHONPATH=CODE)
        src = os.path.join(root, "lb.jsonl")
        with open(src, "w") as f:
            for i in range(5):
                f.write(json.dumps(dict(instruction=f"scene {i} " + "word " * 400, label=[{"Object": "x"}])) + "\n")
        out = subprocess.run([sys.executable, os.path.join(CODE, "scripts", "make_sets.py"), "longbench", "--src", src], env=env, cwd=CODE, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        rows = [json.loads(l) for l in open(os.path.join(root, "data", "sets", "lb_t2i.jsonl"))]
        assert len(rows) == 5 and rows[0]["id"] == "lb_0000" and rows[0]["split"] in ("dev", "test")
        assert json.load(open(os.path.join(root, "data", "checklists", "lb_t2i.json")))["lb_0000"] == [{"Object": "x"}]
        d = strip_generated(lab_plan()).to_dict()
        d["id"], d["prompt"] = "lb_0000", rows[0]["prompt"]
        pdir = os.path.join(root, "data", "plans", "lb_t2i", "stub")
        os.makedirs(pdir)
        json.dump(d, open(os.path.join(pdir, "lb_0000.json"), "w"))
        out = subprocess.run([sys.executable, os.path.join(CODE, "scripts", "plan.py"), "--set", "lb_t2i", "--tag", "stub", "--limit", "1"], env=env, cwd=CODE, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        summ = json.load(open(os.path.join(pdir, "_summary.json")))
        assert summ["ok"] == 1 and sum(summ["cond_tiers"].values()) == 1 and summ["mean_conds"] > 10, summ

        def fake(messages, model=None, tag="", max_tokens=0, **kw):
            if tag.startswith("audit_beyond"):
                return json.dumps({"a_pipette": {"beyond": True, "multi": False, "reason": "right hand"}, "handover": {"beyond": False, "multi": True, "reason": "two things"}})
            if tag.startswith("audit_facts"):
                return json.dumps({"facts": ["a researcher wearing goggles", "a window"]})
            return json.dumps({"F1": 2, "F2": None})

        a = audit_plan(lab_plan(), chat=fake)
        assert a["n_beyond"] == 1 and a["n_multi"] == 1 and a["missing"] == ["a window"] and a["n_questions"] > 5
        test_facts_pass()
        test_atomize_pass()
        test_atomize_retype()
        test_atomize_alias()
        test_atomize_ship()
        test_atomize_lift_repeats()
        test_atomize_dependency_message()
        test_atomize_retry_and_cycle()
        test_atomize_alias_same_kind()
        test_atomize_cleanup_parts()
        test_atomize_participants_and_relative()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_facts_pass():
    """facts_pass: facts listed without the plan, matched to requirements, nodes written only for the unmatched ones;
    invalid additions are dropped, non-still facts are filtered before matching."""
    calls = []

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        calls.append(tag)
        if tag.startswith("plan_facts"):
            return json.dumps({"facts": ["a researcher wearing goggles", "the beaker is bubbling", "a bench", "the beaker stands on the bench", "the lights flicker slowly"]})
        if tag.startswith("plan_match"):
            return json.dumps({"F1": "R2", "F2": 7, "F3": None, "F4": None})
        assert tag.startswith("cover_facts") and "- a bench" in messages[-1]["content"] and "flicker" not in messages[-1]["content"]
        return json.dumps({"nodes": [
            {"id": "bench", "type": "entity", "class": "bench", "plural": "benches", "count": 1, "count_exact": True, "desc": "a laboratory bench", "question": "Is there a laboratory bench?", "box": [0, 600, 1024, 900], "evidence": "a beaker bubbles on the bench"},
            {"id": "beaker_on_bench", "type": "relation", "of": ["beaker", "bench"], "desc": "the beaker stands on the bench", "question": "Is the beaker standing on the bench?", "evidence": "a beaker bubbles on the bench"},
            {"id": "bad_attr", "type": "attribute", "of": "nobody", "desc": "x", "question": "Is x?", "evidence": "a beaker bubbles on the bench"},
            {"id": "sup_beside_beaker", "type": "relation", "of": ["supervisor", "beaker"], "desc": "the supervisor is beside the beaker", "question": "Is the supervisor beside the beaker?", "evidence": "a beaker bubbles on the bench"},
            {"id": "made_up", "type": "state", "of": "beaker", "desc": "the beaker is empty", "question": "Is the beaker empty?", "evidence": "the beaker is completely empty now"},
        ], "edges": [{"from": "bench", "to": "beaker", "kind": "support"}]})

    d = strip_generated(lab_plan()).to_dict()
    d["prompt"] = "In a chemistry lab, a researcher wearing goggles holds a pipette. A beaker bubbles on the bench. There is no window."
    out, info = facts_pass(d, chat=fake)
    assert calls == ["plan_facts:" + d["id"]] * 2 + ["plan_match:" + d["id"], "cover_facts:" + d["id"]], calls
    assert info["n_facts_raw"] == 5 and info["n_facts"] == 4 and info["n_uncovered"] == 2, info
    assert info["proposed"] == 5 and [u["id"] for u in info["ungrounded"]] == ["bad_attr", "sup_beside_beaker", "made_up"], info
    assert info["added"] == 2 and info["dropped"] == 0, info
    ids = {n["id"] for n in out["nodes"]}
    assert "bench" in ids and "beaker_on_bench" in ids and "bad_attr" not in ids and "sup_beside_beaker" not in ids
    assert all("evidence" not in n for n in out["nodes"])
    assert out["meta"]["facts_pass"]["added"] == 2 and len(d["nodes"]) + 2 == len(out["nodes"])

    def nothing(messages, model=None, tag="", max_tokens=0, **kw):
        if tag.startswith("plan_facts"):
            return json.dumps({"facts": ["a researcher wearing goggles"]})
        return json.dumps({"F1": "R2"})

    out2, info2 = facts_pass(d, chat=nothing)
    assert out2 is d and info2["n_uncovered"] == 0 and info2["added"] == 0
    assert _grow_box([500, 500, 520, 520], 1024, 1024, 48) == [486, 486, 534, 534] and _grow_box([0, 0, 10, 300], 1024, 1024, 48) == [0, 0, 48, 300]


def test_atomize_pass():
    """atomize_pass on the lab plan: a group that would lose words of the old question, a group with an ungrounded new
    node and a group whose new entity has no box are not applied in the first round and go back to the planner once with
    the reason; the repaired groups are applied; only a comparison is added; a deletion needs a sound, smell,
    temperature or motion word in the node's own question and nothing else lost; a stated count is set."""
    d = strip_generated(lab_plan()).to_dict()
    d["prompt"] = ("In a chemistry lab, a researcher wearing safety goggles holds a pipette in the right hand. A supervisor in a white coat "
                   "stands next to a computer screen on the right and points at the screen while talking to a researcher holding a notebook. "
                   "A glass beaker over an alcohol lamp is bubbling and the lamp hums softly. The supervisor is taller than the researcher holding a notebook. "
                   "Three researchers work in the lab. There is no window.")
    d["nodes"].append(dict(id="beaker_hums", type="state", of="beaker", desc="the lamp hums", question="Is the lamp humming?"))
    d["nodes"].append(dict(id="sup_wall", type="state", of="supervisor", desc="the supervisor stands by the wall", question="Is the supervisor standing by the wall?"))
    seen, prompts = [], []
    coat = {"id": "supervisor_coat", "type": "attribute", "of": "supervisor", "desc": "the supervisor wears a white coat", "question": "Is the supervisor wearing a white coat?", "evidence": "in a white coat ..."}
    lamp = {"id": "lamp", "type": "entity", "class": "alcohol lamp", "plural": "alcohol lamps", "count": 1, "desc": "an alcohol lamp", "question": "Is there an alcohol lamp?", "evidence": "A glass beaker over an alcohol lamp is bubbling"}
    under = {"id": "lamp_under", "type": "relation", "of": ["beaker", "lamp"], "desc": "the beaker is over the lamp", "question": "Is the beaker over the alcohol lamp?", "evidence": "A glass beaker over an alcohol lamp is bubbling"}

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        seen.append(tag)
        prompts.append(messages[-1]["content"])
        if tag.startswith("atomize_repair"):
            return json.dumps({"rewrite": [
                {"node": {"id": "supervisor", "type": "entity", "class": "supervisor", "desc": "a supervisor", "question": "Is there a supervisor?"},
                 "new": [coat, {"id": "screen", "type": "entity", "class": "computer screen", "plural": "computer screens", "count": 1, "desc": "a computer screen", "question": "Is there a computer screen?", "box": [700, 100, 830, 300], "evidence": "next to a computer screen"},
                         {"id": "sup_next_screen", "type": "relation", "of": ["supervisor", "screen"], "desc": "the supervisor stands next to the screen", "question": "Is the supervisor next to the computer screen?", "evidence": "A supervisor in a white coat stands next to a computer screen"}]},
                {"node": {"id": "beaker", "type": "entity", "class": "beaker", "desc": "a glass beaker", "question": "Is there a glass beaker?"},
                 "new": [dict(lamp, box=[720, 730, 840, 800]), under]}]})
        assert "supervisor [entity]" in messages[-1]["content"]
        return json.dumps({
            "rewrite": [
                {"node": {"id": "supervisor", "type": "entity", "class": "supervisor", "desc": "a supervisor", "question": "Is there a supervisor?"}, "new": [coat]},
                {"node": {"id": "researcher_b", "type": "entity", "class": "researcher", "desc": "a researcher", "question": "Is there a researcher?"},
                 "new": [{"id": "b_notebook", "type": "attribute", "of": "researcher_b", "desc": "the researcher holds a notebook", "question": "Is the researcher holding a notebook?", "evidence": "holding a notebook"},
                         {"id": "b_novel", "type": "state", "of": "researcher_b", "desc": "the researcher reads a novel", "question": "Is the researcher reading a novel?", "evidence": "the researcher secretly reads a novel"}]},
                {"node": {"id": "beaker", "type": "entity", "class": "beaker", "desc": "a glass beaker", "question": "Is there a glass beaker?"}, "new": [lamp, under]}],
            "add": [{"id": "sup_taller", "type": "relation", "of": ["supervisor", "researcher_b"], "desc": "the supervisor is taller than the researcher holding a notebook", "question": "Is the supervisor taller than the researcher holding a notebook?", "evidence": "The supervisor is taller than the researcher holding a notebook"},
                    {"id": "sup_watch", "type": "state", "of": "supervisor", "desc": "the supervisor points at the screen", "question": "Is the supervisor pointing at the screen?", "evidence": "points at the screen while talking"}],
            "set_count": [{"id": "researcher_a", "count": 3, "evidence": "Three researchers work in the lab"}],
            "delete": [{"id": "beaker_hums", "why": "sound"}, {"id": "researcher_c", "why": "not allowed: an entity"}, {"id": "beaker_bubbling", "why": "redundant with the beaker node"},
                       {"id": "sup_wall", "why": "sound"}],
            "edges": []})

    out, info = atomize_pass(d, chat=fake)
    assert seen == ["atomize:" + d["id"], "atomize_repair:" + d["id"]], seen
    rp = prompts[1]
    assert "- supervisor: no node asks these words" in rp and "next, computer" in rp and "- researcher_b: new node b_novel" in rp and "- beaker:" in rp and "box" in rp, rp
    by = {n["id"]: n for n in out["nodes"]}
    assert by["supervisor"]["question"] == "Is there a supervisor?" and {"supervisor_coat", "screen", "sup_next_screen"} <= set(by), info
    assert by["researcher_b"]["question"] == "Is there a researcher holding a notebook?" and "b_notebook" not in by and "b_novel" not in by, info  # left out of the repair: stays as it was
    assert by["beaker"]["question"] == "Is there a glass beaker?" and by["lamp"]["box"] == [720, 730, 840, 800] and by["lamp_under"]["of"] == ["beaker", "lamp"], info
    assert "sup_taller" in by and by["sup_taller"]["of"] == ["supervisor", "researcher_b"] and "sup_watch" not in by and [x["id"] for x in info["add_refused"]] == ["sup_watch"], info
    assert "beaker_hums" not in by and "researcher_c" in by and "beaker_bubbling" in by and "sup_wall" in by, info
    assert [x["id"] for x in info["deleted"]] == ["beaker_hums"] and [x["id"] for x in info["kept_delete"]] == ["beaker_bubbling", "sup_wall"], info
    assert by["researcher_a"]["count"] == 3 and info["set_count"] == 1, info
    assert info["rewrites"] == 5 and info["rewrites_kept"] == 2 and info["repair"]["kept"] == 2 and info["added"] == 1 and "beaker" in info["reverted"] or info["rewrites_kept"] == 2, info
    outcomes = {(g["id"], g["round"]): g["outcome"] for g in info["groups"]}
    assert outcomes == {("supervisor", 1): "lost", ("researcher_b", 1): "ungrounded", ("beaker", 1): "reverted", ("supervisor", 2): "kept", ("beaker", 2): "kept"}, outcomes
    assert all("evidence" not in n and "_grp" not in n for n in out["nodes"])
    assert out["meta"]["atomize"]["rewrites"] == 2 and out["meta"]["atomize"]["deleted"] == 1 and out["meta"]["atomize"]["repair"] == 2
    from draftdag.schema import Plan
    assert not Plan(out).validate(max_nodes=128)[0]

    def nothing(messages, model=None, tag="", max_tokens=0, **kw):
        return json.dumps({"rewrite": [], "add": [], "set_count": [], "delete": [], "edges": []})

    out2, info2 = atomize_pass(d, chat=nothing)
    assert out2 is d and info2["rewrites"] == 0 and not info2["deleted"] and info2["repair"] is None


def ship_plan():
    W = H = 1024
    nodes = [
        dict(id="bg", type="background", desc="stormy waters under a tall mast", question="Does the image show stormy waters under a tall mast?", box=[0, 0, W, H]),
        dict(id="longship", type="entity", **{"class": "longship"}, plural="longships", count=1, count_exact=True, desc="a massive Viking longship with a curved hull", question="Is there a massive Viking longship with a curved hull?", box=[100, 300, 924, 800]),
        dict(id="oars", type="entity", **{"class": "oar"}, plural="oars", count=10, count_exact=True, desc="ten wooden oars painted with blue and yellow geometric patterns", question="Are there ten wooden oars painted with blue and yellow geometric patterns?", box=[150, 500, 874, 700]),
        dict(id="deck", type="entity", **{"class": "deck"}, plural="decks", count=1, count_exact=True, desc="a wooden deck secured by iron nails", question="Is there a wooden deck secured by iron nails?", box=[200, 450, 800, 650]),
        dict(id="lantern", type="entity", **{"class": "lantern"}, plural="lanterns", count=1, count_exact=True, desc="a lantern on the mast", question="Is there a lantern on the mast?", box=[480, 100, 540, 200]),
        dict(id="deck_nails", type="attribute", of="deck", desc="the deck planks are secured by iron nails", question="Are the wooden deck planks secured by iron nails?"),
        dict(id="hull_look", type="attribute", of="longship", desc="the hull is dark and covered in barnacles", question="Is the hull dark and covered in barnacles?"),
        dict(id="ship_condition", type="state", of="longship", desc="the longship is undamaged and free of rot", question="Is the longship undamaged and free of rot?"),
        dict(id="ship_creaks", type="state", of="longship", desc="the longship creaks loudly", question="Is the longship creaking loudly?"),
        dict(id="oars_splash", type="state", of="oars", desc="the oars splash and creak", question="Are the oars splashing and creaking?"),
        dict(id="oars_on_ship", type="relation", of=["oars", "longship"], desc="the oars extend from the sides of the longship", question="Do the oars extend from the sides of the longship?"),
        dict(id="no_rot", type="absent", desc="signs of rot", question="Are there any signs of rot on the ship?"),
    ]
    return dict(id="ship_0001", prompt=("A massive Viking longship with a curved, dark hull covered in barnacles sails through stormy waters under a tall mast. "
                                        "The longship is not damaged and shows no signs of rot, though it creaks loudly. Ten wooden oars extend from both sides, "
                                        "each painted with blue and yellow geometric patterns, splashing and creaking. The deck is made of wooden planks secured by "
                                        "iron nails. A lantern hangs on the mast."), canvas=[W, H], nodes=nodes, edges=[])


def test_atomize_ship():
    """Format variants seen in the 10-09 run: a shortened entity without new nodes is applied when the plan already asks
    the rest (deck) and goes to the repair round when it would lose words (oars); an attribute returned unchanged with
    its parts as new nodes hands its id to the first part (hull_look), so the bundled question does not stay beside its
    parts; a new node that an absent node already asks is not added (ship_no_rot); a new entity that makes an untouched
    entity place itself on it reverts its group (mast); deletions follow the node's own words."""
    from draftdag.schema import Plan
    d = ship_plan()
    assert not Plan(json.loads(json.dumps(d))).validate(max_nodes=128)[0]
    seen, prompts = [], []

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        seen.append(tag)
        prompts.append(messages[-1]["content"])
        if tag.startswith("atomize_repair"):
            return json.dumps({"rewrite": [
                {"node": {"id": "oars", "type": "entity", "class": "oar", "plural": "oars", "count": 10, "desc": "ten wooden oars", "question": "Are there ten wooden oars?"},
                 "new": [{"id": "oars_pattern", "type": "attribute", "of": "oars", "desc": "the oars are painted with blue and yellow geometric patterns", "question": "Are the oars painted with blue and yellow geometric patterns?", "evidence": "each painted with blue and yellow geometric patterns"}]}]})
        return json.dumps({
            "rewrite": [
                {"node": {"id": "bg", "type": "background", "desc": "stormy waters", "question": "Does the image show stormy waters?"},
                 "new": [{"id": "mast", "type": "entity", "class": "mast", "plural": "masts", "count": 1, "desc": "a tall mast", "question": "Is there a tall mast?", "box": [490, 50, 530, 500], "evidence": "under a tall mast"}]},
                {"node": {"id": "longship", "type": "entity", "class": "longship", "desc": "a massive Viking longship", "question": "Is there a massive Viking longship?"},
                 "new": [{"id": "longship_hull", "type": "attribute", "of": "longship", "desc": "the longship has a curved hull", "question": "Does the longship have a curved hull?"}]},
                {"node": {"id": "oars", "type": "entity", "class": "oar", "desc": "ten wooden oars", "question": "Are there ten wooden oars?"}, "new": []},
                {"node": {"id": "deck", "type": "entity", "class": "deck", "desc": "a wooden deck", "question": "Is there a wooden deck?"}, "new": [], "already": ["deck_nails"]},
                {"node": dict(d["nodes"][6]),
                 "new": [{"id": "hull_dark", "type": "attribute", "of": "longship", "desc": "the hull is dark", "question": "Is the hull dark?"},
                         {"id": "hull_barnacles", "type": "attribute", "of": "longship", "desc": "the hull is covered in barnacles", "question": "Is the hull covered in barnacles?"}]},
                {"node": {"id": "ship_condition", "type": "state", "of": "longship", "desc": "the longship is undamaged", "question": "Is the longship undamaged?"},
                 "new": [{"id": "ship_no_rot", "type": "state", "of": "longship", "desc": "the longship shows no signs of rot", "question": "Does the longship show no signs of rot?", "evidence": "shows no signs of rot"}]}],
            "add": [{"id": "oars_wood", "type": "attribute", "of": "oars", "desc": "the oars are wooden", "question": "Are the oars wooden?", "evidence": "Ten wooden oars extend"}],
            "set_count": [], "edges": [{"from": "bg", "to": "mast", "kind": "support"}],
            "delete": [{"id": "ship_creaks", "why": "sound"}, {"id": "oars_splash", "why": "sound"}, {"id": "oars_on_ship", "why": "redundant"}]})

    out, info = atomize_pass(d, chat=fake)
    by = {n["id"]: n for n in out["nodes"]}
    assert seen == ["atomize:ship_0001", "atomize_repair:ship_0001"], seen
    assert "- oars: no node asks these words" in prompts[1] and "painted, blue, yellow, geometric, patterns" in prompts[1] and "- bg:" in prompts[1] and "lantern" in prompts[1], prompts[1]
    assert by["bg"]["question"] == "Does the image show stormy waters under a tall mast?" and "mast" not in by and "bg" in info["reverted"], info  # mast makes "a lantern on the mast" place itself on an entity
    assert by["longship"]["question"] == "Is there a massive Viking longship?" and "longship_hull" in by
    assert by["oars"]["question"] == "Are there ten wooden oars?" and "oars_pattern" in by, info
    assert by["deck"]["question"] == "Is there a wooden deck?" and "deck_nails" in by
    assert by["hull_look"]["question"] == "Is the hull dark?" and "hull_dark" not in by and "hull_barnacles" in by, info
    assert by["ship_condition"]["question"] == "Is the longship undamaged?" and "ship_no_rot" not in by and "no_rot" in by, info
    assert "ship_creaks" not in by and "oars_splash" in by and "oars_on_ship" in by and "oars_wood" not in by, info
    assert [x["id"] for x in info["kept_delete"]] == ["oars_splash"] and info["kept_delete"][0]["lost"] == ["splash"], info["kept_delete"]
    assert info["rewrites_kept"] == 5 and info["repair"]["kept"] == 1, info
    assert not Plan(out).validate(max_nodes=128)[0]


def test_atomize_retype():
    """A background that lists an object is shortened and the object becomes an entity; a state that places one entity on
    that object is retyped to a relation under the same id; an entity the plan already has is not added twice; when the
    background group fails (its new entity has no box) the retype that depends on it is reverted too."""
    d = strip_generated(lab_plan()).to_dict()
    d["prompt"] = "In a chemistry laboratory with a long workbench and reagent cabinets, a glass beaker stands on the workbench over an alcohol lamp. A supervisor in a white coat watches."
    d["nodes"].append(dict(id="beaker_on_bench", type="state", of="beaker", desc="the beaker stands on the workbench", question="Is the glass beaker standing on the workbench?"))

    def answer(box):
        bench = {"id": "workbench", "type": "entity", "class": "workbench", "plural": "workbenches", "count": 1, "count_exact": True, "desc": "a long workbench", "question": "Is there a long workbench?", "evidence": "a long workbench"}
        if box:
            bench["box"] = [0, 600, 1024, 900]
        return json.dumps({"rewrite": [
            {"node": {"id": "bg", "type": "background", "desc": "a chemistry laboratory", "question": "Is this a chemistry laboratory?"},
             "new": [bench, {"id": "supervisor", "type": "entity", "class": "supervisor", "desc": "a supervisor", "question": "Is there a supervisor?", "box": [840, 200, 1010, 660], "evidence": "A supervisor in a white coat watches"}]},
            {"node": {"id": "beaker_on_bench", "type": "relation", "of": ["beaker", "workbench"], "desc": "the beaker stands on the workbench", "question": "Is the glass beaker standing on the workbench?", "evidence": "a glass beaker stands on the workbench"}, "new": []}],
            "add": [], "set_count": [], "delete": [], "edges": [{"from": "workbench", "to": "beaker", "kind": "support"}, {"from": "ghost", "to": "beaker", "kind": "support"}]})

    out, info = atomize_pass(d, chat=lambda *a, **k: answer(True))
    by = {n["id"]: n for n in out["nodes"]}
    assert by["bg"]["question"] == "Is this a chemistry laboratory?" and by["workbench"]["type"] == "entity", info
    assert by["beaker_on_bench"]["type"] == "relation" and by["beaker_on_bench"]["of"] == ["beaker", "workbench"], info
    assert sum(1 for n in out["nodes"] if n["id"] == "supervisor") == 1 and info["dups"] == 1, info
    assert {"from": "workbench", "to": "beaker", "kind": "support"} in out["edges"] and not any(e.get("from") == "ghost" for e in out["edges"]), out["edges"]
    assert info["rewrites_kept"] == 2 and not info["reverted"], info
    out2, info2 = atomize_pass(d, chat=lambda *a, **k: answer(False))
    by2 = {n["id"]: n for n in out2["nodes"]}
    assert by2["bg"]["question"] == d["nodes"][0]["question"] and "workbench" not in by2, info2
    assert by2["beaker_on_bench"]["type"] == "state" and set(info2["reverted"]) == {"bg", "beaker_on_bench"}, info2


def test_atomize_alias():
    """A bundled entity split into two new entities without keeping its id: the first new entity takes the old id (its
    children keep their parent), references to the alias are rewritten; the kept node listed among the new ones is merged."""
    d = strip_generated(lab_plan()).to_dict()
    d["prompt"] = "In a chemistry lab, a supervisor in a white coat stands next to a computer screen. A glass beaker over an alcohol lamp is bubbling."

    def ans(*a, **k):
        return json.dumps({"rewrite": [
            {"node": dict(next(n for n in d["nodes"] if n["id"] == "beaker")),
             "new": [{"id": "glass_beaker", "type": "entity", "class": "beaker", "plural": "beakers", "count": 1, "desc": "a glass beaker", "question": "Is there a glass beaker?", "box": [700, 560, 860, 720], "evidence": "A glass beaker"},
                     {"id": "lamp", "type": "entity", "class": "alcohol lamp", "plural": "alcohol lamps", "count": 1, "desc": "an alcohol lamp", "question": "Is there an alcohol lamp?", "box": [720, 730, 840, 800], "evidence": "over an alcohol lamp"},
                     {"id": "beaker_over_lamp", "type": "relation", "of": ["glass_beaker", "lamp"], "desc": "the beaker is over the lamp", "question": "Is the glass beaker over the alcohol lamp?", "evidence": "A glass beaker over an alcohol lamp"}]},
            {"node": {"id": "bg", "type": "background", "desc": d["nodes"][0]["desc"], "question": d["nodes"][0]["question"]},
             "new": [{"id": "bg", "type": "background", "desc": "a chemistry laboratory", "question": "Is this a chemistry laboratory?"},
                     {"id": "bench", "type": "entity", "class": "workbench", "plural": "workbenches", "count": 1, "desc": "a long workbench", "question": "Is there a workbench?", "box": [0, 600, 1024, 900], "evidence": "a long workbench"}]}],
            "add": [], "set_count": [], "delete": [], "edges": [{"from": "lamp", "to": "glass_beaker", "kind": "support"}]})

    out, info = atomize_pass(d, chat=ans)
    by = {n["id"]: n for n in out["nodes"]}
    assert by["beaker"]["question"] == "Is there a glass beaker?" and "glass_beaker" not in by and by["beaker_over_lamp"]["of"] == ["beaker", "lamp"], info
    assert by["beaker_bubbling"]["of"] == "beaker" and {"from": "lamp", "to": "beaker", "kind": "support"} in out["edges"], out["edges"]
    assert by["bg"]["question"] == "Is this a chemistry laboratory?" and "bench" in by and info["rewrites_kept"] == 2, info


def test_atomize_lift_repeats():
    """A background split that adds an entity turns an untouched state placing its entity on it into a relation (as the
    first pass does); a kept node that still asks what one of its new nodes asks is not applied; a sound state is
    deleted even when the entity's own description uses the word."""
    from draftdag.schema import Plan
    W = H = 1024
    d = dict(id="court_0001", canvas=[W, H], edges=[], prompt=("A beach volleyball court with sand and a net. A player wears a blue jersey with white numbers. "
                                                            "A dog sits near the net and barks loudly. A crackling campfire with burning logs burns behind the court."), nodes=[
        dict(id="bg", type="background", desc="a beach volleyball court with sand and a net", question="Does the image show a beach volleyball court with sand and a net?", box=[0, 0, W, H]),
        dict(id="player", type="entity", **{"class": "player"}, plural="players", count=1, count_exact=True, desc="a volleyball player", question="Is there a volleyball player?", box=[100, 200, 400, 900]),
        dict(id="dog", type="entity", **{"class": "dog"}, plural="dogs", count=1, count_exact=True, desc="a dog", question="Is there a dog?", box=[600, 700, 800, 900]),
        dict(id="campfire", type="entity", **{"class": "campfire"}, plural="campfires", count=1, count_exact=True, desc="a crackling campfire with burning logs", question="Is there a campfire with burning logs?", box=[820, 600, 1000, 800]),
        dict(id="fire_scene", type="state", of="campfire", desc="the campfire crackles with burning logs", question="Is the campfire crackling with burning logs?"),
        dict(id="player_look", type="attribute", of="player", desc="the player wears a blue jersey with white numbers", question="Does the player wear a blue jersey with white numbers?"),
        dict(id="dog_near_net", type="state", of="dog", desc="the dog sits near the net", question="Is the dog sitting near the net?"),
        dict(id="dog_barks", type="state", of="dog", desc="the dog barks loudly", question="Is the dog barking loudly?"),
        dict(id="fire_crackles", type="state", of="campfire", desc="the campfire crackles", question="Is the campfire crackling?"),
    ])
    assert not Plan(json.loads(json.dumps(d))).validate(max_nodes=128)[0]

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        if tag.startswith("atomize_repair"):
            assert "- player_look: the node" in messages[-1]["content"] and "still asks what new node look_numbers" in messages[-1]["content"], messages[-1]["content"]
            return json.dumps({"rewrite": []})
        return json.dumps({"rewrite": [
            {"node": {"id": "bg", "type": "background", "desc": "a beach volleyball court with sand", "question": "Does the image show a beach volleyball court with sand?"},
             "new": [{"id": "net", "type": "entity", "class": "net", "plural": "nets", "count": 1, "desc": "a volleyball net", "question": "Is there a volleyball net?", "box": [0, 300, 1024, 600], "evidence": "with sand and a net"}]},
            {"node": {"id": "player_look", "type": "attribute", "of": "player", "desc": "the player wears a jersey with white numbers", "question": "Does the player wear a jersey with white numbers?"},
             "new": [{"id": "look_blue", "type": "attribute", "of": "player", "desc": "the jersey is blue", "question": "Is the player's jersey blue?"},
                     {"id": "look_numbers", "type": "attribute", "of": "player", "desc": "the jersey has white numbers", "question": "Does the jersey have white numbers?"}]},
            {"node": {"id": "campfire", "type": "entity", "class": "campfire", "desc": "a campfire", "question": "Is there a campfire?"},
             "new": [{"id": "fire_logs", "type": "state", "of": "campfire", "desc": "logs burn in the campfire", "question": "Are there burning logs in the campfire?"}]},
            {"node": {"id": "fire_scene", "type": "state", "of": "campfire", "desc": "the campfire has burning logs", "question": "Does the campfire have burning logs?"}, "new": []}],
            "add": [], "set_count": [], "edges": [{"from": "bg", "to": "net", "kind": "support"}],
            "delete": [{"id": "dog_barks", "why": "sound"}, {"id": "fire_crackles", "why": "sound"}]})

    out, info = atomize_pass(d, chat=fake)
    by = {n["id"]: n for n in out["nodes"]}
    assert by["bg"]["question"] == "Does the image show a beach volleyball court with sand?" and "net" in by, info
    assert by["dog_near_net"]["type"] == "relation" and by["dog_near_net"]["of"] == ["dog", "net"], by["dog_near_net"]
    assert by["player_look"]["question"] == "Does the player wear a blue jersey with white numbers?" and "look_blue" not in by and "look_numbers" not in by, info
    assert {g["id"]: g["outcome"] for g in info["groups"]} == {"bg": "kept", "player_look": "repeats", "campfire": "kept", "fire_scene": "kept"}, info["groups"]
    assert by["campfire"]["question"] == "Is there a campfire?" and "fire_logs" in by and "fire_scene" not in by and [x["id"] for x in info["cleanup"]] == ["fire_scene"], info  # the trimmed node only repeated fire_logs
    assert "dog_barks" not in by and "fire_crackles" not in by and [x["id"] for x in info["deleted"]] == ["dog_barks", "fire_crackles"], info
    assert not Plan(out).validate(max_nodes=128)[0]


def test_atomize_dependency_message():
    """A retype that needs an entity of a rewrite that was not applied is reverted, and the repair message names that
    rewrite, so the planner can fix both in the repair round."""
    W = H = 1024
    d = dict(id="kit_0001", canvas=[W, H], edges=[], prompt="A modern kitchen with granite countertops and windows. A sharp knife lies on the granite countertop.", nodes=[
        dict(id="bg", type="background", desc="a modern kitchen with granite countertops and windows", question="Does the image show a modern kitchen with granite countertops and windows?", box=[0, 0, W, H]),
        dict(id="knife", type="entity", **{"class": "knife"}, plural="knives", count=1, count_exact=True, desc="a sharp knife", question="Is there a sharp knife?", box=[400, 500, 600, 560]),
        dict(id="knife_on_counter", type="state", of="knife", desc="the knife is on the granite countertop", question="Is the sharp knife on the granite countertop?"),
    ])
    prompts = []

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        prompts.append(messages[-1]["content"])
        if tag.startswith("atomize_repair"):
            return json.dumps({"rewrite": []})
        return json.dumps({"rewrite": [
            {"node": {"id": "bg", "type": "background", "desc": "a modern kitchen", "question": "Does the image show a modern kitchen?"},
             "new": [{"id": "countertop", "type": "entity", "class": "countertop", "plural": "countertops", "count": 1, "desc": "a granite countertop", "question": "Is there a granite countertop?", "box": [0, 500, 1024, 800], "evidence": "with granite countertops"}]},
            {"node": {"id": "knife_on_counter", "type": "relation", "of": ["knife", "countertop"], "desc": "the knife is on the countertop", "question": "Is the sharp knife on the granite countertop?", "evidence": "A sharp knife lies on the granite countertop"}, "new": []}],
            "add": [], "set_count": [], "delete": [], "edges": []})

    out, info = atomize_pass(d, chat=fake)
    by = {n["id"]: n for n in out["nodes"]}
    assert by["bg"]["question"] == d["nodes"][0]["question"] and by["knife_on_counter"]["type"] == "state" and "countertop" not in by, info
    assert "- bg: no node asks these words" in prompts[1] and "window" in prompts[1], prompts[1]
    assert "- knife_on_counter: node knife_on_counter needs entity countertop, a new node of the rewrite of bg, which was not applied" in prompts[1], prompts[1]


def test_atomize_retry_and_cycle():
    """A reply that is not one JSON object is asked for once more; support edges the planner proposed that close a cycle
    are dropped, and the rewrites stay."""
    from draftdag.schema import Plan
    d = ship_plan()
    tags = []

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        tags.append(tag)
        if tag.startswith("atomize:"):
            return '{"rewrite": [{"node": {"id": "deck" ... let me think about whether the deck needs a rewrite'
        assert messages[-1]["content"].endswith("nothing before or after it."), messages[-1]["content"][-120:]
        return json.dumps({"rewrite": [
            {"node": {"id": "deck", "type": "entity", "class": "deck", "desc": "a wooden deck", "question": "Is there a wooden deck?"}, "new": [], "already": ["deck_nails"]}],
            "add": [], "set_count": [], "delete": [],
            "edges": [{"from": "longship", "to": "deck", "kind": "support"}, {"from": "deck", "to": "lantern", "kind": "support"}, {"from": "lantern", "to": "longship", "kind": "support"}]})

    out, info = atomize_pass(d, chat=fake)
    assert tags == ["atomize:ship_0001", "atomize_retry:ship_0001"] and info.get("parse_retry"), (tags, info.get("error"))
    by = {n["id"]: n for n in out["nodes"]}
    assert by["deck"]["question"] == "Is there a wooden deck?" and not out["edges"], (out["edges"], info.get("error"))
    assert not Plan(out).validate(max_nodes=128)[0]

    def broken(messages, model=None, tag="", max_tokens=0, **kw):
        return "not json at all"

    out2, info2 = atomize_pass(d, chat=broken)
    assert out2 is d and info2["error"] == "parse"


def test_atomize_alias_same_kind():
    """An entity returned unchanged hands its id only to a new entity of the same kind: "souvenir tables and craft tables
    with supplies" must not become "craft supplies" (every node about the tables would then mean the supplies)."""
    W = H = 1024
    d = dict(id="fair_0001", canvas=[W, H], edges=[], prompt="On Main Street there are souvenir tables and craft tables with supplies.", nodes=[
        dict(id="bg", type="background", desc="Main Street", question="Does the image show Main Street?", box=[0, 0, W, H]),
        dict(id="craft_tables", type="entity", **{"class": "table"}, plural="tables", count=2, count_exact=False, desc="souvenir tables and craft tables with supplies", question="Are there souvenir tables and craft tables with supplies?", box=[100, 500, 900, 800]),
    ])
    seen = []

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        seen.append(tag)
        if tag.startswith("atomize_repair"):
            assert "- craft_tables: the node still asks everything" in messages[-1]["content"], messages[-1]["content"]
            return json.dumps({"rewrite": []})
        return json.dumps({"rewrite": [
            {"node": dict(d["nodes"][1]),
             "new": [{"id": "supplies", "type": "entity", "class": "supply", "plural": "supplies", "count": 2, "desc": "craft supplies", "question": "Are there craft supplies?", "box": [300, 450, 600, 520], "evidence": "craft tables with supplies"}]}],
            "add": [], "set_count": [], "delete": [], "edges": []})

    out, info = atomize_pass(d, chat=fake)
    by = {n["id"]: n for n in out["nodes"]}
    assert by["craft_tables"]["question"] == "Are there souvenir tables and craft tables with supplies?" and by["craft_tables"]["class"] == "table" and "supplies" not in by, info
    assert seen == ["atomize:fair_0001", "atomize_repair:fair_0001"] and info["groups"][0]["outcome"] == "unchanged", info["groups"]


def test_atomize_cleanup_parts():
    """After the rewrites, a rewritten child node that only bundles nodes another rewrite made goes; one that merely
    shares a word with another fact ("a red cup" and "a red hat") stays."""
    from draftdag.schema import Plan
    W = H = 1024
    d = dict(id="hall_0001", canvas=[W, H], edges=[], prompt=("In a school hallway a man in a red hat holds a red cup that smells of coffee. A cleaning cart with a mop and a "
                                                           "bucket of soapy water stands nearby; the cart holds a mop, a bucket of soapy water and towels."), nodes=[
        dict(id="bg", type="background", desc="a school hallway", question="Does the image show a school hallway?", box=[0, 0, W, H]),
        dict(id="man", type="entity", **{"class": "man"}, plural="men", count=1, count_exact=True, desc="a man", question="Is there a man?", box=[100, 200, 400, 900]),
        dict(id="cart", type="entity", **{"class": "cart"}, plural="carts", count=1, count_exact=True, desc="a cleaning cart with a mop and a bucket of soapy water", question="Is there a cleaning cart with a mop and a bucket of soapy water?", box=[600, 500, 900, 900]),
        dict(id="man_hat", type="attribute", of="man", desc="the man wears a red hat", question="Is the man wearing a red hat?"),
        dict(id="man_cup", type="attribute", of="man", desc="the man holds a red cup that smells of coffee", question="Is the man holding a red cup that smells of coffee?"),
        dict(id="cart_contents", type="attribute", of="cart", desc="the cart holds a mop, a bucket of soapy water and towels", question="Does the cart hold a mop, a bucket of soapy water and towels?"),
        dict(id="cart_load", type="attribute", of="cart", desc="the cart holds a mop and a bucket of soapy water", question="Does the cleaning cart hold a mop and a bucket of soapy water?"),
    ])

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        assert not tag.startswith("atomize_repair"), messages[-1]["content"]
        return json.dumps({"rewrite": [
            {"node": {"id": "cart", "type": "entity", "class": "cart", "desc": "a cleaning cart", "question": "Is there a cleaning cart?"},
             "new": [{"id": "cart_mop", "type": "attribute", "of": "cart", "desc": "the cart holds a mop", "question": "Does the cart hold a mop?"},
                     {"id": "cart_bucket", "type": "attribute", "of": "cart", "desc": "the cart holds a bucket of soapy water", "question": "Does the cart hold a bucket of soapy water?"}]},
            {"node": {"id": "cart_contents", "type": "attribute", "of": "cart", "desc": "the cart holds a mop and a bucket of soapy water", "question": "Does the cart hold a mop and a bucket of soapy water?"},
             "new": [{"id": "cart_towels", "type": "attribute", "of": "cart", "desc": "the cart holds towels", "question": "Does the cart hold towels?"}]},
            {"node": {"id": "man_cup", "type": "attribute", "of": "man", "desc": "the man holds a red cup", "question": "Is the man holding a red cup?"}, "new": []}],
            "add": [], "set_count": [], "delete": [], "edges": []})

    out, info = atomize_pass(d, chat=fake)
    by = {n["id"]: n for n in out["nodes"]}
    assert {g["id"]: g["outcome"] for g in info["groups"]} == {"cart": "kept", "cart_contents": "kept", "man_cup": "kept"}, info["groups"]
    assert "cart_contents" not in by and "cart_load" not in by and {"cart_mop", "cart_bucket", "cart_towels"} <= set(by) and sorted(c["id"] for c in info["cleanup"]) == ["cart_contents", "cart_load"], info["cleanup"]  # cart_load was untouched
    assert by["man_cup"]["question"] == "Is the man holding a red cup?" and "man_hat" in by, info
    assert not Plan(out).validate(max_nodes=128)[0]


def test_atomize_participants_and_relative():
    """A kept relation the planner put on another entity takes that entity when it is in the plan (and keeps its old
    participants' place otherwise: the rewrite is not applied); a position relative to another object ("on the left side
    of the waterhole") is not a frame position, so the node is not deleted; one in the image frame is."""
    from draftdag.schema import Plan
    W = H = 1024
    d = dict(id="sav_0001", canvas=[W, H], edges=[], prompt=("A savannah with a waterhole in the center. Four elephants drink on the left side of the waterhole; a boy reaches "
                                                          "for a cookie jar on the coffee table, and a girl stands on the left side of the image."), nodes=[
        dict(id="bg", type="background", desc="a savannah", question="Does the image show a savannah?", box=[0, 0, W, H]),
        dict(id="waterhole", type="entity", **{"class": "waterhole"}, plural="waterholes", count=1, count_exact=True, desc="a waterhole", question="Is there a waterhole?", box=[300, 500, 700, 700]),
        dict(id="elephants", type="entity", **{"class": "elephant"}, plural="elephants", count=4, count_exact=True, desc="four elephants", question="Are there exactly four elephants?", box=[0, 400, 300, 800]),
        dict(id="boy", type="entity", **{"class": "boy"}, plural="boys", count=1, count_exact=True, desc="a boy", question="Is there a boy?", box=[600, 200, 800, 700]),
        dict(id="girl", type="entity", **{"class": "girl"}, plural="girls", count=1, count_exact=True, desc="a girl", question="Is there a girl?", box=[0, 100, 150, 500]),
        dict(id="table", type="entity", **{"class": "coffee table"}, plural="coffee tables", count=1, count_exact=True, desc="a coffee table", question="Is there a coffee table?", box=[650, 600, 950, 800]),
        dict(id="jar", type="entity", **{"class": "cookie jar"}, plural="cookie jars", count=1, count_exact=True, desc="a cookie jar", question="Is there a cookie jar?", box=[750, 520, 850, 620]),
        dict(id="elephants_left", type="state", of="elephants", desc="the elephants are on the left side of the waterhole", question="Are the four elephants on the left side of the waterhole?"),
        dict(id="girl_left", type="state", of="girl", desc="the girl stands on the left side of the image", question="Is the girl positioned on the left side of the image?"),
        dict(id="boy_reaching", type="relation", of=["boy", "table"], desc="the boy reaches for the cookie jar", question="Is the boy reaching for the cookie jar on the coffee table?"),
        dict(id="girl_jar", type="relation", of=["girl", "jar"], desc="the girl drops a coin into the cookie jar", question="Is the girl dropping a coin into the cookie jar?"),
    ])
    assert not Plan(json.loads(json.dumps(d))).validate(max_nodes=128)[0]

    prompts = []

    def fake(messages, model=None, tag="", max_tokens=0, **kw):
        if tag.startswith("atomize_repair"):
            prompts.append(messages[-1]["content"])
            return json.dumps({"rewrite": []})
        return json.dumps({"rewrite": [
            {"node": {"id": "girl_jar", "type": "relation", "of": ["girl", "jar"], "desc": "the girl drops a coin", "question": "Is the girl dropping a coin?"},
             "new": [], "already": []},
            {"node": {"id": "boy_reaching", "type": "relation", "of": ["boy", "jar"], "desc": "the boy reaches for the cookie jar", "question": "Is the boy reaching for the cookie jar?"},
             "new": [{"id": "jar_on_table", "type": "relation", "of": ["jar", "table"], "desc": "the cookie jar is on the coffee table", "question": "Is the cookie jar on the coffee table?", "evidence": "a cookie jar on the coffee table"}]}],
            "add": [], "set_count": [], "edges": [],
            "delete": [{"id": "elephants_left", "why": "frame position"}, {"id": "girl_left", "why": "frame position"}]})

    out, info = atomize_pass(d, chat=fake)
    by = {n["id"]: n for n in out["nodes"]}
    assert by["boy_reaching"]["of"] == ["boy", "jar"] and by["boy_reaching"]["question"] == "Is the boy reaching for the cookie jar?" and by["jar_on_table"]["of"] == ["jar", "table"], info
    assert "elephants_left" in by and "girl_left" not in by and [x["id"] for x in info["deleted"]] == ["girl_left"], info
    assert by["girl_jar"]["question"] == "Is the girl dropping a coin into the cookie jar?" and "does not name jar" in prompts[0], prompts
    assert not Plan(out).validate(max_nodes=128)[0]


if __name__ == "__main__":
    test_planner_offline()
    print("planner offline tests ok")
