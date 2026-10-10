import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures import lab_plan

from redag.conditions import affected_conditions, build_conditions, is_askable, region_of
from redag.schema import Plan, strip_generated


def test_generated_nodes_and_graph():
    plan = lab_plan()
    types = {plan.typ(v) for v in plan.order}
    assert {"count", "pair", "quality"} <= types
    assert plan.class_count("researcher") == 3
    pairs = [v for v in plan.order if plan.typ(v) == "pair"]
    assert len(pairs) == 3
    assert plan.interferes("researcher_a", "researcher_b")
    assert set(plan.parents("handover")) == {"researcher_c", "researcher_a"}
    assert "a_pipette" in plan.children("researcher_a")
    assert "handover" in plan.descendants("researcher_c")
    assert plan.has_path("researcher_c", "handover")
    assert not plan.has_path("beaker", "supervisor")
    assert plan.support_parents("beaker") == ["bg"]
    lev = plan.levels()
    assert lev["handover"] > lev["researcher_a"]
    errs, warns = plan.validate()
    assert not errs


def test_conditions_gating_and_affected():
    plan = lab_plan()
    conds = build_conditions(plan)
    by = {c.cid: c for c in conds}
    assert by["n:a_pipette"].parents == ["researcher_a"]
    assert by["c:cnt_researcher"].inv is True
    sts = {"n:researcher_a": "fail"}
    assert not is_askable(by["n:a_pipette"], sts)
    sts = {"n:researcher_a": "pass"}
    assert is_askable(by["n:a_pipette"], sts)
    boxes = {"researcher_a": [400, 150, 620, 650], "beaker": [700, 560, 860, 720]}
    assert region_of(plan, by["n:a_pipette"], boxes) == [400, 150, 620, 650]
    assert region_of(plan, by["n:handover"], boxes)[0] == 80
    aff = affected_conditions(plan, conds, boxes, [[700, 560, 860, 720]], ["beaker"])
    assert "n:beaker_bubbling" in aff and "c:cnt_researcher" in aff and "q:q_vis" in aff
    assert "n:researcher_c" not in aff and "n:c_faces_a" not in aff
    assert "n:supervisor" in aff
    aff2 = affected_conditions(plan, conds, boxes, [[400, 150, 620, 650]], ["researcher_a"])
    assert "n:handover" in aff2 and "n:researcher_b" in aff2 and "p:pair_researcher_a__researcher_b" in aff2


def test_roundtrip_and_strip():
    plan = lab_plan()
    d = plan.to_dict()
    p2 = Plan(d)
    assert p2.order == plan.order
    s = strip_generated(plan)
    assert all(s.typ(v) not in ("count", "pair", "quality") for v in s.order)
    assert not any(e.get("kind") == "interfere" for e in s.edges)


def test_validation_catches_bad_plans():
    plan = lab_plan()
    s = strip_generated(plan)
    d = s.to_dict()
    d["nodes"][1]["question"] = "Is there a researcher at 300, 400?"
    errs, _ = Plan(d).validate()
    assert any("coordinates" in e for e in errs)
    d = s.to_dict()
    d["nodes"][6]["of"] = "nope"
    errs, _ = Plan(d).validate()
    assert any("of = one entity" in e for e in errs)
    d = s.to_dict()
    d["edges"].append({"from": "beaker", "to": "bg", "kind": "support"})
    errs, _ = Plan(d).validate()
    assert errs


def test_inexact_count_has_no_count_node():
    from redag.schema import add_generated_nodes

    d = lab_plan().to_dict()
    d["nodes"] = [n for n in d["nodes"] if n["type"] in ("background", "entity", "attribute", "state", "action", "relation", "absent")]
    d["edges"] = [e for e in d["edges"] if e.get("kind") != "interfere"]
    d["nodes"].append(dict(id="tomatoes", type="entity", **{"class": "tomato"}, plural="tomatoes", count=2, count_exact=False, desc="several red tomatoes", question="Are there several red tomatoes?", box=[700, 800, 900, 900]))
    d["nodes"].append(dict(id="bad", type="entity", **{"class": "lemon"}, plural="lemons", count=2, count_exact="maybe", desc="lemons", question="Are there lemons?", box=[100, 800, 300, 900]))
    p = Plan(d)
    errs, _ = p.validate()
    assert any("count_exact" in e for e in errs)
    d["nodes"] = [n for n in d["nodes"] if n["id"] != "bad"]
    p = add_generated_nodes(Plan(d))
    assert not p.validate()[0]
    assert "cnt_tomato" not in p.nodes and "cnt_researcher" in p.nodes
    assert p.class_count("tomato") == 2 and not p.class_count_exact("tomato") and p.class_count_exact("researcher")


if __name__ == "__main__":
    test_generated_nodes_and_graph()
    test_conditions_gating_and_affected()
    test_roundtrip_and_strip()
    test_validation_catches_bad_plans()
    test_inexact_count_has_no_count_node()
    print("schema tests ok")
