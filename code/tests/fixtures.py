from draftdag.schema import Plan, add_generated_nodes


def lab_plan(W=1024, H=1024):
    nodes = [
        dict(id="bg", type="background", desc="a chemistry laboratory with a long workbench and reagent cabinets", question="Is this a chemistry laboratory with a workbench?", box=[0, 0, W, H]),
        dict(id="researcher_a", type="entity", cls=None, **{"class": "researcher"}, desc="a researcher wearing safety goggles", question="Is there a researcher wearing safety goggles?", box=[400, 150, 620, 650], count=1),
        dict(id="researcher_b", type="entity", **{"class": "researcher"}, desc="a researcher holding a notebook", question="Is there a researcher holding a notebook?", box=[640, 180, 820, 650], count=1),
        dict(id="researcher_c", type="entity", **{"class": "researcher"}, desc="a researcher standing in front of a fume hood on the left", question="Is there a researcher standing in front of a fume hood on the left?", box=[80, 160, 300, 660], count=1),
        dict(id="beaker", type="entity", **{"class": "beaker"}, desc="a glass beaker over an alcohol lamp", question="Is there a glass beaker over an alcohol lamp?", box=[700, 560, 860, 720], count=1),
        dict(id="supervisor", type="entity", **{"class": "supervisor"}, desc="a supervisor in a white coat next to a computer screen on the right", question="Is there a supervisor in a white coat next to a computer screen on the right?", box=[840, 200, 1010, 660], count=1),
        dict(id="a_pipette", type="attribute", of="researcher_a", desc="the goggled researcher holds a pipette in the right hand", question="Is the researcher wearing goggles holding a pipette in the right hand?"),
        dict(id="beaker_bubbling", type="state", of="beaker", desc="the beaker's liquid is bubbling", question="Is the liquid in the beaker bubbling?"),
        dict(id="c_faces_a", type="state", of="researcher_c", desc="the researcher at the fume hood faces the goggled researcher", question="Is the researcher at the fume hood turned toward the researcher wearing goggles?"),
        dict(id="handover", type="action", of=["researcher_c", "researcher_a"], desc="the researcher at the fume hood hands a bottle to the goggled researcher", question="Is the researcher at the fume hood handing a bottle to the researcher wearing goggles?"),
        dict(id="supervisor_points", type="relation", of=["supervisor", "researcher_b"], desc="the supervisor points at the screen while talking to the researcher with the notebook", question="Is the supervisor pointing at a screen while talking to the researcher holding a notebook?"),
        dict(id="no_window", type="absent", desc="a window", question="Is there a window in the image?"),
    ]
    for n in nodes:
        n.pop("cls", None)
    edges = [dict({"from": "bg", "to": "beaker", "kind": "support"})]
    plan = Plan(dict(id="lab_0001", prompt="chemistry lab scene", canvas=[W, H], nodes=nodes, edges=edges))
    errs, warns = plan.validate()
    assert not errs, errs
    return add_generated_nodes(plan)
