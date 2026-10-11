"""The corrected CoS-DAG is a visual dependency graph, not a prompt chain."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "draftdag_poc"))
from cos_dag import build_graph, validate_graph


def test_semantic_branches_join_only_at_dependencies():
    graph = build_graph()
    nodes = graph["nodes"]
    order = validate_graph(graph)
    assert len(order) == len(nodes)
    assert nodes["cat_truck"]["depends_on"] == ["cat", "truck_covered"]
    assert nodes["ladybug_bridge"]["depends_on"] == ["ladybugs", "oak", "red_ball"]
    assert set(nodes["bridge_cluster"]["depends_on"]) == {
        "ladybug_bridge", "ground_cluster", "right_cluster"
    }
    assert nodes["scene_assembly"]["node_type"] == "assembly"
    assert nodes["final_render"]["depends_on"][0] == "scene_assembly"
    assert all(nodes[name]["status"] == "planned_unexecuted" for name in nodes)
    assert order.index("cat_truck") < order.index("right_cluster") < order.index("scene_assembly")


def test_invalid_visual_artifact_and_cycle_are_rejected():
    graph = build_graph()
    broken = deepcopy(graph)
    broken["nodes"]["cat_truck"]["input_artifacts"] = ["cat/nonexistent.png"]
    with pytest.raises(ValueError, match="unavailable artifact"):
        validate_graph(broken)
    broken = deepcopy(graph)
    broken["nodes"]["cat"]["depends_on"] = ["cat_truck"]
    with pytest.raises(ValueError, match="Cycle"):
        validate_graph(broken)


def test_checklist_coverage_is_complete():
    graph = build_graph()
    broken = deepcopy(graph)
    broken["nodes"]["cat_truck"]["covers_checklist_ids"] = []
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_graph(broken)
