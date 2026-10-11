"""The executable approximation merges local PNG results only once globally."""
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "draftdag_poc"))
from png_cos_dag import foreground, graph, validate_graph


def test_local_branches_converge_into_one_scene_render():
    nodes = graph()["nodes"]
    order = validate_graph(graph())
    assert len(order) == len(nodes)
    assert nodes["ladybug_bridge"]["depends_on"] == ["retriever_ball", "terrier_oak"]
    assert {"woman_hose", "bicycle_fence", "terrier_oak", "cat_truck", "ladybug_bridge"}.issubset(
        nodes["scene_assembly"]["depends_on"]
    )
    assert order.index("cat_truck") < order.index("scene_assembly") < order.index("final_render")
    assert sum(item["node_type"] == "single_global_render" for item in nodes.values()) == 1
    assert nodes["final_render"]["input_artifacts"] == [
        "scene_assembly/composition.png", "scene_assembly/draft.png"
    ]


def test_flat_sketch_background_is_transparent():
    source = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(source).rectangle((25, 25, 75, 75), fill="red")
    result = foreground(source, (100, 100))
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((50, 50))[3] > 200
