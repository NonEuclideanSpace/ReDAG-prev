"""The CoS DAG debugger must reflect the executed graph and embed its PNGs."""
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "draftdag_poc"))
from visualize_png_dag import build_payload, topological_layers, write_html


def test_cycle_is_rejected():
    nodes = {"a": {"depends_on": ["b"]}, "b": {"depends_on": ["a"]}}
    try:
        topological_layers(nodes)
    except ValueError as exc:
        assert "Cycle" in str(exc)
    else:
        raise AssertionError("Cyclic graph was accepted")


def test_html_contains_outputs_and_actual_model_reference_order(tmp_path):
    run = tmp_path / "run"
    first = run / "nodes" / "first"
    second = run / "nodes" / "second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    for path in (first / "draft.png", first / "partial.png", second / "final.png"):
        Image.new("RGB", (8, 6), "red").save(path)

    graph = {"nodes": {
        "first": {"node_type": "unit", "depends_on": [], "output_artifacts": ["draft.png", "partial.png"]},
        "second": {"node_type": "render", "depends_on": ["first"],
                   "input_artifacts": ["first/partial.png"], "output_artifacts": ["final.png"]},
    }}
    (run / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    (first / "result.json").write_text(json.dumps({"status": "success", "reference_order": [str(first / "draft.png")]}), encoding="utf-8")
    (second / "result.json").write_text(json.dumps({"status": "success", "reference_order": [str(first / "partial.png")]}), encoding="utf-8")

    payload = build_payload(run)
    assert [node["layer"] for node in payload["nodes"]] == [0, 1]
    assert payload["edges"] == [{"from": "first", "to": "second", "artifacts": ["partial.png"]}]
    assert payload["nodes"][1]["model_refs"][0]["key"] == "nodes/first/partial.png"
    assert len(payload["assets"]) == 3
    assert not payload["missing"]

    output = write_html(run)
    html = output.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "first" in html and "second" in html
    assert "https://" not in html
