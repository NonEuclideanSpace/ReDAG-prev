"""HiCoGen-inspired, subject-centric DAG contract for the fixed scene.

This module plans and validates visual dependencies. It deliberately does not
pretend that Qwen's PNG-reference endpoint implements clean-latent synthesis.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROMPT = (HERE / "prompt.txt").read_text(encoding="utf-8").strip()
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def node(node_id: str, kind: str, deps: list[str], scope: str,
         covers: list[str] | None = None, anchor: str | None = None) -> dict:
    return {
        "node_id": node_id,
        "node_type": kind,
        "operation": {
            "layout": "structural_layout_draft",
            "unit": "single_semantic_unit_generation",
            "attribute": "single_attribute_generation",
            "subject": "clean_latent_conditioned_binding",
            "relation": "clean_latent_conditioned_relation",
            "subscene": "clean_latent_conditioned_local_composition",
            "assembly": "spatially_constrained_visual_assembly",
            "final": "one_scene_level_render",
        }[kind],
        "depends_on": deps,
        "target_scope": scope,
        "covers_checklist_ids": covers or [],
        "spatial_anchor": anchor,
        "input_artifacts": [
            artifact for dep in deps for artifact in (
                ("layout/layout.json", "layout/draft.png") if dep == "layout" else
                ("scene_assembly/composition.png", "scene_assembly/placements.json") if dep == "scene_assembly" else
                (f"{dep}/partial.png", f"{dep}/clean_latent.pt")
            )
        ],
        "output_artifacts": (
            ["layout.json", "draft.png"] if kind == "layout" else
            ["composition.png", "placements.json"] if kind == "assembly" else
            ["final.png"] if kind == "final" else
            ["draft.png", "partial.png", "clean_latent.pt"]
        ),
        "status": "planned_unexecuted",
        "preservation_rule": "Do not redraw completed full-scene pixels; consume parent visual artifacts and synthesize only this node's target scope.",
    }


def build_graph() -> dict:
    # Leaves describe independently renderable semantic units. Fine attributes
    # are bound to their owner before any cross-subject relation is synthesized.
    specs = [
        node("layout", "layout", [], "Meadow composition: left woman/fence, right oak/truck, distant tower; reserved relation corridors"),
        node("woman", "unit", [], "One woman, bent pose", ["woman"], "left foreground"),
        node("straw_hat", "attribute", [], "Straw hat belonging to woman"),
        node("floral_dress", "attribute", [], "Floral sundress belonging to woman"),
        node("hose", "unit", [], "Garden hose and adjustable nozzle"),
        node("retriever", "unit", [], "Running golden retriever", ["retriever"], "center-left ground"),
        node("red_ball", "unit", [], "Red tennis ball", ["red_ball"], "center ground"),
        node("terrier", "unit", [], "Small terrier in digging pose"),
        node("leather_harness", "attribute", [], "Small leather harness belonging to terrier"),
        node("oak", "unit", [], "Ancient oak with low branch", anchor="right"),
        node("fence", "unit", [], "Rusted fence", anchor="left background"),
        node("bicycle", "unit", [], "Bicycle"),
        node("basket_daisies", "attribute", [], "Wicker basket containing wilted daisies"),
        node("truck", "unit", [], "Pickup truck with visible hood", ["truck"], "near oak"),
        node("ivy", "attribute", [], "Ivy covering pickup truck"),
        node("cat", "unit", [], "One perched cat"),
        node("clock_tower", "unit", [], "Distant clock tower", ["clock_tower"], "far background"),
        node("drone", "unit", [], "Small hovering drone"),
        node("ladybugs", "unit", [], "Swarm of recognizable ladybugs", ["ladybugs"]),
        node("woman_styled", "subject", ["woman", "straw_hat", "floral_dress"], "Bind hat and dress to the woman; no other subjects", anchor="left foreground"),
        node("terrier_equipped", "subject", ["terrier", "leather_harness"], "Bind harness to terrier and retain digging pose"),
        node("bicycle_loaded", "subject", ["bicycle", "basket_daisies"], "Put wicker basket with wilted daisies on bicycle", ["wicker_basket"]),
        node("truck_covered", "subject", ["truck", "ivy"], "Put ivy on the pickup without obscuring its hood"),
        node("environment", "subscene", ["layout", "oak", "fence", "clock_tower"], "Meadow, sky, oak-right, fence-left, distant clock tower"),
        node("woman_hose", "relation", ["woman_styled", "hose"], "Woman hand adjusts nozzle; water forms a visible curved arc", ["woman_adjusts_hose", "water_arc"], "left foreground"),
        node("retriever_ball", "relation", ["retriever", "red_ball"], "Retriever chases red ball", anchor="center ground"),
        node("terrier_oak", "relation", ["terrier_equipped", "oak"], "Terrier actively digs at oak trunk base", ["terrier_digs_oak"], "right oak base"),
        node("bicycle_fence", "relation", ["bicycle_loaded", "fence"], "Bicycle physically leans against rusted fence", ["bicycle_fence"], "left background"),
        node("cat_truck", "relation", ["cat", "truck_covered"], "Cat paws perched ON the truck hood", ["cat_on_hood"], "right near oak"),
        node("ladybug_bridge", "relation", ["ladybugs", "oak", "red_ball"], "Ladybug chain starts on low oak branch and reaches toward red ball", ["ladybug_bridge"], "oak-to-ball corridor"),
        node("drone_sky", "relation", ["drone", "environment"], "Drone above scene, not on ground", ["drone"], "upper sky"),
        node("left_cluster", "subscene", ["woman_hose", "bicycle_fence"], "Left interactions with consistent scale"),
        node("ground_cluster", "subscene", ["retriever_ball"], "Dog chasing ball, with unoccluded red ball"),
        node("right_cluster", "subscene", ["terrier_oak", "cat_truck"], "Oak-base digging and cat-on-hood, without identity confusion"),
        node("bridge_cluster", "subscene", ["ladybug_bridge", "ground_cluster", "right_cluster"], "Cross-branch bridge must terminate near the SAME ball and oak", ["bridge_endpoints"]),
        node("scene_assembly", "assembly", ["layout", "environment", "left_cluster", "ground_cluster", "right_cluster", "bridge_cluster", "drone_sky"], "Spatially place completed local visual units once; do not re-render each prior unit"),
        node("final_render", "final", ["scene_assembly", "left_cluster", "ground_cluster", "right_cluster", "bridge_cluster"], "One scene-level refinement with preserved visual conditions; no repeated whole-scene additions"),
    ]
    graph = {
        "fixed_prompt_sha256": PROMPT_SHA256,
        "source": "HiCoGen Chain of Synthesis, adapted as a branch-based DAG; not a reproduction of its trained model",
        "paper": "https://arxiv.org/html/2511.19965",
        "execution_status": "plan_only_backend_missing",
        "required_backend": "subject-conditioned composer accepting reusable clean latents and a spatially constrained scene assembler",
        "current_backend_limitation": "Qwen-Image-2.1 endpoint accepts PNG references but exposes neither reusable clean latents nor a protected region/mask merge. Repeated calls regenerate the whole scene.",
        "nodes": {item["node_id"]: item for item in specs},
    }
    validate_graph(graph)
    return graph


def validate_graph(graph: dict) -> list[str]:
    nodes = graph["nodes"]
    order = []
    visiting = set()
    visited = set()

    def visit(name: str) -> None:
        if name not in nodes:
            raise ValueError(f"Unknown DAG dependency: {name}")
        if name in visiting:
            raise ValueError(f"Cycle in DAG at {name}")
        if name in visited:
            return
        visiting.add(name)
        for dep in nodes[name]["depends_on"]:
            visit(dep)
        visiting.remove(name)
        visited.add(name)
        order.append(name)

    for name in nodes:
        visit(name)
    for name, value in nodes.items():
        for artifact in value["input_artifacts"]:
            dep, filename = artifact.split("/", 1)
            if dep not in value["depends_on"] or filename not in nodes[dep]["output_artifacts"]:
                raise ValueError(f"{name} consumes unavailable artifact {artifact}")
    checklist = {row["id"] for row in json.loads((HERE / "constraints.json").read_text(encoding="utf-8"))["checklist"]}
    covered = {item for value in nodes.values() for item in value["covers_checklist_ids"]}
    if covered != checklist:
        raise ValueError(f"Checklist coverage mismatch: missing={checklist - covered}, unknown={covered - checklist}")
    if nodes["final_render"]["depends_on"] == ["scene_assembly"]:
        raise ValueError("Final render must receive local visual results as well as the assembly")
    return order


def write_graph() -> Path:
    graph = build_graph()
    destination = HERE / "cos_dag" / "graph.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(write_graph())
