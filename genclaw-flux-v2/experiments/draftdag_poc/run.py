"""Fixed-prompt baseline, corrected CoS-DAG plan, and preserved legacy runs."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from cc_genclaw.config import load_genclaw_config
from cc_genclaw.llm import LLMClient, LLMConfig
from cc_genclaw.runtime.agent_runner import prepare_agent_run, run_prompt
from cc_genclaw.runtime.todo_state import get_todo_state
from cc_genclaw.tools.code_scene_draft import CodeSceneDraftTool
from cc_genclaw.tools.i2i import I2ITool
from cc_genclaw.tools._code_scene_draft.render import render_to_png
from cc_genclaw.tools.vlm_review import _extract_json_object

PROMPT_PATH = HERE / "prompt.txt"
PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
SEED = 260500
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

CHECKLIST = [
    ("retriever", "A golden retriever is present and recognizable."),
    ("red_ball", "A red tennis ball is present."),
    ("woman", "A woman is present."),
    ("woman_adjusts_hose", "The woman visibly adjusts or holds the garden hose nozzle."),
    ("water_arc", "The hose water forms a curved arc."),
    ("terrier_digs_oak", "A distinct terrier is visibly digging at the oak base."),
    ("bicycle_fence", "A bicycle leans against the fence."),
    ("wicker_basket", "The bicycle has a wicker basket containing wilted daisies."),
    ("truck", "A pickup truck is present near the oak."),
    ("cat_on_hood", "The cat is visibly perched on the truck hood, not beside it."),
    ("clock_tower", "A distant clock tower is visible."),
    ("drone", "A drone is visible above the scene."),
    ("ladybugs", "A swarm of ladybugs is visible."),
    ("ladybug_bridge", "The ladybugs visibly form a continuous bridge or chain."),
    ("bridge_endpoints", "That ladybug bridge visibly runs from a low oak branch toward the red tennis ball."),
]

CONSTRAINTS = {
    "prompt_sha256": PROMPT_SHA256,
    "entities": ["golden retriever", "red tennis ball", "woman", "straw hat", "floral sundress", "garden hose", "ancient oak tree", "terrier", "rusted fence", "bicycle", "wicker basket", "wilted daisies", "ivy-covered pickup truck", "cat", "distant clock tower", "drone", "swarm of ladybugs"],
    "attributes": [
        {"entity": "golden retriever", "value": "golden"},
        {"entity": "tennis ball", "value": "red"},
        {"entity": "woman", "value": "straw hat and floral sundress"},
        {"entity": "terrier", "value": "small leather harness"},
        {"entity": "fence", "value": "rusted"},
        {"entity": "bicycle", "value": "wicker basket with wilted daisies"},
        {"entity": "pickup truck", "value": "covered with ivy"},
    ],
    "relations": [
        {"subject": "woman", "predicate": "adjusts", "object": "garden hose"},
        {"subject": "terrier", "predicate": "digs_at_base_of", "object": "oak tree"},
        {"subject": "bicycle", "predicate": "leans_against", "object": "fence"},
        {"subject": "pickup truck", "predicate": "near", "object": "oak tree"},
        {"subject": "cat", "predicate": "on_hood_of", "object": "pickup truck"},
        {"subject": "ladybugs", "predicate": "bridge_between", "object": ["low oak branch", "red tennis ball"]},
        {"subject": "drone", "predicate": "above", "object": "scene"},
    ],
    "actions": ["retriever chases ball", "woman bends to adjust hose", "hose creates water arc", "terrier digs at oak base", "drone hovers", "ladybugs form bridge"],
    "global_constraints": ["sun-dappled meadow", "woman on left", "ancient oak on right", "rusted fence behind woman", "truck near oak", "clock tower in distant background"],
    "checklist": [{"id": key, "description": desc} for key, desc in CHECKLIST],
}

NODES = {
    "D0": {"depends_on": [], "targets": ["meadow", "woman left slot", "oak right slot", "fence background", "truck near tree slot", "clock tower background"], "instruction": "Create a sparse 1600x900 structural SVG background: sky, sun-dappled meadow, rusted fence behind the left half, distant small clock tower. Reserve uncluttered space for woman at x=250..500, oak at x=980..1220, and truck at x=1210..1580. Do not draw woman, tree, truck, dogs, bicycle, cat, drone, ladybugs, ball, or hose yet. Flat colors; no words, labels, frames or panels. The scene is one continuous landscape."},
    "D1": {"depends_on": ["D0"], "targets": ["woman", "straw hat", "floral sundress", "woman adjusts hose", "curved water arc", "bicycle leans against fence", "wicker basket with wilted daisies"], "instruction": "Add only the woman bending in a straw hat and floral sundress, her hand on the garden hose nozzle; a conspicuous blue curved water arc from the nozzle; a bicycle visibly leaning against the existing fence with a wicker basket of wilted daisies. Woman x=250..490 y=420..790, bicycle x=550..760 y=590..800. Do not redraw fence or other subjects."},
    "D2": {"depends_on": ["D0"], "targets": ["oak tree right", "golden retriever chases red tennis ball", "terrier in harness digging at oak base"], "instruction": "Add only an ancient oak on the right with trunk x=1020..1180 y=360..820 and a low branch extending toward x=930 y=520; a golden retriever at x=720..900 y=640..790 chasing a red tennis ball centered near x=960 y=740; a much smaller terrier wearing a leather harness actively digging dirt at the trunk base near x=1080 y=780. Keep the ball clearly visible. Do not redraw previous subjects."},
    "D3": {"depends_on": ["D0", "D2"], "targets": ["ivy-covered pickup truck near oak", "cat perched on truck hood"], "instruction": "Add only a parked pickup truck at x=1200..1570 y=630..810, close to the oak. The truck FRONT/HOOD is on the LEFT at x=1210..1330 y=660..720. Put ivy climbing on truck body and a clearly identifiable cat perched directly ON TOP OF THE HOOD at x=1250..1300 y=610..675, paws touching hood. Do not add cats elsewhere or redraw the oak."},
    "D4": {"depends_on": ["D2", "D3"], "targets": ["drone above scene", "ladybug swarm", "ladybugs form bridge from low oak branch toward red tennis ball"], "instruction": "Add only a drone hovering high above the scene near x=770 y=190, and a visually continuous curved chain of many small red-and-black ladybugs starting ON the oak's low branch near x=930 y=520 and extending downward toward the red tennis ball near x=960 y=740. Make the bridge/chain unmistakable and keep the ball visible. No other objects."},
    "D5": {"depends_on": ["D1", "D2", "D3", "D4"], "targets": ["merge D0-D4 without deleting any layer"], "instruction": "Assemble the preserved SVG background and four additive groups into one complete structural draft. Preserve every prior group byte-for-byte in the XML tree; no fresh model generation."},
    "D6": {"depends_on": ["D5"], "targets": ["photorealistic appearance", "sun-dappled light", "grass", "cloudy sky", "preserve layout and relationships"], "instruction": "Use the complete D5 draft as the i2i composition reference. Photorealistic landscape, natural grass, sun-dappled light, partly cloudy sky, realistic local detail. Preserve the exact objects, positions and relations from the reference draft."},
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize() -> None:
    load_genclaw_config()
    os.environ["CC_GENCLAW_IMAGE_PROVIDER"] = "qwen_remote"
    os.environ["QWEN_IMAGE_MODEL_NAME"] = "qwen-image"
    os.environ["QWEN_IMAGE_SEED"] = str(SEED)
    os.environ["QWEN_IMAGE_WIDTH"] = "1024"
    os.environ["QWEN_IMAGE_HEIGHT"] = "576"
    os.environ["QWEN_IMAGE_STEPS"] = "16"
    os.environ["QWEN_IMAGE_GUIDANCE"] = "1.0"
    os.environ["QWEN_IMAGE_STRENGTH"] = "0.65"
    os.environ["QWEN_IMAGE_TIMEOUT"] = "1800"
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / ".browsers")
    os.environ["CHAINLIT_NO_TELEMETRY"] = "true"
    write_json(HERE / "constraints.json", CONSTRAINTS)


def tool_json(result):
    payload = json.loads(result.content)
    if result.is_error or payload.get("status") != "success":
        raise RuntimeError(payload.get("error", result.content))
    return payload


def fresh_run_dir(name: str) -> Path:
    path = HERE / name / "runs"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["CC_GENCLAW_RUNS_DIR"] = str(path)
    from cc_genclaw.runtime.session_dir import reset_session_dir
    reset_session_dir(str(path))
    return path


def copy_draft(source: Path, target_dir: Path) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source.parent
    for name in ("final.svg", "final.html", "layout.json"):
        path = source_dir / name
        if path.is_file():
            shutil.copy2(path, target_dir / name)
    shutil.copy2(source, target_dir / "draft.png")
    return {"source_png": str(source), "draft_sha256": sha(target_dir / "draft.png")}


def final_prompt() -> str:
    return (PROMPT + "\n\nPhotorealistic single coherent landscape, sun-dappled lighting, natural grass and partly cloudy sky. "
            "Use the structural draft as the composition anchor. Preserve all visible objects and exact spatial relations, "
            "especially cat ON truck hood, terrier digging at oak base, and the ladybug bridge from oak branch toward red ball.")


def generate_final(input_png: Path, output_png: Path) -> dict:
    payload = tool_json(I2ITool().execute({"image_path": str(input_png.resolve()), "prompt": final_prompt()}))
    source = Path(payload["final_path"])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_png)
    metadata = source.with_suffix(".json")
    if metadata.is_file():
        shutil.copy2(metadata, output_png.with_suffix(".json"))
        if json.loads(metadata.read_text()).get("backend") == "mock":
            raise RuntimeError("Mock image is not a valid experiment result")
    return {"source_final": str(source), "final_sha256": sha(output_png), "generation_metadata": str(output_png.with_suffix(".json"))}


def baseline() -> None:
    out = HERE / "baseline"
    if (out / "final.png").exists():
        raise FileExistsError("Baseline already exists; keep it fixed for comparison")
    fresh_run_dir("baseline")
    print("baseline: GenClaw planner", flush=True)
    run = prepare_agent_run(enabled_tool_names=["todo_write"], max_turns=4)
    planning_request = ("For this fixed image prompt, call todo_write once to plan: generate one global SVG draft with "
                        "code_scene_draft, then render it once with i2i. Do not generate an image now.\n\n" + PROMPT)
    messages = run_prompt(planning_request, run)
    plan = get_todo_state().snapshot()
    if not plan:
        raise RuntimeError("GenClaw planner did not produce todo_write plan")
    write_json(out / "plan.json", {"prompt_sha256": PROMPT_SHA256, "planner": "GenClaw agent/todo_write", "model": run.model, "todos": plan, "messages": [str(getattr(m, "stop_reason", "")) for m in messages], "executed_pipeline": ["code_scene_draft", "i2i"], "note": "Raw planner output is recorded unchanged. The runner executes the fixed single-draft pipeline even if the generic planner proposes t2i or reaches max_turns."})
    print("baseline: one code_scene_draft", flush=True)
    draft_payload = tool_json(CodeSceneDraftTool().execute({"prompt": PROMPT, "self_review": False}))
    draft_info = copy_draft(Path(draft_payload["draft_png_path"]), out)
    print("baseline: i2i", flush=True)
    final_info = generate_final(out / "draft.png", out / "final.png")
    write_json(out / "artifacts.json", {"prompt_sha256": PROMPT_SHA256, "pipeline": ["GenClaw planner", "one code_scene_draft", "one i2i"], "plan": str(out / "plan.json"), "draft": draft_info, "final": final_info, "source_session": str(run.session.dir), "limitations": ["Planner is used for the plan; tool execution is deterministic to avoid agent loops."]})


def svg_root(svg_path: Path) -> ET.Element:
    root = ET.parse(svg_path).getroot()
    if root.tag != f"{{{SVG_NS}}}svg":
        raise ValueError(f"Expected standalone SVG: {svg_path}")
    return root


def normalize_canvas(root: ET.Element) -> ET.Element:
    width = float(re.match(r"[0-9.]+", root.get("width", "") or "").group())
    height = float(re.match(r"[0-9.]+", root.get("height", "") or "").group())
    if width <= 0 or height <= 0:
        raise ValueError("Invalid D0 canvas")
    if (width, height) != (1600, 900):
        # Preserve D0 content as one group while putting later additions in shared coordinates.
        group = ET.Element(f"{{{SVG_NS}}}g", {"id": "draftdag-D0-scaled", "transform": f"scale({1600 / width} {900 / height})"})
        for child in list(root):
            root.remove(child)
            group.append(child)
        root.append(group)
    root.set("width", "1600")
    root.set("height", "900")
    root.set("viewBox", "0 0 1600 900")
    return root


def draw(root: ET.Element, path: Path) -> None:
    svg = ET.tostring(root, encoding="unicode")
    path.write_text(svg, encoding="utf-8")
    render_to_png(svg, str(path.with_name("draft.png")), code_type="svg")


def fragment(instruction: str, prior_svg: Path) -> ET.Element:
    from cc_genclaw.tools._code_scene_draft.generate import _chat_completion, gen_model
    # Reuse the existing scene-draft LLM transport. Its public tool only supports full-scene drafts.
    system = ("You draw one additive SVG layer for a fixed 1600x900 landscape. "
              "Return only a single <g>...</g> SVG fragment, no markdown and no outer <svg>. "
              "Use simple recognizable flat SVG shapes: path, circle, ellipse, rect, polygon, line, polyline. "
              "No text, image, foreignObject, CSS, script, background, gradients, masks or filters. "
              "Do not redraw prior objects. Draw ONLY the requested new objects and relations. "
              "All coordinates are global canvas coordinates. Use explicit fill/stroke colors.")
    user = ("Existing SVG canvas and layout, for coordinate reference only:\n" + prior_svg.read_text(encoding="utf-8")[:18000]
            + "\n\nNew layer task:\n" + instruction)
    last_error = None
    for attempt in range(3):
        try:
            reply = _chat_completion(timeout=180.0, model=gen_model(), messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=8192).choices[0].message.content.strip()
            reply = re.sub(r"^```(?:xml|svg)?\s*|\s*```$", "", reply).strip()
            match = re.search(r"<g(?:\s[^>]*)?>[\s\S]*?</g>\s*$", reply)
            if not match:
                raise ValueError("No standalone SVG <g> fragment")
            group = ET.fromstring(f'<svg xmlns="{SVG_NS}">{match.group(0)}</svg>')[0]
            allowed = {"g", "path", "circle", "ellipse", "rect", "polygon", "line", "polyline"}
            if any(tag.tag.split("}")[-1] not in allowed for tag in group.iter()):
                raise ValueError("Disallowed SVG element")
            if len(list(group.iter())) < 4:
                raise ValueError("Fragment is too sparse")
            return group
        except (ValueError, ET.ParseError, AttributeError) as exc:
            last_error = exc
            user += f"\n\nPrevious format error: {exc}. Return valid <g> only."
    raise RuntimeError(f"Could not make SVG fragment: {last_error}")


def make_node(node_id: str, input_artifacts: list[str]) -> dict:
    out = HERE / "dag" / node_id
    out.mkdir(parents=True, exist_ok=True)
    spec = NODES[node_id]
    (out / "instruction.txt").write_text(spec["instruction"] + "\n", encoding="utf-8")
    node = {"node_id": node_id, "depends_on": spec["depends_on"], "target_constraints": spec["targets"], "input_artifacts": input_artifacts, "instruction": spec["instruction"], "output_artifacts": [], "prompt_sha256": PROMPT_SHA256}
    write_json(out / "node.json", node)
    return node


def finish_node(node: dict, result: dict, outputs: list[Path]) -> None:
    out = HERE / "dag" / node["node_id"]
    node["output_artifacts"] = [str(p.resolve()) for p in outputs]
    write_json(out / "node.json", node)
    write_json(out / "result.json", result)


def legacy_svg_dag() -> None:
    base = HERE / "dag"
    if (base / "final.png").exists():
        raise FileExistsError("DAG final already exists; keep it fixed for comparison")
    fresh_run_dir("dag")
    d0 = base / "D0"
    n0 = make_node("D0", [str(PROMPT_PATH.resolve()), str((HERE / "constraints.json").resolve())])
    print("D0: layout via code_scene_draft", flush=True)
    p0 = tool_json(CodeSceneDraftTool().execute({"prompt": NODES["D0"]["instruction"], "self_review": False}))
    copy_draft(Path(p0["draft_png_path"]), d0)
    source_svg = d0 / "final.svg"
    if not source_svg.is_file():
        raise RuntimeError("D0 did not produce SVG")
    root = normalize_canvas(svg_root(source_svg))
    draw(root, source_svg)
    finish_node(n0, {"status": "success", "method": "code_scene_draft", "canvas": [1600, 900], "svg_sha256": sha(source_svg)}, [d0 / "draft.png", source_svg])

    branch_source = {"D1": source_svg, "D2": source_svg, "D3": base / "D2" / "final.svg", "D4": base / "D3" / "final.svg"}
    for node_id in ("D1", "D2", "D3", "D4"):
        print(f"{node_id}: additive draft layer", flush=True)
        folder = base / node_id
        previous = branch_source[node_id]
        node = make_node(node_id, [str(previous.resolve()), str(previous.with_name("draft.png").resolve())])
        root = svg_root(previous)
        preserved = [ET.tostring(c, encoding="utf-8") for c in root]
        group = fragment(NODES[node_id]["instruction"], previous)
        group.set("id", f"draftdag-{node_id}")
        root.append(group)
        path = folder / "final.svg"
        draw(root, path)
        after = [ET.tostring(c, encoding="utf-8") for c in root][:-1]
        if after != preserved:
            raise RuntimeError(f"{node_id} changed a preserved layer")
        (folder / "layer.svg").write_text(ET.tostring(group, encoding="unicode"), encoding="utf-8")
        finish_node(node, {"status": "success", "method": "additive_svg_layer", "preserved_prior_groups": len(preserved), "layer_sha256": sha(folder / "layer.svg"), "svg_sha256": sha(path), "automatic_visibility_check": "skipped"}, [folder / "layer.svg", path, folder / "draft.png"])

    print("D5: merge preserved layers", flush=True)
    merged = base / "D5"
    d4_svg = base / "D4" / "final.svg"
    n5 = make_node("D5", [str((base / f"{n}" / "layer.svg").resolve()) for n in ("D1", "D2", "D3", "D4")] + [str(d4_svg.resolve())])
    root = svg_root(d4_svg)
    d0_children = len(list(svg_root(source_svg)))
    d1_group = ET.parse(base / "D1" / "layer.svg").getroot()
    root.insert(d0_children, d1_group)
    group_ids = [child.get("id") for child in root if child.get("id", "").startswith("draftdag-D")]
    if group_ids[-4:] != ["draftdag-D1", "draftdag-D2", "draftdag-D3", "draftdag-D4"]:
        raise RuntimeError(f"D5 layer merge order is invalid: {group_ids}")
    draw(root, merged / "final.svg")
    finish_node(n5, {"status": "success", "method": "merge independent D1 and D2-D4 SVG branches", "merged_layers": group_ids[-4:], "svg_sha256": sha(merged / "final.svg"), "automatic_visibility_check": "skipped"}, [merged / "final.svg", merged / "draft.png"])

    print("D6: final i2i refinement", flush=True)
    d6 = base / "D6"
    n6 = make_node("D6", [str((merged / "draft.png").resolve()), str((merged / "final.svg").resolve())])
    shutil.copy2(merged / "draft.png", d6 / "draft.png")
    final = generate_final(d6 / "draft.png", base / "final.png")
    shutil.copy2(base / "final.png", d6 / "final.png")
    finish_node(n6, {"status": "success", "method": "i2i", "final": final, "limitation": "Qwen img2img redraws a full image; SVG layer preservation does not guarantee pixel-level preservation in final rendering."}, [d6 / "draft.png", d6 / "final.png", base / "final.png"])


def dag() -> None:
    from png_cos_dag import run
    run()


def judge(image_path: Path) -> dict:
    cfg = LLMConfig.from_env()
    model = os.environ.get("DRAFTDAG_JUDGE_MODEL", "qwen-vl-plus")
    cfg.model = model
    cfg.timeout = 180
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    criteria = "\n".join(f"{key}: {desc}" for key, desc in CHECKLIST)
    request = ("Inspect ONLY this image. For each criterion, answer true only if visibly supported; "
               "if missing, ambiguous, too small to identify, or merely implied, answer false. "
               "Judge relations/actions strictly. Do not infer from an expected prompt or a structural draft. "
               "Return JSON only: {\"checks\":[{\"id\":...,\"satisfied\":true/false,\"reason\":...}]} "
               "with exactly these IDs, in order:\n" + criteria)
    response = LLMClient(cfg).create(messages=[{"role": "user", "content": [
        {"type": "text", "text": request},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
    ]}], model=model)
    content = response["choices"][0]["message"]["content"]
    obj = _extract_json_object(content)
    checks = obj.get("checks")
    if not isinstance(checks, list) or [x.get("id") for x in checks] != [x[0] for x in CHECKLIST]:
        raise ValueError("Judge omitted or reordered checklist entries")
    if any(type(x.get("satisfied")) is not bool or not str(x.get("reason", "")).strip() for x in checks):
        raise ValueError("Judge returned invalid evidence")
    return {"model": model, "image": str(image_path.resolve()), "checks": checks, "satisfied": sum(x["satisfied"] for x in checks), "total": len(CHECKLIST)}


def compare(dag_image: Path | None = None, output_path: Path | None = None) -> None:
    baseline_image = HERE / "baseline" / "final.png"
    dag_image = dag_image or HERE / "cos_png_dag" / "final.png"
    output_path = output_path or HERE / "comparison_cos_png_dag.json"
    if not baseline_image.is_file() or not dag_image.is_file():
        raise FileNotFoundError("Both final images are required before comparison")
    print("compare: blind same-checklist VLM review of baseline", flush=True)
    a = judge(baseline_image)
    print("compare: blind same-checklist VLM review of DAG", flush=True)
    b = judge(dag_image)
    rows = [{"id": key, "criterion": desc, "baseline": a["checks"][i], "draftdag": b["checks"][i]}
            for i, (key, desc) in enumerate(CHECKLIST)]
    write_json(output_path, {"prompt_sha256": PROMPT_SHA256, "method": "same VLM checklist, each final image evaluated independently, uncertain counted false", "baseline": {"satisfied": a["satisfied"], "total": a["total"], "image": a["image"]}, "draftdag": {"satisfied": b["satisfied"], "total": b["total"], "image": b["image"]}, "judge_model": a["model"], "per_constraint": rows, "limitations": ["One prompt, one stochastic generation per arm, one VLM evaluation per image; not a statistical result.", "VLM judgments may be wrong; inspect image and reasons manually.", "DAG uses extra LLM calls and hand-chosen coordinates; any gain is not attributable solely to DAG structure.", "The final Qwen img2img stage redraws the whole canvas and may remove draft constraints."]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "dag", "dag-plan", "compare", "legacy-dag", "legacy-refine", "legacy-compare", "all"])
    args = parser.parse_args()
    initialize()
    if args.stage in {"baseline", "all"}:
        baseline()
    if args.stage in {"dag", "all"}:
        dag()
    if args.stage == "dag-plan":
        from cos_dag import write_graph
        print(write_graph())
    if args.stage == "legacy-dag":
        from branch_dag import run_branch_dag
        run_branch_dag()
    if args.stage == "legacy-refine":
        from branch_dag import refine_existing
        refine_existing()
    if args.stage in {"compare", "all"}:
        compare()
    if args.stage == "legacy-compare":
        compare(HERE / "dag" / "final.png", HERE / "comparison_legacy_rerun.json")


if __name__ == "__main__":
    main()
