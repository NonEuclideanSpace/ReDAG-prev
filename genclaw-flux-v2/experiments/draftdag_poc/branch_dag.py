"""Four independent visual branches followed by multi-reference diffusion merges."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil

from PIL import Image

import run as poc
from cc_genclaw.tools.code_scene_draft import CodeSceneDraftTool
from cc_genclaw.tools.i2i import I2ITool


BRANCHES = {
    "object_branch": {
        "scope": "Props and objects: ball, hose, bicycle and basket, truck, drone",
        "constraints": ["red tennis ball", "garden hose with nozzle", "bicycle with wicker basket of wilted daisies", "ivy-covered pickup truck", "small drone"],
        "draft_prompt": "1600x900 landscape structural SVG draft, clean object reference study. Show ONLY a red tennis ball, green garden hose with nozzle, bicycle with a wicker basket containing visibly wilted daisies, ivy-covered pickup truck with its front hood clearly recognizable, and a drone. Give each object its own visible space; keep a neutral pale background. No people, animals, trees, fence, ladybugs or spatial interactions.",
        "render_prompt": "Render this object sketch as a photorealistic reference image of the key scene props. Preserve red tennis ball, coiled garden hose and nozzle, a bicycle with wicker basket holding wilted daisies, an ivy-covered pickup truck with visible front hood, and a small drone. Keep objects identifiable and separate. Do not add people or animals.",
    },
    "environment_branch": {
        "scope": "Static landscape, composition and light",
        "constraints": ["sun-dappled meadow", "ancient oak on right with low branch", "rusted fence behind left side", "distant clock tower", "partly cloudy sky"],
        "draft_prompt": "1600x900 landscape structural SVG draft of a single rural meadow. Sun-dappled grassy ground, partly cloudy sky, a large ancient oak on the RIGHT with a low branch reaching toward the center, rusted fence across the LEFT background, and a small distant clock tower. Leave open foreground space on left for a woman and dogs, on right for a truck. Draw no people, animals, bicycle, ball, hose, drone or ladybugs.",
        "render_prompt": "Render this structural landscape sketch as a photorealistic meadow background. Ancient oak with low branch on the right, rusty fence behind the left foreground, small distant clock tower, grass, partly cloudy sun-dappled sky. Preserve open foreground space for later characters, props and relations. No people, animals or vehicles yet.",
    },
    "relation_branch": {
        "scope": "Interactions and spatial relations with minimal endpoint context",
        "constraints": ["cat on truck hood", "terrier digging at oak base", "woman adjusting hose with curved water arc", "bicycle leaning against fence", "ladybugs bridging low oak branch toward red tennis ball", "drone above scene"],
        "draft_prompt": "1600x900 landscape structural SVG draft focused on INTERACTIONS, not a complete scene. On the left a woman bends with hand visibly adjusting a hose nozzle and a curved arc of water; a bicycle touches and leans against a short fence segment. On the right a small terrier actively digs soil at the oak trunk base, a cat sits with paws ON the hood of a pickup truck, and a continuous red-and-black ladybug chain visibly starts at a LOW oak branch and reaches toward a red tennis ball. Put a small drone above. Use only enough endpoint objects to make these exact relations clear. Plain sparse background, no extra subjects.",
        "render_prompt": "Photorealistic study of the exact relations in this sketch. Cat's paws ON pickup truck hood; terrier actively DIGGING dirt at oak base; woman touching garden hose nozzle with curved water arc; bicycle LEANING against fence; ladybugs forming a continuous bridge STARTING on a low oak branch TOWARD red ball; drone above. Preserve these spatial interactions, with only minimal scene context.",
    },
}

CHARACTERS = {
    "woman": {
        "scope": "One woman in a straw hat and floral sundress",
        "constraints": ["woman", "straw hat", "floral sundress"],
        "draft_prompt": "1600x900 structural SVG draft of ONLY ONE woman, full body, straw hat and floral sundress, bending slightly. One person, no animals or props. Leave a little grassy ground and pale sky around her. No other figures, no panels or text.",
        "render_prompt": "Photorealistic full-body image of exactly ONE woman in a straw hat and floral sundress in a sunny grassy meadow, bending slightly. Natural meadow background consistent with a rural oak landscape. No other people, animals or props.",
    },
    "retriever": {
        "scope": "One golden retriever",
        "constraints": ["golden retriever", "running pose"],
        "draft_prompt": "1600x900 structural SVG draft of ONLY ONE golden retriever dog running, recognizable large floppy ears and golden coat. Grass and pale sky only. No other animals, people, props, panels or text.",
        "render_prompt": "Photorealistic image of exactly ONE golden retriever in a running pose on a sunlit grassy meadow. Recognizable golden coat. Natural meadow background. No other animals, people or props.",
    },
    "terrier": {
        "scope": "One small terrier in leather harness",
        "constraints": ["small terrier", "leather harness", "digging pose"],
        "draft_prompt": "1600x900 structural SVG draft of ONLY ONE small terrier wearing a visible brown leather harness, low to the ground in a digging pose. Grass and pale sky only. No oak, other animals, people, props, panels or text.",
        "render_prompt": "Photorealistic image of exactly ONE small terrier wearing a visible leather harness on a grassy meadow, front paws in a digging pose. Natural outdoor background. No other animals or people.",
    },
    "cat": {
        "scope": "One cat",
        "constraints": ["cat", "perched pose"],
        "draft_prompt": "1600x900 structural SVG draft of ONLY ONE recognizable cat sitting in a perched pose, paws clearly visible. Grass and pale sky only. No truck or other animals, people, props, panels or text.",
        "render_prompt": "Photorealistic image of exactly ONE cat in a perched pose on a sunny rural meadow, paws clearly visible. Natural outdoor background. No other animals or people, no truck.",
    },
}

CHARACTER_NODES = [f"character_{name}" for name in CHARACTERS]

MERGES = [
    ("merge_1", ["environment_branch", "object_branch"], "Merge static environment and key objects. Reference A is the EXISTING photoreal meadow and must remain the main composition. Reference B is its structural sketch. Reference C is the object sketch; Reference D is the photoreal object render. Add the red ball, garden hose, bicycle with wicker flower basket, ivy-covered truck near oak, and drone into ONE coherent meadow scene. Preserve oak on right, fence on left and distant clock tower. Do not create a collage or separate panels."),
    ("merge_3", ["merge_2", "relation_branch"], "Continue the existing scene in Reference A. Reference B is the relation sketch; Reference C is the rendered relation study. Correct the interactions locally: cat PERCHED ON truck hood, terrier ACTIVELY DIGGING at oak base, woman ADJUSTING hose with clear curved water arc, bicycle LEANING against fence, ladybugs forming a visible bridge FROM low oak branch TOWARD red ball. Keep the existing environment and subjects. One coherent scene, no panels."),
]


def node_dir(name: str) -> Path:
    return poc.HERE / "dag" / name


def record_node(name: str, node_type: str, deps: list[str], scope: str, inputs: list[Path], instruction: str, outputs: list[Path], result: dict) -> None:
    folder = node_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
    poc.write_json(folder / "node.json", {
        "node_id": name,
        "node_type": node_type,
        "depends_on": deps,
        "target_scope": scope,
        "input_artifacts": [str(p.resolve()) for p in inputs],
        "output_artifacts": [str(p.resolve()) for p in outputs],
        "prompt_sha256": poc.PROMPT_SHA256,
    })
    poc.write_json(folder / "result.json", result)


def character_crop_box(layout: dict, canvas: tuple[int, int]) -> tuple[int, int, int, int]:
    """Bound the character parts, excluding layout-wide sky/grass rectangles."""
    canvas_w, canvas_h = canvas
    parts = []
    for item in layout.get("compositional_deconstruction", {}).get("elements", []):
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x0, y0, x1, y1 = box
        if not all(isinstance(value, (int, float)) for value in box):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        description = str(item.get("desc", "")).lower()
        if (x1 - x0) > canvas_w * 0.9 or any(word in description for word in ("grass band", "background", "sky", "ground strip")):
            continue
        parts.append((x0, y0, x1, y1))
    if not parts:
        raise ValueError("Character layout contains no usable foreground bboxes")
    left = max(0, min(p[0] for p in parts))
    top = max(0, min(p[1] for p in parts))
    right = min(canvas_w, max(p[2] for p in parts))
    bottom = min(canvas_h, max(p[3] for p in parts))
    subject_w, subject_h = right - left, bottom - top
    if subject_w <= 0 or subject_h <= 0:
        raise ValueError("Invalid character foreground bbox")

    # Keep context around limbs, while avoiding tiny or extreme-aspect model inputs.
    # Qwen-Image-2.1 rounds non-32 dimensions, so use 32-pixel local canvases.
    width = min((canvas_w // 32) * 32, 32 * math.ceil(max(256, subject_w * 1.24, subject_h * 0.62) / 32))
    height = min((canvas_h // 32) * 32, 32 * math.ceil(max(256, subject_h * 1.16, subject_w * 0.62) / 32))
    if width == canvas_w and height == canvas_h:
        raise ValueError("Character crop would still be full-scene size")
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    x0 = min(canvas_w - width, max(0, round(center_x - width / 2)))
    y0 = min(canvas_h - height, max(0, round(center_y - height / 2)))
    return x0, y0, x0 + width, y0 + height


def crop_character_draft(folder: Path) -> dict:
    draft = folder / "draft.png"
    layout = json.loads((folder / "layout.json").read_text(encoding="utf-8"))
    with Image.open(draft) as source:
        canvas = source.size
        box = character_crop_box(layout, canvas)
        # Retain the complete generator output for provenance, but draft.png is local.
        shutil.copy2(draft, folder / "full_draft.png")
        source.crop(box).save(draft)
    metadata = {"method": "union_of_foreground_layout_bboxes_with_context", "source_canvas": list(canvas),
                "crop_box_xyxy": list(box), "local_size": [box[2] - box[0], box[3] - box[1]],
                "full_draft": str((folder / "full_draft.png").resolve())}
    poc.write_json(folder / "crop.json", metadata)
    return metadata


def generate_branch_draft(name: str, spec: dict) -> None:
    folder = node_dir(name)
    folder.mkdir(parents=True, exist_ok=True)
    payload = poc.tool_json(CodeSceneDraftTool().execute({"prompt": spec["draft_prompt"], "self_review": False}))
    poc.copy_draft(Path(payload["draft_png_path"]), folder)
    if name in CHARACTER_NODES:
        crop_character_draft(folder)
    if not (folder / "draft.png").is_file():
        raise RuntimeError(f"{name}: draft missing")
    print(f"{name}: draft ready", flush=True)


def generate_with_refs(prompt: str, refs: list[Path], destination: Path, output_size: tuple[int, int] | None = None) -> dict:
    if len(refs) > 8:
        raise ValueError("Worker supports at most eight direct reference images")
    if not refs or any(not p.is_file() for p in refs):
        raise FileNotFoundError("A required visual reference is missing")
    original_size = (os.environ.get("QWEN_IMAGE_WIDTH"), os.environ.get("QWEN_IMAGE_HEIGHT"))
    try:
        if output_size is not None:
            os.environ["QWEN_IMAGE_WIDTH"], os.environ["QWEN_IMAGE_HEIGHT"] = map(str, output_size)
        payload = poc.tool_json(I2ITool().execute({
            "image_path": str(refs[0].resolve()),
            "reference_image_paths": [str(p.resolve()) for p in refs[1:]],
            "prompt": prompt,
        }))
    finally:
        for key, value in zip(("QWEN_IMAGE_WIDTH", "QWEN_IMAGE_HEIGHT"), original_size):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    source = Path(payload["final_path"])
    metadata_source = source.with_suffix(".json")
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    if metadata.get("backend") != "qwen-image" or metadata.get("mode") != "img2img":
        raise RuntimeError("Expected real Qwen-Image-2.1 img2img output")
    if metadata.get("reference_count") != len(refs):
        raise RuntimeError("Worker did not account for all visual references")
    if metadata.get("condition_image_count") != len(refs):
        raise RuntimeError("Worker did not confirm conditioning on every visual reference")
    if output_size is not None:
        with Image.open(source) as rendered:
            if rendered.size != output_size:
                raise RuntimeError(f"Local partial size {rendered.size} differs from character draft {output_size}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    shutil.copy2(metadata_source, destination.with_suffix(".json"))
    return {"status": "success", "backend": metadata["backend"], "mode": metadata["mode"], "reference_count": len(refs), "reference_order": [str(p.resolve()) for p in refs], "image_sha256": poc.sha(destination), "source_image": str(source)}


def existing_generation(destination: Path, refs: list[Path]) -> dict:
    metadata = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("backend") != "qwen-image" or metadata.get("condition_image_count") != len(refs):
        raise RuntimeError(f"Cannot resume incompatible generation: {destination}")
    return {"status": "success", "backend": "qwen-image", "mode": metadata.get("mode"), "reference_count": len(refs), "reference_order": [str(p.resolve()) for p in refs], "image_sha256": poc.sha(destination), "source_image": str(destination), "reused_from_existing": True}


def refine_existing() -> None:
    """Consume every branch and merge image in bounded direct-reference rounds."""
    base = poc.HERE / "dag"
    final = base / "final.png"
    if final.exists():
        raise FileExistsError("Final image already exists; preserve it before refining again")
    character_drafts = [node_dir(name) / "draft.png" for name in CHARACTER_NODES]
    character_partials = [node_dir(name) / "partial.png" for name in CHARACTER_NODES]
    other_drafts = [node_dir(name) / "draft.png" for name in BRANCHES]
    other_partials = [node_dir(name) / "partial.png" for name in BRANCHES]
    merge_names = ["merge_1", "merge_2_woman", "merge_2_retriever", "merge_2_terrier", "merge_2", "merge_3"]
    merges = [node_dir(name) / "partial.png" for name in merge_names]
    folder = node_dir("final_refinement")
    folder.mkdir(parents=True, exist_ok=True)
    groups = [
        ("round_1", character_drafts[:2] + character_partials[:2], "Use the woman and retriever drafts and rendered partials to correct identity. Keep Reference A as ONE meadow scene."),
        ("round_2", character_drafts[2:] + character_partials[2:], "Use the terrier and cat drafts and rendered partials to correct identity. Keep Reference A as ONE meadow scene."),
        ("round_3", other_drafts, "Use the object, environment and relation sketches to recover scene props, landscape and interactions."),
        ("round_4", other_partials, "Use the object, environment and relation rendered partials to recover their visual details."),
        ("round_5", merges[:3], "Compare the environment/object merge and the first two character merges to retain the established composition."),
        ("final", merges[3:], "Compare the remaining character merges and relation merge; retain all subjects in ONE final scene."),
    ]
    previous = node_dir("merge_3") / "partial.png"
    results = []
    outputs = []
    instructions = []
    for step, sources, instruction in groups:
        print(f"final_refinement: {step}, {1 + len(sources)} direct references", flush=True)
        destination = final if step == "final" else folder / f"{step}.png"
        prompt = (instruction + " Reference A is the current merged scene and must remain one continuous picture. "
                  "Other references are studies only; do not copy their backgrounds or make panels. "
                  "Keep cat on truck hood, terrier digging at oak base, woman adjusting hose, and ladybugs bridging branch to ball. " + poc.PROMPT)
        refs = [previous, *sources]
        if destination.is_file() and destination.with_suffix(".json").is_file():
            results.append(existing_generation(destination, refs))
        else:
            results.append(generate_with_refs(prompt, refs, destination))
        instructions.append(prompt)
        outputs.extend([destination, destination.with_suffix(".json")])
        previous = destination

    inputs = character_drafts + character_partials + other_drafts + other_partials + merges
    limitation = "All upstream images are consumed across six Qwen calls with at most five references. Seven references at 1024x576 exceeded 47 GiB VRAM in an earlier attempt. Qwen-Image-2.1 regenerates output from noise; strength is not applied."
    result = {"status": "success", "method": "six direct-reference Qwen refinement rounds", "rounds": results, "final_sha256": poc.sha(final), "limitation": limitation}
    record_node("final_refinement", "final", [*merge_names, "character_branch", *BRANCHES], "One coherent scene conditioned on all character, other branch and merge images", inputs, "\n\n".join(instructions), outputs, result)
    poc.write_json(base / "artifacts.json", {"prompt_sha256": poc.PROMPT_SHA256, "branches": ["character_branch", *BRANCHES], "merges": merge_names, "final": str(final.resolve()), "final_sha256": poc.sha(final), "limitation": limitation})


def run_branch_dag() -> None:
    base = poc.HERE / "dag"
    if (base / "final.png").exists():
        raise FileExistsError("DAG final already exists; preserve existing experiment before rerunning")
    poc.fresh_run_dir("dag")
    character_specs = {f"character_{name}": spec for name, spec in CHARACTERS.items()}
    all_specs = {**BRANCHES, **character_specs}
    character_merge_names = ["merge_2_woman", "merge_2_retriever", "merge_2_terrier", "merge_2"]
    poc.write_json(base / "dag.json", {
        "prompt_sha256": poc.PROMPT_SHA256,
        "nodes": {name: {"node_type": "branch", "depends_on": []} for name in all_specs} | {
            "character_branch": {"node_type": "branch_group", "depends_on": CHARACTER_NODES},
            "merge_1": {"node_type": "merge", "depends_on": ["environment_branch", "object_branch"]},
            "merge_2_woman": {"node_type": "merge", "depends_on": ["merge_1", "character_woman"]},
            "merge_2_retriever": {"node_type": "merge", "depends_on": ["merge_2_woman", "character_retriever"]},
            "merge_2_terrier": {"node_type": "merge", "depends_on": ["merge_2_retriever", "character_terrier"]},
            "merge_2": {"node_type": "merge", "depends_on": ["merge_2_terrier", "character_cat"]},
            "merge_3": {"node_type": "merge", "depends_on": ["merge_2", "relation_branch"]},
            "final_refinement": {"node_type": "final", "depends_on": ["merge_1", *character_merge_names, "merge_3", "character_branch", *BRANCHES]},
        },
        "execution_note": "Seven independent drafts run concurrently. Four character drafts are cropped to foreground layout bboxes; each character diffusion partial keeps its own local size. Scene merges remain full size.",
    })

    pending_drafts = {name: spec for name, spec in all_specs.items()
                      if not all((node_dir(name) / filename).is_file() for filename in
                                 (("draft.png", "layout.json", "crop.json", "full_draft.png")
                                  if name in CHARACTER_NODES else ("draft.png", "layout.json")))}
    print(f"branches: generating {len(pending_drafts)} missing independent drafts", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {pool.submit(generate_branch_draft, name, spec): name for name, spec in pending_drafts.items()}
        for future in as_completed(tasks):
            future.result()

    for name, spec in all_specs.items():
        print(f"{name}: diffusion partial", flush=True)
        folder = node_dir(name)
        draft = folder / "draft.png"
        partial = folder / "partial.png"
        local_size = None
        if name in CHARACTER_NODES:
            with Image.open(draft) as image:
                local_size = image.size
        if partial.is_file() and partial.with_suffix(".json").is_file():
            result = existing_generation(partial, [draft])
            if local_size is not None:
                with Image.open(partial) as existing_partial:
                    if existing_partial.size != local_size:
                        raise RuntimeError(f"Cannot reuse partial {partial}: size does not match local draft")
        else:
            result = generate_with_refs(spec["render_prompt"], [draft], partial, output_size=local_size)
        result["draft_sha256"] = poc.sha(draft)
        result["constraints"] = spec["constraints"]
        outputs = [draft, partial, partial.with_suffix(".json")]
        if local_size is not None:
            result["crop"] = json.loads((folder / "crop.json").read_text(encoding="utf-8"))
            result["partial_size"] = list(local_size)
            outputs.extend([folder / "full_draft.png", folder / "crop.json"])
        record_node(name, "branch", [], spec["scope"], [poc.PROMPT_PATH.resolve(), (poc.HERE / "constraints.json").resolve()], spec["draft_prompt"] + "\n\nDiffusion: " + spec["render_prompt"], outputs, result)

    character_images = [node_dir(name) / kind for name in CHARACTER_NODES for kind in ("draft.png", "partial.png")]
    record_node("character_branch", "branch_group", CHARACTER_NODES, "Four separate character studies", character_images,
                "Aggregate the four independent character draft and Qwen partial image pairs; no combined character image is generated.", character_images,
                {"status": "success", "character_images": {name: {"draft": str((node_dir(f"character_{name}") / "draft.png").resolve()), "partial": str((node_dir(f"character_{name}") / "partial.png").resolve())} for name in CHARACTERS}})

    name, deps, instruction = MERGES[0]
    refs = [node_dir("environment_branch") / "partial.png", node_dir("environment_branch") / "draft.png", node_dir("object_branch") / "draft.png", node_dir("object_branch") / "partial.png"]
    print("merge_1: environment + objects", flush=True)
    partial = node_dir(name) / "partial.png"
    result = existing_generation(partial, refs) if partial.is_file() and partial.with_suffix(".json").is_file() else generate_with_refs(instruction, refs, partial)
    record_node(name, "merge", deps, "Environment plus objects", refs, instruction, [partial, partial.with_suffix(".json")], result)

    previous_name = "merge_1"
    for index, character in enumerate(CHARACTERS):
        name = character_merge_names[index]
        subject = f"character_{character}"
        refs = [node_dir(previous_name) / "partial.png", node_dir(subject) / "draft.png", node_dir(subject) / "partial.png"]
        instruction = (f"Continue Reference A as one meadow scene. References B and C show ONLY the {character} as sketch and photoreal render. "
                       f"Add exactly one {character} in the scene, preserving all existing environment and object pixels as much as the model allows. "
                       "No white cutout background, panels, duplicate characters or collage. "
                       "Woman belongs on left by hose; retriever chases red ball; terrier digs at oak base; cat sits on truck hood.")
        print(f"{name}: add only {character}", flush=True)
        partial = node_dir(name) / "partial.png"
        result = existing_generation(partial, refs) if partial.is_file() and partial.with_suffix(".json").is_file() else generate_with_refs(instruction, refs, partial)
        record_node(name, "merge", [previous_name, subject], f"Add {character} identity to existing scene", refs, instruction, [partial, partial.with_suffix(".json")], result)
        previous_name = name

    name, _, instruction = MERGES[1]
    refs = [node_dir("merge_2") / "partial.png", node_dir("relation_branch") / "draft.png", node_dir("relation_branch") / "partial.png"]
    print("merge_3: relations", flush=True)
    partial = node_dir(name) / "partial.png"
    result = existing_generation(partial, refs) if partial.is_file() and partial.with_suffix(".json").is_file() else generate_with_refs(instruction, refs, partial)
    record_node(name, "merge", ["merge_2", "relation_branch"], "Correct spatial and action relations", refs, instruction, [partial, partial.with_suffix(".json")], result)

    refine_existing()
