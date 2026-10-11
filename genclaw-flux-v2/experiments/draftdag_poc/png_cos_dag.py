"""Executable PNG-reference approximation of a subject/relation CoS DAG.

Only local units and local relationships are rendered before a *single*
full-scene diffusion call. This is not HiCoGen's clean-latent synthesis.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

import run as poc
from branch_dag import CHARACTERS, crop_character_draft, existing_generation, generate_with_refs
from cc_genclaw.tools.code_scene_draft import CodeSceneDraftTool


ROOT = poc.HERE / "cos_png_dag"
CANVAS = (1024, 576)
LOCAL = (640, 640)

ENVIRONMENT = {
    "draft": "1600x900 single continuous structural SVG landscape. Sun-dappled meadow, ancient oak on RIGHT with one low branch pointing toward the middle, rusted fence in LEFT background, tiny clock tower in distance, small drone in upper sky. Keep center, left foreground and truck area near oak relatively empty for separately generated local relationship units. No people, dogs, cat, truck, bicycle, ball, hose or ladybug bridge. No panels, labels or text.",
    "render": "Reference A is a structural layout. Photorealistic meadow with oak on RIGHT, low branch reaching center, rusted fence LEFT background, clock tower far away and small drone in sky. Preserve open space for later local relationship units. One landscape, no panels.",
}

RELATIONS = {
    "woman_hose": {
        "depends_on": ["woman"],
        "draft": "1600x900 SVG LOCAL relationship study on a plain near-white background. Center one woman wearing straw hat and floral sundress, bending, her HAND ON the garden hose nozzle; draw one obvious blue CURVED water arc. Only this woman, hose, and water. No other objects, no landscape, no panels or text.",
        "render": "Reference A gives the exact local action sketch; Reference B gives the woman's identity. Render one photoreal woman in straw hat and floral sundress visibly adjusting a hose nozzle while a curved water arc flows from it. Plain unobtrusive outdoor background. Keep woman, hand, nozzle, and arc in one coherent local image, no collage.",
    },
    "bicycle_fence": {
        "depends_on": [],
        "draft": "1600x900 SVG LOCAL relationship study centered on a plain near-white background. One bicycle visibly LEANING AGAINST a short rusted fence segment; the bicycle carries a wicker basket CONTAINING wilted daisies. Show contact between bicycle and fence. No person, truck, oak or other subjects; no panels or text.",
        "render": "Reference A is the exact local sketch. Photoreal bicycle leaning against rusted fence, with wicker basket containing visibly wilted daisies on the bicycle. One continuous local scene, no additional subjects or panels.",
    },
    "retriever_ball": {
        "depends_on": ["retriever"],
        "draft": "1600x900 SVG LOCAL relationship study on plain near-white background. One recognizable golden retriever gallops toward ONE small RED tennis ball immediately ahead of it. Show chasing direction and keep ball separate from dog's body. No other dog, tree, person, truck, panels or text.",
        "render": "Reference A shows chasing composition; Reference B shows the golden retriever identity. Render ONE photoreal running golden retriever chasing a distinct small red tennis ball immediately ahead. Simple meadow ground only, no other animals or panels.",
    },
    "terrier_oak": {
        "depends_on": ["terrier"],
        "draft": "1600x900 SVG LOCAL relationship study on plain near-white background. On the right, base of one large oak trunk and a LOW branch; at the base, ONE SMALL terrier in a leather harness actively DIGS with front paws into a visible dirt hole. Make paw-to-soil contact unmistakable. No retriever, cat, truck, panels or text.",
        "render": "Reference A shows exact dog-to-oak interaction; Reference B shows terrier identity. Photoreal small terrier with leather harness actively digging a dirt hole at the base of an ancient oak. Local close composition; no other animals or panels.",
    },
    "cat_truck": {
        "depends_on": ["cat"],
        "draft": "1600x900 SVG LOCAL relationship study on plain near-white background. One ivy-covered pickup truck with front HOOD clearly visible; ONE cat perched directly ON TOP of the hood, all four paws touching the hood. Cat not beside the truck, not on roof. No dog, person, panels or text.",
        "render": "Reference A gives precise cat-on-hood composition; Reference B gives cat identity. Photoreal cat with paws PERCHED ON the front hood of an ivy-covered pickup truck. Show truck and contact clearly in one local image; no panels or other animals.",
    },
    "ladybug_bridge": {
        "depends_on": ["retriever_ball", "terrier_oak"],
        "draft": "1600x900 SVG LOCAL relationship diagram on plain near-white background. Left: one low oak BRANCH. Right-lower: ONE small RED tennis ball. A continuous, dense BRIDGE made from many recognizable red-and-black LADYBUGS runs from the low branch TOWARD the ball; both endpoints visible. No dog, truck, person, panels or text.",
        "render": "Reference A is the precise bridge diagram; Reference B shows the red ball; Reference C shows the oak. Render a visually continuous bridge of many recognizable red-and-black ladybugs extending FROM the low oak branch TOWARD a small red tennis ball. Focus ONLY on this unusual relationship; no dogs, truck, people or panels.",
    },
}

# Coordinates are scene-space anchor slots, not per-step whole-scene additions.
PLACEMENTS = {
    "bicycle_fence": (55, 305, 385, 555),
    "woman_hose": (105, 225, 460, 545),
    "retriever_ball": (365, 345, 690, 550),
    "terrier_oak": (665, 300, 860, 550),
    "cat_truck": (750, 285, 1020, 520),
    "ladybug_bridge": (620, 200, 805, 455),
}


ASSEMBLY_INPUTS = [
    "environment/draft.png", "environment/partial.png",
    "bicycle_fence/draft.png", "bicycle_fence/partial.png",
    "woman_hose/draft.png", "woman_hose/partial.png",
    "retriever/partial.png", "retriever_ball/draft.png",
    "terrier_oak/draft.png", "terrier_oak/partial.png",
    "cat_truck/draft.png", "cat_truck/partial.png",
    "ladybug_bridge/draft.png", "ladybug_bridge/partial.png",
]


def folder(name: str) -> Path:
    return ROOT / "nodes" / name


def write_node(name: str, kind: str, deps: list[str], instruction: str,
               inputs: list[Path], outputs: list[Path], result: dict) -> None:
    target = folder(name)
    target.mkdir(parents=True, exist_ok=True)
    (target / "instruction.txt").write_text(instruction + "\n", encoding="utf-8")
    poc.write_json(target / "node.json", {
        "node_id": name, "node_type": kind, "depends_on": deps,
        "target_scope": instruction, "input_artifacts": [str(p.resolve()) for p in inputs],
        "output_artifacts": [str(p.resolve()) for p in outputs],
        "prompt_sha256": poc.PROMPT_SHA256,
    })
    poc.write_json(target / "result.json", result)


def draft(name: str, instruction: str, crop: bool) -> Path:
    target = folder(name)
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "draft.png"
    if destination.is_file() and (target / "layout.json").is_file():
        if crop and not (target / "crop.json").is_file():
            crop_character_draft(target)
        if crop:
            fit_local_draft(target)
        return destination
    payload = poc.tool_json(CodeSceneDraftTool().execute({"prompt": instruction, "self_review": False}))
    poc.copy_draft(Path(payload["draft_png_path"]), target)
    if crop:
        crop_character_draft(target)
        fit_local_draft(target)
    return destination


def fit_local_draft(target: Path) -> None:
    """Bound local diffusion cost without changing the scene-level canvas."""
    path = target / "draft.png"
    with Image.open(path) as source:
        width, height = source.size
        if max(width, height) > 768 or width * height > 500_000:
            scale = min(768 / max(width, height), math.sqrt(500_000 / (width * height)))
            new_size = (max(64, 32 * round(width * scale / 32)),
                        max(64, 32 * round(height * scale / 32)))
            source.resize(new_size, Image.Resampling.LANCZOS).save(path)
    crop_path = target / "crop.json"
    metadata = json.loads(crop_path.read_text(encoding="utf-8"))
    with Image.open(path) as fitted:
        metadata["diffusion_size"] = list(fitted.size)
    poc.write_json(crop_path, metadata)


def render(name: str, instruction: str, refs: list[Path], local: bool) -> dict:
    target = folder(name) / "partial.png"
    if target.is_file() and target.with_suffix(".json").is_file():
        result = existing_generation(target, refs)
    else:
        size = None
        if local:
            with Image.open(refs[0]) as image:
                size = image.size
        result = generate_with_refs(instruction, refs, target, output_size=size)
    if local:
        with Image.open(refs[0]) as first, Image.open(target) as generated:
            if first.size != generated.size:
                raise RuntimeError(f"{name}: local draft and partial sizes differ")
    return result


def foreground(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Soft background removal for flat SVG sketches; not applied to photos."""
    rgb = ImageOps.contain(source.convert("RGB"), size)
    background = rgb.getpixel((0, 0))
    flat = Image.new("RGB", rgb.size, background)
    diff = ImageChops.difference(rgb, flat).convert("L")
    mask = diff.point(lambda value: min(255, max(0, (value - 24) * 5)))
    mask = mask.filter(ImageFilter.GaussianBlur(1))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def paste_study(canvas: Image.Image, source_path: Path, box: tuple[int, int, int, int]) -> None:
    with Image.open(source_path) as source:
        fitted = foreground(source, (box[2] - box[0], box[3] - box[1]))
    x = box[0] + (box[2] - box[0] - fitted.width) // 2
    y = box[1] + (box[3] - box[1] - fitted.height) // 2
    canvas.alpha_composite(fitted, (x, y))


def assembly_draft() -> Path:
    target = folder("scene_assembly")
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "draft.png"
    if destination.is_file():
        return destination
    with Image.open(folder("environment") / "draft.png") as source:
        canvas = source.convert("RGBA").resize(CANVAS, Image.Resampling.LANCZOS)
    for name in ("bicycle_fence", "woman_hose", "retriever_ball", "terrier_oak", "cat_truck", "ladybug_bridge"):
        paste_study(canvas, folder(name) / "draft.png", PLACEMENTS[name])
    canvas.convert("RGB").save(destination)
    poc.write_json(target / "placements.json", {
        "canvas": list(CANVAS), "boxes_xyxy": PLACEMENTS,
        "method": "Local SVG drafts form the structure; local rendered PNGs are spatially alpha/feather composited into a photographic reference.",
        "photo_sources": {
            "background": "environment/partial.png",
            "bicycle_fence": "bicycle_fence/partial.png",
            "woman_hose": "woman_hose/partial.png",
            "retriever_ball": "retriever/partial.png plus a red ball painted at x=640..662,y=443..465; retriever_ball/partial.png was rejected because it remained a silhouette",
            "terrier_oak": "terrier_oak/partial.png",
            "cat_truck": "cat_truck/partial.png",
            "ladybug_bridge": "ladybug_bridge/partial.png",
        },
    })
    return destination


def paste_render(canvas: Image.Image, source_path: Path, box: tuple[int, int, int, int],
                 feather_photo: bool = False) -> None:
    with Image.open(source_path) as source:
        fitted = ImageOps.contain(source.convert("RGB"), (box[2] - box[0], box[3] - box[1]))
    if feather_photo:
        mask = Image.new("L", fitted.size, 0)
        inset = max(3, min(fitted.size) // 10)
        ImageDraw.Draw(mask).ellipse((inset, inset, fitted.width - inset, fitted.height - inset), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(3, inset // 2)))
        fitted = fitted.convert("RGBA")
        fitted.putalpha(mask)
    else:
        fitted = foreground(fitted, fitted.size)
    x = box[0] + (box[2] - box[0] - fitted.width) // 2
    y = box[1] + (box[3] - box[1] - fitted.height) // 2
    canvas.alpha_composite(fitted, (x, y))


def assembly_photo() -> Path:
    """Reuse local rendered PNG pixels in scene space before one final call."""
    target = folder("scene_assembly")
    target.mkdir(parents=True, exist_ok=True)
    destination = target / "composition.png"
    if destination.is_file():
        return destination
    with Image.open(folder("environment") / "partial.png") as source:
        canvas = source.convert("RGBA").resize(CANVAS, Image.Resampling.LANCZOS)
    for name in ("bicycle_fence", "woman_hose"):
        paste_render(canvas, folder(name) / "partial.png", PLACEMENTS[name])
    # The retriever-ball local render failed to turn its SVG silhouette into a
    # dog; preserve the successful retriever unit plus a red ball placement.
    paste_render(canvas, folder("retriever") / "partial.png", PLACEMENTS["retriever_ball"], True)
    ImageDraw.Draw(canvas).ellipse((640, 443, 662, 465), fill=(215, 25, 18, 255))
    paste_render(canvas, folder("terrier_oak") / "partial.png", PLACEMENTS["terrier_oak"], True)
    paste_render(canvas, folder("cat_truck") / "partial.png", PLACEMENTS["cat_truck"])
    paste_render(canvas, folder("ladybug_bridge") / "partial.png", PLACEMENTS["ladybug_bridge"])
    canvas.convert("RGB").save(destination)
    return destination


def graph() -> dict:
    nodes = {name: {"node_type": "character_unit", "depends_on": [], "output_artifacts": ["draft.png", "partial.png"]}
             for name in CHARACTERS}
    nodes["environment"] = {"node_type": "environment_unit", "depends_on": [], "output_artifacts": ["draft.png", "partial.png"]}
    for name, spec in RELATIONS.items():
        nodes[name] = {"node_type": "local_relation", "depends_on": spec["depends_on"],
                       "output_artifacts": ["draft.png", "partial.png"]}
    nodes["scene_assembly"] = {"node_type": "spatial_composite", "depends_on": ["environment", "bicycle_fence", "woman_hose", "retriever", "retriever_ball", "terrier_oak", "cat_truck", "ladybug_bridge"], "output_artifacts": ["draft.png", "composition.png", "placements.json"]}
    nodes["final_render"] = {"node_type": "single_global_render", "depends_on": ["scene_assembly"], "output_artifacts": ["final.png"]}
    for name, item in nodes.items():
        item["node_id"] = name
        item["input_artifacts"] = [f"{dep}/{'draft.png' if name == 'scene_assembly' else 'partial.png'}" for dep in item["depends_on"]]
        if name == "final_render":
            item["input_artifacts"] = ["scene_assembly/composition.png", "scene_assembly/draft.png"]
        if name == "scene_assembly":
            item["input_artifacts"] = ASSEMBLY_INPUTS
    return {"prompt_sha256": poc.PROMPT_SHA256, "backend": "qwen-image-2.1 PNG-reference approximation",
            "limitation": "No clean-latent reuse or pixel locking; local rendered PNGs are spatially composited, followed by one full-scene diffusion call.",
            "nodes": nodes}


def validate_graph(spec: dict) -> list[str]:
    nodes = spec["nodes"]
    order, active, done = [], set(), set()

    def visit(name: str) -> None:
        if name not in nodes:
            raise ValueError(f"Unknown PNG-DAG node {name}")
        if name in active:
            raise ValueError(f"PNG-DAG cycle at {name}")
        if name in done:
            return
        active.add(name)
        for parent in nodes[name]["depends_on"]:
            visit(parent)
        active.remove(name)
        done.add(name)
        order.append(name)

    for name, value in nodes.items():
        visit(name)
        for artifact in value["input_artifacts"]:
            parent, filename = artifact.split("/", 1)
            if parent not in value["depends_on"] or filename not in nodes[parent]["output_artifacts"]:
                raise ValueError(f"{name} consumes unavailable artifact {artifact}")
    if nodes["final_render"]["input_artifacts"] != ["scene_assembly/composition.png", "scene_assembly/draft.png"]:
        raise ValueError("Final render must consume the spatial photo composition and structural draft")
    if any("scene_assembly" in item["depends_on"] for name, item in nodes.items() if name != "final_render"):
        raise ValueError("Only one full-scene diffusion call is allowed")
    return order


def run() -> None:
    if (ROOT / "final.png").is_file():
        raise FileExistsError("PNG-CoS-DAG final already exists; preserve it before rerunning")
    poc.fresh_run_dir("cos_png_dag")
    specification = graph()
    validate_graph(specification)
    poc.write_json(ROOT / "graph.json", specification)

    # Independent unit sketches can be drafted concurrently. The one GPU
    # worker still serializes actual diffusion calls.
    root_specs = {**{name: spec["draft_prompt"] for name, spec in CHARACTERS.items()},
                  "environment": ENVIRONMENT["draft"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(draft, name, prompt, name != "environment"): name
                   for name, prompt in root_specs.items()}
        for future in as_completed(futures):
            future.result()
    for name in CHARACTERS:
        local_draft = folder(name) / "draft.png"
        result = render(name, CHARACTERS[name]["render_prompt"], [local_draft], True)
        write_node(name, "character_unit", [], CHARACTERS[name]["render_prompt"], [poc.PROMPT_PATH],
                   [local_draft, folder(name) / "partial.png"], result)
        print(f"{name}: local subject ready", flush=True)
    env_draft = folder("environment") / "draft.png"
    result = render("environment", ENVIRONMENT["render"], [env_draft], False)
    write_node("environment", "environment_unit", [], ENVIRONMENT["render"], [poc.PROMPT_PATH],
               [env_draft, folder("environment") / "partial.png"], result)
    print("environment: ready", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(draft, name, spec["draft"], True): name for name, spec in RELATIONS.items()}
        for future in as_completed(futures):
            future.result()
    for name, spec in RELATIONS.items():
        refs = [folder(name) / "draft.png"] + [folder(dep) / "partial.png" for dep in spec["depends_on"]]
        result = render(name, spec["render"], refs, True)
        write_node(name, "local_relation", spec["depends_on"], spec["render"], refs,
                   [folder(name) / "draft.png", folder(name) / "partial.png"], result)
        print(f"{name}: local relation ready", flush=True)

    sketch = assembly_draft()
    photo = assembly_photo()
    assembly_deps = graph()["nodes"]["scene_assembly"]["depends_on"]
    write_node("scene_assembly", "spatial_composite", assembly_deps,
               "Place local sketches and rendered PNGs in fixed scene slots exactly once; no diffusion at this node.",
               [ROOT / "nodes" / item for item in ASSEMBLY_INPUTS],
               [sketch, photo, folder("scene_assembly") / "placements.json"],
               {"status": "success", "method": "deterministic spatial composition of local sketches and rendered PNGs",
                "draft_sha256": poc.sha(sketch), "composition_sha256": poc.sha(photo)})
    refs = [photo, sketch]
    instruction = ("Reference A is an already assembled single-scene photographic composition; preserve its positions and "
                   "relationships while removing cutout seams. Reference B is the corresponding structural layout. "
                   "Do not make a grid, panels, floating pasted cutouts, or giant subjects. Render ONE coherent "
                   "photorealistic scene. Preserve: cat ON truck hood, terrier DIGGING at oak base, woman ADJUSTING "
                   "hose with water arc, bicycle LEANING against fence, ladybug bridge FROM oak branch TOWARD red ball. " + poc.PROMPT)
    print("final_render: one full-scene Qwen call with assembled photo and structural PNG", flush=True)
    final = ROOT / "final.png"
    result = generate_with_refs(instruction, refs, final)
    write_node("final_render", "single_global_render", graph()["nodes"]["final_render"]["depends_on"],
               instruction, refs, [final, final.with_suffix(".json")], result)
    poc.write_json(ROOT / "artifacts.json", {"prompt_sha256": poc.PROMPT_SHA256,
                                              "final": str(final.resolve()), "final_sha256": poc.sha(final),
                                              "limitation": graph()["limitation"]})
