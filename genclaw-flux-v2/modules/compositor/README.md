# Layer Composition prototype

No ComfyUI runtime, installation or workflow is used. The design borrows explicit
layer transforms, stable z-order and normal source-over blending from the
[official compositor](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_compositor.py).
Only Pillow and NumPy are required for composition; Qwen generation reuses
`src/cc_genclaw/tools/_image_gen/qwen_remote.py` and the existing diffusion worker.

## Modules

- `layer.py`: source RGBA, ordinary alpha mask, minimum visible bbox crop,
  uniform scale, clockwise rotation, position, opacity, z-order, source digest.
- `composition.py`: editable layer collection, stable order, rendering and IR API.
- `renderer.py`: linear-light alpha blending, transform rasterization, clipping,
  foreground-only rendering. No diffusion dependency.
- `io.py`: versioned IR, relative source paths, SHA-256 verification, transform
  and crop provenance, exact replay via `SceneComposition.from_ir`.
- `generation.py`: sends **one scene_layout.png** to the existing Qwen worker;
  neither original references nor a contact sheet are sent individually.

Coordinates are integer canvas pixels. `position` anchors the top-left of the
alpha-cropped, scaled asset **before** rotation. Positive angles rotate clockwise
about its center. Equal z-order preserves insertion order. Objects retain native
aspect ratio. White alpha means visible, unlike ComfyUI's inverted optional mask.
Transparent assets are not resized to full canvas size.

## Running

From the repository root, start the existing real model worker (no ComfyUI):

```bash
CUDA_VISIBLE_DEVICES=0 /home/lipz/miniconda3/envs/uno_env/bin/python server/diffusion_server.py \
  --backend qwen-image --port 8190 --model-path /home/hdd4/lpz/Qwen-Image-2.1
```

In another terminal, with the project's existing environment:

```bash
.venv/bin/python examples/multi_reference_compositor_demo.py \
  --reference_dir ./references \
  --layout examples/layouts/night_parking.json \
  --base-url http://127.0.0.1:8190 \
  --prompt "Night parking lot crime scene. A man wearing a black hoodie is grabbing a handbag from a woman wearing a white dress. Another person is running toward them from the right side. A fourth person is standing beside a black car observing. Wet ground, street lights, cinematic photography."
```

The five PNG files must exist; the template does not generate identities, poses,
or cutouts. Edit its positions/scales to fit your assets. Source paths and optional
`background` are relative to `reference_dir`; absolute paths also work. Canvas
dimensions for Qwen must be multiples of 32 in [64,1536]. Optional background
must match canvas exactly. Omitting `--layout` lays sorted PNGs out in a simple row:
**this is not a scene planner** and may include unwanted PNGs in that directory.
`--layout-only` works offline without a worker; `--output-dir` selects a fresh,
empty directory. Prompt text is saved and passed unchanged, not replaced with
concatenated descriptions. Defaults: 30 steps, seed 261011, guidance 1.

Runnable smoke test using existing project cutouts:

```bash
.venv/bin/python examples/multi_reference_compositor_demo.py \
  --reference_dir experiments/draftdag_poc/compositor_demo/layers \
  --layout examples/layouts/meadow_smoke.json \
  --base-url http://127.0.0.1:8190 \
  --prompt "Photorealistic meadow scene. Preserve the reference objects and their positions: bicycle with wicker basket and rustic fence at the left; woman in straw hat and floral sundress at center-left; golden retriever running in the lower center; ivy-covered pickup truck with a cat on its hood at the right. Replace the gray background with a continuous grassy meadow, an ancient oak behind the truck on the right, a distant clock tower, partly cloudy sky and warm sunlight. Add coherent contact shadows, realistic textures and lighting."
```

## Artifacts and interpretation

Each run creates `experiments/layer_composition_poc/<unique-run>/` with:

- `scene_ir.json`: all source hashes, minimal alpha crop, position, scale,
  rotation, raster bbox, z-order, opacity, arbitrary metadata and dependency IDs.
- `scene_layout.png`: opaque visual layout, actual and sole model reference.
- `foreground.png`, `foreground_mask.png`: transparent foreground and alpha union.
- `prompt.txt`: original prompt.
- `final.png`, `final.json`: raw Qwen render and worker metadata (not a post-pasted
  image). Absent in layout-only mode or failed generation.
- `artifacts.json`: input/output provenance and generation status, seed, steps,
  actual worker metadata and limitations. Failures preserve layout/IR for debugging.

Sources are tracked, not copied into the run. Keep them for replay; changing one
causes default hash verification to fail. Mask is a future editing artifact,
**not** currently a mask fed into Qwen. Its pipeline uses soft visual references,
not hard pixel/identity/position constraints; `strength` is not applied by this
backend. Lighting/background improvement and preservation must be evaluated on
actual renders. A meadow smoke test is not evidence of four-person improvement.

## Extension to DAG

Layer `node_id`/`depends_on` can reference branch outputs, source hashes can identify
drafts/partial renders, and composition IDs can become merge node IDs. `relations`
and arbitrary metadata retain semantic constraints without pretending to enforce
them. A scheduler can later load selected IRs, compose their layers, and publish
new IR plus PNG artifacts at merge nodes. Local edits change one layer's transform
or source, then replay; a masked backend could consume `foreground_mask.png`.
Pose/ControlNet require separate real control adapters, not merely new JSON keys.
This version does not implement a scheduler, pose control or learned planner.

Tests: `.venv/bin/python -m pytest -q tests/test_layer_compositor.py`.
