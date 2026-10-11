# DraftDAG: fixed-prompt Chain-of-Synthesis DAG

## Current design and status

`cos_dag.py` preserves a validated clean-latent **conceptual plan**. Run
`.venv/bin/python experiments/draftdag_poc/run.py dag-plan` to regenerate
`cos_dag/graph.json`; it is not an executable backend. With the user's
approval to use PNG references, `run.py dag` now executes `png_cos_dag.py`,
a smaller runnable approximation. Its artifacts are under `cos_png_dag/`.

The plan adapts HiCoGen's [Chain of Synthesis](https://arxiv.org/html/2511.19965):
generate minimal subject/attribute visual units, bind attributes to their
owners, synthesize local subject-object relations using the previous visual
outputs, then combine independent local subscenes by dependency. The ladybug
bridge is a genuine cross-branch dependency on both oak and ball. Only after
the local branches are complete may a spatially constrained scene assembler
and **one** final renderer produce a full-scene image. HiCoGen itself is a
hierarchical chain; the parallel branch DAG is our extension, not a claim
about the paper's exact architecture. Its latent-level synthesis and RL
training are not reproduced by this repository.

The runnable approximation first generates local subject and relation
draft/render pairs. It then assembles **both** a 1024×576 structural sketch
and a photographed PNG scene from the local partial renders at fixed spatial
slots. Exactly one scene-level Qwen call receives these two PNGs. No full-scene
image is repeatedly regenerated. Each node records visual
inputs and outputs, and the graph validator rejects cycles or missing
artifacts. `run.py compare` expects `cos_png_dag/final.png` and will not
reuse the legacy image as a new result. The Qwen backend still starts from
noise and cannot lock pixels or replicate HiCoGen clean-latent synthesis;
the comparison must be read as an approximation, not a paper reproduction.

`cos_png_dag/attempt_5refs/` preserves the first PNG-DAG attempt, where two
local subscenes were rendered again and then supplied with other references
to the final Qwen call. The right subscene turned into a multi-panel collage.
The current implementation instead directly places the successful local
renders in scene coordinates before the single final call.

## Current PNG-DAG result

### SAM 2.1 transparent cutout helper

`scripts/segment_image_sam2.py` extracts one prompted object from any source
image as an RGBA PNG using the local SAM 2.1 checkpoint. SAM2 needs a spatial
prompt (`--box` or `--point`); it cannot identify an object from text alone.
The saved RGB pixels come from the original image, while the predicted mask
becomes the alpha channel. The example below extracts the retriever from this
run's final image into `cos_png_dag/segmentation/retriever_transparent.png`:

```bash
/home/lipz/miniconda3/envs/uno_env/bin/python scripts/segment_image_sam2.py \
  --image experiments/draftdag_poc/cos_png_dag/final.png \
  --output experiments/draftdag_poc/cos_png_dag/segmentation/retriever_transparent.png \
  --box 420 390 630 520 --device cuda:2 --crop --margin 12
```

Use `--point X Y` for a positive click, `--negative-point X Y` to exclude a
region, omit `--crop` to retain the original canvas, and add `--overwrite` only
when intentionally replacing a previous output. Coordinates refer to the
original image dimensions.

### ComfyUI-inspired PNG layer compositor demo

`compositor_demo/manifest.json` places four independent SAM2 cutouts
(bicycle/fence, cat/truck, woman, retriever) on the existing environment
partial image. `scripts/composite_transparent_layers.py` implements the small
normal-blend subset used here: stable `z_index` ordering, x/y placement,
optional resizing, opacity, existing PNG alpha and optional inverse masks.
Color is composited in linear-light space, as in ComfyUI's compositor. This
is deterministic pixel compositing, **not** a ComfyUI workflow or diffusion
refinement, and segmentation artifacts can remain visible.

```bash
.venv/bin/python scripts/composite_transparent_layers.py \
  --manifest experiments/draftdag_poc/compositor_demo/manifest.json \
  --output experiments/draftdag_poc/compositor_demo/composite.png \
  --save-stages experiments/draftdag_poc/compositor_demo/stages
```

All cutouts, four incremental stage PNGs, the layer manifest, and the final
`composite.png` are preserved under `compositor_demo/` for visual debugging.
`layers/truck.png` is a rejected trial cutout whose box clipped the cat's
head; the manifest uses `layers/cat_truck.png` instead.

The more photoreal variant uses `compositor_demo/photoreal_background.png`,
generated with the local Qwen-Image-2.1 worker in **text-to-image** mode (no
flat sketch reference; seed 261011, 40 steps). The request recorded guidance
4, but the pipeline logged that classifier-free guidance was not enabled
because no negative prompt was supplied. Its exact prompt
is in `compositor_demo/photoreal_background_prompt.txt`. The new
`manifest_photoreal.json` has precisely the same four layer entries as the
original manifest; only the background path changes. To reproduce after
starting the Qwen worker as described below:

```bash
QWEN_IMAGE_BASE_URL=http://127.0.0.1:8190 QWEN_IMAGE_MODEL_NAME=qwen-image \
QWEN_IMAGE_WIDTH=1024 QWEN_IMAGE_HEIGHT=576 QWEN_IMAGE_STEPS=40 \
QWEN_IMAGE_GUIDANCE=4 QWEN_IMAGE_SEED=261011 \
  /home/lipz/miniconda3/envs/uno_env/bin/python scripts/qwen_smoke.py \
  --prompt-file experiments/draftdag_poc/compositor_demo/photoreal_background_prompt.txt \
  --output experiments/draftdag_poc/compositor_demo/photoreal_background.png
.venv/bin/python scripts/composite_transparent_layers.py \
  --manifest experiments/draftdag_poc/compositor_demo/manifest_photoreal.json \
  --output experiments/draftdag_poc/compositor_demo/composite_photoreal_background.png \
  --save-stages experiments/draftdag_poc/compositor_demo/stages_photoreal
```

These commands refuse to overwrite existing outputs unless `--overwrite` is
given. No whole-scene diffusion refinement was run on the photoreal composite:
the four local cutouts and their exact placements remain intact. Remaining
edge and shadow mismatches are visible and should not be mistaken for a fully
harmonized final render.

### Foreground-locked background-conditioning attempt

`compositor_demo/foreground_locked_attempt/` is a separate follow-up. Qwen
receives the **already positioned** `composite_photoreal_background.png` as
Reference A and produces `model_raw.png`. The compositor then uses that
output as a replacement background and reapplies the original four RGBA
cutouts at unchanged coordinates, producing `final_locked.png`. It also
exports `foreground_mask.png` (white = protected original foreground). The
exact prompt and run metadata are saved next to the images.

This follows the pixel-preservation idea behind ComfyUI's
`ImageCompositeMasked`/mask utilities, but is **not true masked inpainting**:
the installed `QwenImage21Pipeline` call accepts an image reference but no
`mask_image`, and the worker ignores `strength` for this backend. Qwen
regenerates the whole image first; the foreground is restored afterwards.
The opaque foreground pixels were verified exactly unchanged (63,665 pixels,
maximum RGB difference 0), while most unprotected background pixels changed.
Seam/shadow improvement was modest, and model-generated ghosts outside the
protected masks remain a possible failure mode.

```bash
QWEN_IMAGE_BASE_URL=http://127.0.0.1:8190 QWEN_IMAGE_MODEL_NAME=qwen-image \
QWEN_IMAGE_WIDTH=1024 QWEN_IMAGE_HEIGHT=576 QWEN_IMAGE_STEPS=30 \
QWEN_IMAGE_GUIDANCE=1 QWEN_IMAGE_SEED=261013 \
  /home/lipz/miniconda3/envs/uno_env/bin/python scripts/qwen_smoke.py \
  --prompt-file experiments/draftdag_poc/compositor_demo/foreground_locked_attempt/integration_prompt.txt \
  --reference experiments/draftdag_poc/compositor_demo/composite_photoreal_background.png \
  --output experiments/draftdag_poc/compositor_demo/foreground_locked_attempt/model_raw.png
.venv/bin/python scripts/composite_transparent_layers.py \
  --manifest experiments/draftdag_poc/compositor_demo/manifest_photoreal.json \
  --background-override experiments/draftdag_poc/compositor_demo/foreground_locked_attempt/model_raw.png \
  --foreground-mask-output experiments/draftdag_poc/compositor_demo/foreground_locked_attempt/foreground_mask.png \
  --output experiments/draftdag_poc/compositor_demo/foreground_locked_attempt/final_locked.png
```

Add `--overwrite` only when intentionally replacing a saved artifact.

To inspect the executed DAG visually, open `cos_png_dag/debug.html` in a
browser. It is a self-contained offline page: all 13 executed nodes, their
15 dependency edges, actual drafts, local renders, model reference images in
order, and spatial-composition inputs are embedded in one HTML file. Click a
node to see all its image inputs/outputs and recorded metadata; click an image
to zoom. The page is generated from `cos_png_dag/graph.json` and the node
records, not the conceptual `cos_dag/` plan. Rebuild it after a new run with:

```bash
.venv/bin/python experiments/draftdag_poc/visualize_png_dag.py
```

The assembly metadata explicitly marks the retriever-ball fallback: the
unsuccessful `retriever_ball/partial.png` was not pasted into the assembled
photograph. The actual inputs were the retriever partial image plus a manually
positioned red ball. The visual debugger distinguishes this deterministic
composition from a model call, and distinguishes image inputs from images
actually sent as model references.

For the unchanged fixed prompt and 15-item `qwen-vl-plus` checklist,
`comparison_cos_png_dag.json` scored the baseline **13/15** and this PNG-DAG
**7/15**. The first five-reference PNG-DAG attempt also scored **7/15**;
its review is archived as `comparison_cos_png_dag_5refs.json`. The current
render visibly preserves the golden retriever, red ball, woman, water arc,
bicycle-fence relation, pickup and cat on its hood. It misses or leaves
ambiguous the terrier digging, basket contents, clock tower, drone, woman
adjusting the nozzle and the ladybug bridge. Visible cutout seams remain.
No checklist item or threshold was changed to obtain these scores.

## Historical PNG-reference experiment (not CoS-DAG)

`prompt.txt` is the only source prompt. Its SHA256 is recorded in
`constraints.json`, the historical DAG, and `comparison.json`. The existing baseline is
one GenClaw global SVG draft followed by one Qwen `i2i` image.

`run.py legacy-dag` executes the preserved `branch_dag.py` implementation:

```text
object_branch ──────┐
environment_branch ┴─> merge_1 ──> add woman ─> add retriever ─> add terrier ─> add cat / merge_2
character_woman/retriever/terrier/cat ──────────┘                                  │
relation_branch ────────────────────────────────────────────────────────────────> merge_3 ─> final
```

The character module has **four independent child nodes**:
`character_woman`, `character_retriever`, `character_terrier`, and
`character_cat`. Each child produces its own `draft.png` via GenClaw's
`CodeSceneDraftTool` and its own `partial.png` via Qwen-Image-2.1 diffusion.
`character_branch/node.json` aggregates the eight visual outputs; it does not
make a combined four-character image. Object, environment and relation
branches likewise each produce a draft and diffusion partial. Seven drafts
are generated concurrently; the single local GPU worker serializes diffusion.

Each character's saved `draft.png` is now a local crop around the union of its
foreground layout bboxes, with a small margin, a model-safe aspect ratio,
and dimensions aligned to 32 pixels (required by Qwen-Image-2.1).
`full_draft.png` preserves the original 1600×900 generator output and
`crop.json` records the exact crop coordinates. Its Qwen `partial.png` is
rendered at the same local dimensions, not at full-scene size. Object,
environment, relation, merge, and final images retain scene-sized canvases.
This reduces empty background in character references; it does not provide
pixel-accurate compositing or lock existing scene pixels during Qwen merges.
The run can resume from valid saved drafts, partial images and merge images
after interruption, instead of regenerating all branches.

Each merge calls `I2ITool` with the previous partial scene as Reference A
and the next branch's **draft and diffusion partial image** as additional
condition images. `merge_1` uses environment and object drafts and partials.
Four character merge nodes then add one identity at a time, each consuming
that character's own draft and partial. `merge_3` adds relation draft and
partial. The Qwen worker now forwards all received condition images to
the installed `QwenImage21Pipeline`. Its former behavior used only the first.

Final refinement consumes all character drafts and partials, the other three
branches' drafts and partials, and six merge images across six direct-reference
calls. Reference order and worker-confirmed counts are saved for each call.
An attempt with seven 1024×576 condition images exceeded the 47 GiB GPU;
the completed version uses at most five per call and resumed from saved rounds.

## Run

Start a dedicated Qwen-Image-2.1 worker, for example on GPU 1 and port 8190:

```bash
CUDA_VISIBLE_DEVICES=1 QWEN_IMAGE_MODEL_PATH=/home/hdd4/lpz/Qwen-Image-2.1 \
  /home/lipz/miniconda3/envs/uno_env/bin/python server/diffusion_server.py \
  --backend qwen-image --port 8190 --model-path /home/hdd4/lpz/Qwen-Image-2.1
```

Then from the repository root:

```bash
QWEN_IMAGE_BASE_URL=http://127.0.0.1:8190 .venv/bin/python experiments/draftdag_poc/run.py legacy-dag
.venv/bin/python experiments/draftdag_poc/run.py legacy-compare
```

The existing baseline lives in `baseline/`. To create it from scratch in a
clean experiment directory, run `run.py baseline` with the same worker URL.
Stages refuse to overwrite existing final images. The comparison uses the
same 15-item checklist for both final images. Existing `dag/` and
`comparison.json` are retained as historical artifacts; do not interpret them
as output of the corrected CoS-DAG plan.

## Interpretation and limitations

`dag_fullsize_characters/` and `comparison_fullsize_characters.json` preserve
the immediately preceding split-character run whose references were still
full-frame. `dag_character_group/` and `comparison_character_group.json` preserve the
previous four-characters-in-one-image implementation and its 13/15 versus
5/15 result. The historical `dag/` is the split-character implementation.
`dag_board_attempt/` preserves an unsuccessful final refinement that used
reference boards and produced a four-panel collage. Its source images and
model metadata remain available. A copy of those failed final artifacts is
also under `dag/failed_board_attempt/`. `dag_svg_only/` and
`comparison_svg_only.json` preserve the previous SVG-only
branch experiment, which did **not** produce branch partial images. The even
earlier chain attempt is in `dag_initial_chain/` and
`comparison_initial_chain.json`. Neither belongs to the corrected CoS-DAG plan.
`dag_crop_alignment_attempt/` holds the interrupted first crop attempt,
which showed that a 752-pixel draft width was silently rounded to 736 pixels
by the Qwen backend; the historical cropped-character run uses 32-pixel alignment.

Every merge is a conditional Qwen generation from previous visual artifacts,
but `QwenImage21Pipeline` begins its output latents from noise. The request's
`strength=0.65` is **not applied** by this backend; it is reported as such in
new worker metadata. Previous pixels and satisfied relations are not locked.
Mixing flat SVG sketches with photoreal partials, especially white-background
images, visibly affected merge output. Sequential passes compound drift.
This experiment uses
one prompt and more model calls on the DAG arm, so a score difference alone
would not isolate the cause. The baseline planner raw output in `plan.json`
proposed `t2i` and reached its turn limit; the baseline runner explicitly
executed one `code_scene_draft` followed by one `i2i` instead.

## Result for this fixed prompt

On the unchanged `qwen-vl-plus` 15-item checklist, the baseline scored
**13/15**, the preceding full-frame-character DAG **5/15**, and this local-bbox
character DAG **6/15**. This run's character draft and Qwen partial dimensions
match exactly: woman 480×896, retriever 576×608, terrier 384×544, cat
512×640. The VLM marked red ball, woman, woman holding/adjusting hose,
curved water arc, bicycle leaning against fence, and truck as satisfied.
It marked retriever identity, terrier digging, basket with wilted daisies,
cat on truck hood, clock tower, drone, and all three ladybug criteria as
missing or unclear. Reasons for every item are in `comparison.json`.

Cropping removed most empty background from character references and raised
the checklist score by one point in this single run, but merge/final images
still show strong artifacts and relation loss. This does **not** establish a
quality improvement: the runs are stochastic and the Qwen backend still
regenerates each scene from noise, without locking existing pixels.
