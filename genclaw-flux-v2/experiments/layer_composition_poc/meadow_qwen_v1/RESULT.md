# Layer Composition / Qwen smoke result

Real backend: local `/home/hdd4/lpz/Qwen-Image-2.1`, seed 261014, 30 steps,
1024 x 576. Worker generation time: 30.179 seconds. Both `reference_count` and
`condition_image_count` are 1. The sole model reference is `scene_layout.png`.
Original prompt is saved verbatim in `prompt.txt`.

Four existing transparent assets were used: woman, retriever, cat+truck and
bicycle+fence. These are **not** four separate people. See source hashes and
alpha-cropped transforms in `scene_ir.json`. The layout has no background asset:
Qwen generated the environmental completion while seeing the placed subjects.

Manual visual inspection (not VLM scoring or quantitative identity evaluation):

- The neutral gray layout became a grassy meadow with sky, a clock tower and
  an oak on the right.
- Woman remains center-left, bicycle/fence left, running dog lower center and
  cat/truck right. Cat remains on the truck hood.
- Subjects still resemble the supplied assets; this is not proof of exact
  identity preservation. Dog/head detail and some texture were modified.
- Bicycle cutout's incomplete geometry/white residue remains visible. The
  model did not fully repair it; cutout quality is a bottleneck.
- `final.png` is raw diffusion output, not a foreground-locked recomposite.

Conclusion: the module and one-layout-reference generation path work end to
end. This sample suggests useful background completion without gross layout
relocation, but does **not** establish multi-person improvement or superiority
over direct multi-reference generation. That requires appropriate human assets,
a fixed prompt and a controlled comparison using the same seed/settings.

Verification: `46 passed` for the full project suite; `14 passed` for the new
compositor tests plus the legacy compositor tests. No ComfyUI was used or installed.

Reproduction (worker must be running):

```bash
.venv/bin/python examples/multi_reference_compositor_demo.py \
  --reference_dir experiments/draftdag_poc/compositor_demo/layers \
  --layout examples/layouts/meadow_smoke.json \
  --base-url http://127.0.0.1:8190 \
  --prompt-file experiments/layer_composition_poc/meadow_qwen_v1/prompt.txt \
  --seed 261014 --steps 30
```

The command chooses a new output directory. Preserve source assets for verified
IR replay. Current backend has no hard identity/layout lock or mask conditioning;
worker `strength_applied` is false. Foreground mask is exported for future editing,
not fed into this run.
