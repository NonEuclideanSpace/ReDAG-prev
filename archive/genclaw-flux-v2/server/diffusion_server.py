from __future__ import annotations

import argparse
import base64
import io
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = "/home/hdd4/lpz/Qwen-Image-2.1"
QWEN_BACKENDS = {"qwen-image"}
FILL_BACKENDS = {"flux-fill"}
EDIT_BACKENDS = {"qwen-image-edit", "flux-dev"}
BACKENDS = sorted(QWEN_BACKENDS | FILL_BACKENDS | EDIT_BACKENDS | {"mock"})


def backend_kind(backend: str) -> str:
    if backend in QWEN_BACKENDS:
        return "qwen"
    if backend in FILL_BACKENDS:
        return "fill"
    if backend in EDIT_BACKENDS:
        return "edit"
    return backend


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal["qwen-image", "qwen-image-edit", "flux-fill", "flux-dev"] = "qwen-image"
    prompt: str = Field(min_length=1, max_length=20000)
    images: List[str] = Field(default_factory=list, max_length=8)
    width: int = Field(default=768, ge=64, le=1536, multiple_of=16)
    height: int = Field(default=768, ge=64, le=1536, multiple_of=16)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    steps: int = Field(default=28, ge=1, le=100)
    guidance: float = Field(default=30.0, ge=0, le=100, allow_inf_nan=False)
    strength: float = Field(default=0.6, gt=0, le=1, allow_inf_nan=False)


def decode_image(value: str) -> Image.Image:
    if len(value) > 12 * 1024 * 1024:
        raise ValueError("A reference image exceeds the 12 MiB base64 limit.")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 20_000_000:
                raise ValueError("Reference image exceeds 20 megapixels.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid or oversized reference image.") from exc


def to_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (width, height))
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(fitted, ((width - fitted.width) // 2, (height - fitted.height) // 2))
    return canvas


def fill_canvas(images: List[Image.Image], width: int, height: int):
    if not images:
        return (Image.new("RGB", (width, height), (200, 200, 200)),
                Image.new("L", (width, height), 255), (0, 0, width, height))
    canvas = Image.new("RGB", (width * 2, height), "white")
    main_height = height if len(images) == 1 else height * 2 // 3
    main = ImageOps.contain(images[0], (width, main_height))
    canvas.paste(main, ((width - main.width) // 2, (main_height - main.height) // 2))
    if len(images) > 1:
        cell_width = width // (len(images) - 1)
        for index, source in enumerate(images[1:]):
            ref = ImageOps.contain(source, (cell_width, height - main_height))
            canvas.paste(ref, (index * cell_width + (cell_width - ref.width) // 2,
                               main_height + (height - main_height - ref.height) // 2))
    canvas.paste(Image.new("RGB", (width, height), (200, 200, 200)), (width, 0))
    mask = Image.new("L", canvas.size, 0)
    mask.paste(255, (width, 0, width * 2, height))
    return canvas, mask, (width, 0, width * 2, height)


class DiffusionEngine:
    def __init__(self, backend: str, model_path: str, cpu_offload=False, lora_path=None):
        self.backend = backend
        self.model_path = Path(model_path)
        self.cpu_offload = cpu_offload
        self.lora_path = lora_path
        self.pipe = None
        self.edit_pipe = None

    def missing_files(self):
        if self.backend == "mock":
            return []
        missing = []
        kind = backend_kind(self.backend)
        index_path = self.model_path / "model_index.json"
        components = {}
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text())
                expected = "QwenImage21Pipeline" if kind == "qwen" else (
                    "FluxFillPipeline" if kind == "fill" else "FluxPipeline")
                if index.get("_class_name") != expected:
                    missing.append("model_index.json (pipeline does not match selected backend)")
                components = {k: v for k, v in index.items() if isinstance(v, list) and len(v) >= 2}
            except (ValueError, OSError):
                missing.append("model_index.json (invalid)")
        if kind == "qwen":
            required = ["model_index.json", "transformer/config.json", "vae/config.json",
                        "scheduler/scheduler_config.json", "text_encoder/config.json"]
            for component in ["processor"]:
                if component in components and not (self.model_path / component).is_dir():
                    missing.append(component + "/")
        else:
            required = ["model_index.json", "transformer/config.json", "vae/config.json",
                        "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
                        "tokenizer_2/tokenizer_config.json"]
        for relative in required:
            if not (self.model_path / relative).is_file():
                missing.append(relative)
        weight_components = ["transformer", "vae", "text_encoder"] if kind == "qwen" else [
            "transformer", "vae", "text_encoder", "text_encoder_2"]
        for component in weight_components:
            folder = self.model_path / component
            weights = list(folder.glob("*.safetensors")) + list(folder.glob("*.bin"))
            if not weights:
                missing.append(component + "/<weights>")
            for index in folder.glob("*.index.json"):
                try:
                    shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
                    missing.extend(str((folder / shard).relative_to(self.model_path))
                                   for shard in shards if not (folder / shard).is_file())
                except (ValueError, OSError):
                    missing.append(str(index.relative_to(self.model_path)) + " (invalid)")
        if self.lora_path and not Path(self.lora_path).is_file():
            missing.append("configured LoRA")
        return sorted(set(missing))

    def load(self):
        if self.pipe is not None or self.backend == "mock":
            return
        missing = self.missing_files()
        if missing:
            raise RuntimeError("Local model files missing: " + ", ".join(missing))
        import torch
        import diffusers
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; launch this worker in the GPU environment.")
        kind = backend_kind(self.backend)
        if kind == "qwen":
            cls = getattr(diffusers, "QwenImage21Pipeline", None)
            if cls is None:
                raise RuntimeError("diffusers does not provide QwenImage21Pipeline; install the Qwen-Image-2.1 compatible diffusers build.")
        else:
            cls = diffusers.FluxFillPipeline if kind == "fill" else diffusers.FluxPipeline
        pipe = cls.from_pretrained(str(self.model_path), torch_dtype=torch.bfloat16,
                                   local_files_only=True)
        if self.lora_path:
            path = Path(self.lora_path)
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, local_files_only=True)
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        for enable in ("enable_slicing", "enable_tiling"):
            method = getattr(getattr(pipe, "vae", None), enable, None)
            if callable(method):
                method()
        self.pipe = pipe

    def release_memory(self):
        if self.pipe is None:
            return
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, req: GenerationRequest, images: List[Image.Image], seed: int):
        if self.backend == "mock":
            image = Image.new("RGB", (req.width, req.height), "#dae7f7")
            ImageDraw.Draw(image).text((12, 12), "MOCK TRANSPORT TEST - NOT DIFFUSION", fill="black")
            return image, "mock"
        self.load()
        import torch
        kind = backend_kind(self.backend)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs = dict(num_inference_steps=req.steps, guidance_scale=req.guidance, generator=generator)
        if kind == "qwen":
            if images:
                raise RuntimeError("qwen-image backend does not support reference images in this worker.")
            with torch.inference_mode():
                result = self.pipe(prompt=req.prompt, width=req.width, height=req.height, **kwargs).images[0]
            return result, "text-to-image"
        if kind == "fill":
            canvas, mask, crop = fill_canvas(images, req.width, req.height)
            prompt = req.prompt
            if images:
                prompt = ("A two-panel composition. The left panel contains visual references. "
                          "Its largest image is the main composition reference; smaller images are supplementary. "
                          "Generate one finished image in the right panel, following the main reference's "
                          "object count, positions and composition. Do not draw borders or reference panels "
                          "inside the right panel. " + req.prompt)
            with torch.inference_mode():
                result = self.pipe(prompt=prompt, image=canvas, mask_image=mask,
                                   width=canvas.width, height=canvas.height, **kwargs).images[0]
            return result.crop(crop), "reference-panel-fill" if images else "full-mask-fill"
        if images:
            if self.edit_pipe is None:
                from diffusers import FluxImg2ImgPipeline
                self.edit_pipe = FluxImg2ImgPipeline(
                    scheduler=self.pipe.scheduler, vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder, tokenizer=self.pipe.tokenizer,
                    text_encoder_2=self.pipe.text_encoder_2, tokenizer_2=self.pipe.tokenizer_2,
                    transformer=self.pipe.transformer)
                if self.cpu_offload:
                    self.edit_pipe.enable_model_cpu_offload()
            with torch.inference_mode():
                result = self.edit_pipe(prompt=req.prompt, image=to_canvas(images[0], req.width, req.height),
                                        width=req.width, height=req.height, strength=req.strength, **kwargs).images[0]
            return result, "img2img"
        with torch.inference_mode():
            result = self.pipe(prompt=req.prompt, width=req.width, height=req.height, **kwargs).images[0]
        return result, "text-to-image"


def create_app(engine: DiffusionEngine, token: str = "") -> FastAPI:
    app = FastAPI(title="GenClaw Diffusion worker", version="0.1.0")
    gate = threading.Lock()

    def authorize(authorization):
        if token and not secrets.compare_digest(authorization or "", "Bearer " + token):
            raise HTTPException(401, "Invalid worker token.")

    @app.get("/health")
    def health(authorization: Optional[str] = Header(default=None)):
        authorize(authorization)
        missing = engine.missing_files()
        return {"status": "ok" if not missing else "model_files_missing",
                "backend": engine.backend, "model_files_ready": not missing,
                "model_loaded": engine.pipe is not None, "missing_files": missing,
                "busy": gate.locked(), "max_references": 1 if backend_kind(engine.backend) == "edit" else 8,
                "note": "File presence only; generation validates CUDA and model loading."}

    @app.post("/v1/generate")
    def generate(req: GenerationRequest, authorization: Optional[str] = Header(default=None)):
        authorize(authorization)
        if not req.prompt.strip():
            raise HTTPException(422, "Prompt cannot be blank.")
        if engine.backend != "mock" and req.model != engine.backend:
            raise HTTPException(409, f"Worker runs {engine.backend}, request specifies {req.model}.")
        kind = backend_kind(engine.backend)
        if kind == "edit" and len(req.images) > 1:
            raise HTTPException(422, "qwen-image-edit img2img supports one image; extra references will not be discarded.")
        if kind == "edit" and req.images and int(req.steps * req.strength) < 1:
            raise HTTPException(422, "steps * strength must be at least 1 for img2img.")
        area = req.width * req.height * (2 if kind == "fill" and req.images else 1)
        if area > 2_097_152:
            raise HTTPException(422, "Working canvas exceeds 2 megapixels; reduce width/height.")
        try:
            images = [decode_image(item) for item in req.images]
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not gate.acquire(blocking=False):
            raise HTTPException(429, "Worker busy; wait for the current generation to finish.")
        try:
            start = time.monotonic()
            seed = req.seed if req.seed is not None else secrets.randbelow(2**32)
            image, mode = engine.generate(req, images, seed)
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            return {"image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "backend": engine.backend, "mode": mode, "seed": seed,
                    "width": image.width, "height": image.height, "steps": req.steps,
                    "guidance": req.guidance, "strength": req.strength,
                    "lora": Path(engine.lora_path).name if engine.lora_path else None,
                    "elapsed_seconds": round(time.monotonic() - start, 3),
                    "warnings": ["Experimental reference-panel conditioning; layout/text fidelity is not guaranteed."]
                    if mode == "reference-panel-fill" else []}
        except Exception as exc:
            raise HTTPException(503, f"Diffusion inference failed: {type(exc).__name__}: {str(exc)[:500]}") from exc
        finally:
            engine.release_memory()
            gate.release()

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="qwen-image")
    parser.add_argument("--model-path", default=os.environ.get(
        "QWEN_IMAGE_MODEL_PATH",
        os.environ.get("DIFFUSION_MODEL_PATH", os.environ.get("FLUX_MODEL_PATH", MODEL_PATH))))
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--lora-path", default=None)
    args = parser.parse_args()
    import uvicorn
    app = create_app(DiffusionEngine(args.backend, args.model_path, args.cpu_offload, args.lora_path),
                     token=os.environ.get("QWEN_IMAGE_API_KEY", os.environ.get("FLUX_API_KEY", "")))
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
