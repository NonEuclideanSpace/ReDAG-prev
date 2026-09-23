from __future__ import annotations

import argparse
import base64
import io
import os
import random
import secrets
import threading
import time
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Literal["qwen-image-controlnet", "flux-fill", "flux-dev", "mock"] = "qwen-image-controlnet"
    prompt: str = Field(min_length=1, max_length=20000)
    negative_prompt: str = " "
    images: List[str] = Field(default_factory=list, max_length=8)
    width: int = Field(default=1024, ge=64, le=1536, multiple_of=16)
    height: int = Field(default=1024, ge=64, le=1536, multiple_of=16)
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)
    steps: int = Field(default=28, ge=1, le=100)
    guidance: float = Field(default=4.0, ge=0, le=100, allow_inf_nan=False)
    strength: float = Field(default=0.7, gt=0, le=1, allow_inf_nan=False)
    control_scale: float = Field(default=0.8, ge=0, le=2, allow_inf_nan=False)
    control_mode: Literal["raw", "canny", "lineart"] = "canny"


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


def control_image(draft: Image.Image, width: int, height: int, mode: str) -> Image.Image:
    canvas = to_canvas(draft, width, height)
    if mode == "raw":
        return canvas
    gray = canvas.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda v: 255 if v > 24 else 0)
    if mode == "lineart":
        edges = ImageOps.invert(edges)
    return edges.convert("RGB")


class Engine:
    def __init__(self, backend: str, base_model: str, controlnet_model: str, cpu_offload: bool = False):
        self.backend = backend
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.cpu_offload = cpu_offload
        self.pipe = None

    def load(self) -> None:
        if self.backend == "mock" or self.pipe is not None:
            return
        import torch
        from diffusers import QwenImageControlNetModel, QwenImageControlNetPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; launch this worker in the GPU environment.")
        controlnet = QwenImageControlNetModel.from_pretrained(self.controlnet_model, torch_dtype=torch.bfloat16)
        pipe = QwenImageControlNetPipeline.from_pretrained(self.base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
        if self.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self.pipe = pipe

    def generate(self, req: GenerationRequest, draft: Optional[Image.Image], seed: int) -> Image.Image:
        if self.backend == "mock":
            return self._mock(req, draft, seed)
        import torch

        control = control_image(draft, req.width, req.height, req.control_mode) if draft is not None else Image.new("RGB", (req.width, req.height), "white")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or " ",
            control_image=control,
            controlnet_conditioning_scale=float(req.control_scale),
            width=req.width,
            height=req.height,
            num_inference_steps=int(req.steps),
            true_cfg_scale=float(req.guidance),
            generator=generator,
        )
        return result.images[0]

    def _mock(self, req: GenerationRequest, draft: Optional[Image.Image], seed: int) -> Image.Image:
        rng = random.Random(seed)
        if draft is None:
            base = Image.new("RGB", (req.width, req.height), (200, 200, 200))
        else:
            base = to_canvas(draft, req.width, req.height)
        out = base.filter(ImageFilter.GaussianBlur(radius=1.5 + 2.0 * float(req.strength)))
        px = out.load()
        for _ in range(int(req.width * req.height * 0.002)):
            x, y = rng.randrange(req.width), rng.randrange(req.height)
            r, g, b = px[x, y]
            px[x, y] = (max(0, min(255, r + rng.randint(-25, 25))), max(0, min(255, g + rng.randint(-25, 25))), max(0, min(255, b + rng.randint(-25, 25))))
        return out


def build_app(engine: Engine, token: str) -> FastAPI:
    app = FastAPI(title="DraftDAG generation worker")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": engine.backend, "loaded": engine.pipe is not None or engine.backend == "mock"}

    @app.post("/v1/generate")
    def generate(req: GenerationRequest, authorization: Optional[str] = Header(default=None)):
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="invalid token")
        try:
            draft = decode_image(req.images[0]) if req.images else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        seed = req.seed if req.seed is not None else secrets.randbelow(2**32)
        started = time.time()
        with lock:
            try:
                engine.load()
                image = engine.generate(req, draft, seed)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}"[:600])
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "backend": engine.backend,
            "model": req.model,
            "seed": seed,
            "width": image.width,
            "height": image.height,
            "steps": req.steps,
            "guidance": req.guidance,
            "strength": req.strength,
            "control_scale": req.control_scale,
            "control_mode": req.control_mode,
            "seconds": round(time.time() - started, 2),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-Image ControlNet generation worker for DraftDAG")
    parser.add_argument("--backend", choices=["mock", "qwen-controlnet"], default=os.environ.get("DRAFTDAG_GEN_BACKEND", "mock"))
    parser.add_argument("--base-model", default=os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image"))
    parser.add_argument("--controlnet", default=os.environ.get("QWEN_CONTROLNET_MODEL", "InstantX/Qwen-Image-ControlNet-Union"))
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--token", default=os.environ.get("DRAFTDAG_GEN_TOKEN", ""))
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()
    engine = Engine(args.backend, args.base_model, args.controlnet, cpu_offload=args.cpu_offload)
    if args.preload:
        engine.load()
    import uvicorn

    uvicorn.run(build_app(engine, args.token), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
