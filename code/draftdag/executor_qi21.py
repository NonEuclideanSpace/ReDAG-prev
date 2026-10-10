"""Qwen-Image-2.1 executor (diffusers >= 0.41, QwenImage21Pipeline). Text-to-image and instruction editing share
one model; editing passes the canvas as the condition image. Requires the pipeline module to be importable
(the lab installs diffusers into envs/qi21_site and sets PYTHONPATH)."""
import os
import time

import torch
from PIL import Image

from .executor import Executor
from .paths import models_dir


def load_pipe(model_dir=None, device=None, offload=False, dtype=torch.bfloat16):
    from diffusers import QwenImage21Pipeline

    device = torch.device(device or os.environ.get("P0_DEVICE") or "cuda")
    pipe = QwenImage21Pipeline.from_pretrained(model_dir or models_dir("Qwen-Image-2.1"), torch_dtype=dtype)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


class QI21(Executor):
    name = "qwen-image-2.1"

    def __init__(self, model_dir=None, device=None, steps=40, cfg=1.0, res=1024, offload=None, chunk=None, use_kv_cache=True):
        if offload is None:
            offload = os.environ.get("P0_QI_OFFLOAD", "0") == "1"
        self.pipe, self.device = load_pipe(model_dir, device, offload)
        self.steps, self.cfg, self.res = int(steps), float(cfg), int(res)
        self.chunk = chunk
        self.use_kv_cache = bool(use_kv_cache)
        self.has_lora = False
        self.calls = 0
        self.sec = 0.0

    def _chunk(self, size):
        if self.chunk:
            return int(self.chunk)
        env = os.environ.get("P0_QI_CHUNK")
        if env:
            return int(env)
        return 4 if size <= 768 else 2

    @torch.no_grad()
    def draw(self, prompt, n, seed, size=1024):
        t0 = time.time()
        out = []
        ch = self._chunk(size)
        for b0 in range(0, int(n), ch):
            b = min(ch, int(n) - b0)
            g = torch.Generator("cpu").manual_seed(int(seed) + 104729 * b0)
            out += self.pipe(prompt=prompt, width=int(size), height=int(size), num_inference_steps=self.steps, true_cfg_scale=self.cfg, num_images_per_prompt=b, generator=g, use_kv_cache=self.use_kv_cache).images
        self.calls += int(n)
        self.sec += time.time() - t0
        return [im.convert("RGB") for im in out]

    @torch.no_grad()
    def edit(self, canvas, instruction, n, seed):
        t0 = time.time()
        out = []
        ch = self._chunk(self.res)
        canvas = canvas.convert("RGB")
        for b0 in range(0, int(n), ch):
            b = min(ch, int(n) - b0)
            g = torch.Generator("cpu").manual_seed(int(seed) + 104729 * b0)
            out += self.pipe(prompt=instruction, image=canvas, num_inference_steps=self.steps, true_cfg_scale=self.cfg, num_images_per_prompt=b, generator=g, output_resolution=self.res, use_kv_cache=self.use_kv_cache).images
        self.calls += int(n)
        self.sec += time.time() - t0
        return [im.convert("RGB").resize(canvas.size, Image.LANCZOS) for im in out]

    def load_lora(self, path, adapter_name="default"):
        if os.path.isdir(path):
            self.pipe.load_lora_weights(path, adapter_name=adapter_name)
        else:
            self.pipe.load_lora_weights(os.path.dirname(path), weight_name=os.path.basename(path), adapter_name=adapter_name)
        self.has_lora = True

    def stats(self):
        return dict(name=self.name, calls=self.calls, sec=round(self.sec, 1), steps=self.steps, res=self.res, lora=self.has_lora)
