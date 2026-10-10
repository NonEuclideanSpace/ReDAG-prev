"""Training-side adapter for Qwen-Image-2.1: SDE rollouts with per-token log-probs and differentiable log-prob
recomputation, mirroring QwenImage21Pipeline.__call__ step by step (no KV cache, so rollouts and training see the
same computation). Validate on a GPU with scripts/validate_policy.py before any training run."""
import math
import os

import numpy as np
import torch

from .grpo import sde_coeffs, sde_mean, sde_step, token_logprob


def _qi_mod():
    from diffusers.pipelines.qwenimage21 import pipeline_qwenimage21 as m

    return m


class QI21Policy:
    def __init__(self, pipe, device, res=768, steps=10, noise_level=0.7, train_res=None):
        self.pipe = pipe
        self.device = torch.device(device)
        self.res = int(res)
        self.steps = int(steps)
        self.noise_level = float(noise_level)
        self.transformer = pipe.transformer
        self.vsf = pipe.vae_scale_factor
        self.in_channels = pipe.transformer.config.in_channels
        self._cond_cache = {}
        for m in (pipe.vae, pipe.text_encoder):
            m.requires_grad_(False)
        self.transformer.requires_grad_(False)

    def add_lora(self, rank=32, alpha=None, targets=("to_q", "to_k", "to_v", "to_out.0"), grad_ckpt=True):
        from peft import LoraConfig

        self.transformer.add_adapter(LoraConfig(r=int(rank), lora_alpha=int(alpha or rank), init_lora_weights="gaussian", target_modules=list(targets)))
        if grad_ckpt:
            self.transformer.enable_gradient_checkpointing()
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def lora_state(self):
        from peft.utils import get_peft_model_state_dict

        return {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(self.transformer).items()}

    def save_lora(self, path):
        sd = self.lora_state()
        type(self.pipe).save_lora_weights(save_directory=os.path.dirname(path), transformer_lora_layers=sd, weight_name=os.path.basename(path))

    def size_for(self, canvas):
        m = _qi_mod()
        w, h, _ = m.calculate_dimensions(self.res * self.res, canvas.size[0] / canvas.size[1])
        mult = self.vsf * 2
        return int(w) // mult * mult, int(h) // mult * mult

    @torch.no_grad()
    def encode(self, instruction, canvas, key=None):
        """Text + condition image encoding shared by all candidates of a group."""
        key = key or (instruction, id(canvas))
        if key in self._cond_cache:
            return self._cond_cache[key]
        pipe = self.pipe
        m = _qi_mod()
        img = canvas.convert("RGBA")
        iw, ih, _ = m.calculate_dimensions(self.res * self.res, img.size[0] / img.size[1])
        input_img = pipe.image_processor.resize(img, width=iw, height=ih)
        vae_img = pipe.image_processor.preprocess(img, width=iw, height=ih).unsqueeze(2)
        width, height = self.size_for(canvas)
        pe, pm, ipm = pipe.encode_prompt(image=[input_img], prompt=instruction, device=self.device, num_images_per_prompt=1)
        latents, ref = pipe.prepare_latents([vae_img.to(self.device, pe.dtype)], 1, self.in_channels, height, width, pe.dtype, self.device, torch.Generator("cpu").manual_seed(0))
        n_tokens = latents.shape[1]
        gh, gw = 2 * (height // (self.vsf * 2)), 2 * (width // (self.vsf * 2))
        assert gh * gw == n_tokens, (gh, gw, n_tokens)
        img_shapes = [[(1, ih // self.vsf, iw // self.vsf), (1, height // self.vsf, width // self.vsf)]]
        ipm_full = torch.cat([ipm, ipm.new_ones(ipm.shape[0], n_tokens // 4)], dim=1)
        mu = m.calculate_shift(n_tokens, pipe.scheduler.config.get("base_image_seq_len", 256), pipe.scheduler.config.get("max_image_seq_len", 4096), pipe.scheduler.config.get("base_shift", 0.5), pipe.scheduler.config.get("max_shift", 1.15))
        sig = np.linspace(1.0, 1.0 / self.steps, self.steps)
        m.retrieve_timesteps(pipe.scheduler, self.steps, self.device, sigmas=sig, mu=mu)
        sigmas = pipe.scheduler.sigmas.detach().float().cpu().tolist()
        cond = dict(pe=pe.detach(), pm=None if pm is None else pm.detach(), ipm=ipm_full.detach(), ref=ref.detach(), img_shapes=img_shapes, height=height, width=width, n_tokens=n_tokens, grid=(gh, gw), sigmas=sigmas, canvas_size=canvas.size)
        self._cond_cache[key] = cond
        return cond

    def _pred(self, cond, x, sigma):
        B = x.shape[0]
        hs = torch.cat([cond["ref"].expand(B, -1, -1), x.to(cond["pe"].dtype)], dim=1)
        pe = cond["pe"].expand(B, -1, -1)
        pm = None if cond["pm"] is None else cond["pm"].expand(B, -1)
        ipm = cond["ipm"].expand(B, -1)
        t = torch.full((B,), float(sigma), device=self.device, dtype=cond["pe"].dtype)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            out = self.transformer(hidden_states=hs, timestep=t, encoder_hidden_states=pe, encoder_hidden_states_mask=pm, img_shapes=cond["img_shapes"] * B, img_mask=ipm, return_dict=False)[0]
        return out[:, -x.shape[1]:].float()

    @torch.no_grad()
    def rollout(self, cond, G, seed, sde_steps=None):
        """G candidates from the same canvas and instruction. Returns a trajectory dict with latents per step (cpu, fp16),
        per-token log-probs per step (cpu, fp32), sigmas and decoded images."""
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn((G, cond["n_tokens"], self.in_channels), generator=gen, device=self.device, dtype=torch.float32)
        sig = cond["sigmas"]
        T = len(sig) - 1
        lat = [x.detach().to("cpu", torch.float16)]
        lps = []
        for i in range(T):
            v = self._pred(cond, x, sig[i])
            use_sde = sde_steps is None or i in sde_steps
            if use_sde:
                x_next, mean, std = sde_step(x, v, sig[i], sig[i + 1], sig[1], self.noise_level, generator=gen)
                lp = token_logprob(x_next, mean, std)
            else:
                x_next = x + v * (sig[i + 1] - sig[i])
                lp = torch.zeros(x.shape[:2], device=self.device)
            x = x_next
            lat.append(x.detach().to("cpu", torch.float16))
            lps.append(lp.detach().to("cpu", torch.float32))
        images = self.decode(x, cond)
        return dict(latents=torch.stack(lat), logp=torch.stack(lps), sigmas=sig, images=images, seed=seed, sde_steps=sde_steps)

    def logprob(self, cond, x_t, x_next, step, sigmas):
        """Differentiable per-token log-prob of x_next given x_t at step index `step`."""
        x_t = x_t.to(self.device, torch.float32)
        x_next = x_next.to(self.device, torch.float32)
        v = self._pred(cond, x_t, sigmas[step])
        mean = sde_mean(x_t, v, sigmas[step], sigmas[step + 1], sigmas[1], self.noise_level)
        _, _, step_std = sde_coeffs(sigmas[step], sigmas[step + 1], sigmas[1], self.noise_level)
        return token_logprob(x_next, mean, step_std)

    @torch.no_grad()
    def decode(self, x, cond):
        pipe = self.pipe
        lat = pipe._unpack_latents(x.to(pipe.vae.dtype), cond["height"], cond["width"], self.vsf)
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        std = torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        lat = lat * std + mean
        img = pipe.vae.decode(lat, return_dict=False)[0][:, :, 0]
        ims = pipe.image_processor.postprocess(img, output_type="pil")
        return [im.convert("RGB").resize(cond["canvas_size"]) for im in ims]

    @torch.no_grad()
    def encode_target(self, image, cond):
        """Packed latents of a target image at the group's resolution (for the regression warm-up)."""
        pipe = self.pipe
        img = image.convert("RGBA")
        t = pipe.image_processor.preprocess(img, width=cond["width"], height=cond["height"]).unsqueeze(2).to(self.device, pipe.vae.dtype)
        z = pipe._encode_vae_image(t, None)
        return pipe._pack_latents(z, 1, z.shape[1], int(z.shape[3]), int(z.shape[4])).float()

    def sigma_of(self, cond, u):
        m = _qi_mod()
        c = self.pipe.scheduler.config
        mu = m.calculate_shift(cond["n_tokens"], c.get("base_image_seq_len", 256), c.get("max_image_seq_len", 4096), c.get("base_shift", 0.5), c.get("max_shift", 1.15))
        return math.exp(mu) / (math.exp(mu) + (1.0 / float(u) - 1.0))

    def fm_loss(self, cond, x1, weights=None, generator=None, sigma=None):
        """Flow-matching loss at one noise level, weighted per token. x1: [1, N, C] clean latents."""
        if sigma is None:
            u = float(torch.sigmoid(torch.randn((), generator=generator)).clamp(1e-3, 1.0 - 1e-3))
            sigma = self.sigma_of(cond, u)
        noise = torch.randn(x1.shape, generator=generator).to(self.device)
        x1 = x1.to(self.device)
        x = (1.0 - sigma) * x1 + sigma * noise
        pred = self._pred(cond, x, sigma)
        per = ((pred - (noise - x1)) ** 2).mean(-1)[0]
        w = torch.ones_like(per) if weights is None else weights.to(self.device).float()
        return (per * w).sum() / w.sum().clamp(min=1e-6)

    def ode_reference(self, cond, G, seed):
        """Deterministic ODE sample with the same step schedule, for validating the adapter against the pipeline."""
        return self.rollout(cond, G, seed, sde_steps=set())
