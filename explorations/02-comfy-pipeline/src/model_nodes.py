"""Call upstream ComfyUI model nodes; no sampler/conditioning approximations."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/ComfyUI'))

import nodes
import folder_paths
from comfy_extras.nodes_cfg import CFGNorm
from comfy_extras.nodes_edit_model import ReferenceLatent
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus


def image_tensor(image):
    return torch.from_numpy(np.asarray(image.convert('RGB')).astype(np.float32) / 255)[None]


class ComfyModelNodes:
    def __init__(self):
        weights = str(ROOT / 'models')
        for category in ('unet', 'diffusion_models', 'text_encoders', 'vae'):
            folder_paths.add_model_folder_path(category, weights)
        package = ROOT / 'vendor/ComfyUI-GGUF'
        spec = importlib.util.spec_from_file_location('workflow_gguf', package / '__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.unet_loader = module.NODE_CLASS_MAPPINGS['UnetLoaderGGUF']()

    def execute(self, kind, kwargs):
        kwargs = dict(kwargs)
        if kind == 'UnetLoaderGGUF':
            return self.unet_loader.load_unet(**kwargs)
        if kind == 'CLIPLoader':
            return nodes.CLIPLoader().load_clip(**kwargs)
        if kind == 'VAELoader':
            return nodes.VAELoader().load_vae(**kwargs)
        if kind == 'ModelSamplingAuraFlow':
            return ModelSamplingAuraFlow().patch_aura(**kwargs)
        if kind == 'CFGNorm':
            return CFGNorm.execute(**kwargs).result
        if kind == 'TextEncodeQwenImageEditPlus':
            for key in ('image1', 'image2', 'image3'):
                if key in kwargs:
                    kwargs[key] = image_tensor(kwargs[key])
            return TextEncodeQwenImageEditPlus.execute(**kwargs).result
        if kind == 'ConditioningZeroOut':
            return nodes.ConditioningZeroOut().zero_out(**kwargs)
        if kind == 'VAEEncode':
            kwargs['pixels'] = image_tensor(kwargs['pixels'])
            return nodes.VAEEncode().encode(**kwargs)
        if kind == 'ReferenceLatent':
            return ReferenceLatent.execute(**kwargs).result
        if kind == 'KSamplerAdvanced':
            return nodes.KSamplerAdvanced().sample(**kwargs)
        if kind == 'VAEDecode':
            tensor = nodes.VAEDecode().decode(**kwargs)[0]
            image = Image.fromarray((tensor[0].cpu().numpy() * 255).clip(0,255).astype(np.uint8))
            return (image,)
        raise ValueError(f'Unsupported model node: {kind}')
