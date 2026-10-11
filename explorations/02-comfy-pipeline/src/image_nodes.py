"""Headless implementations of the image-node settings actually present in pipe.json.

Source references and upstream snapshots are in ../provenance. Masks are
foreground alpha (white = keep); ComfyUI LoadImage's MASK is inverse alpha.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageOps


def scale_image(image, aspect_ratio, proportional_width, proportional_height,
                fit, method, round_to_multiple, scale_to_side, scale_to_length,
                background_color, mask=None):
    width, height = image.size
    if aspect_ratio == 'original':
        ratio = width / height
    elif aspect_ratio == 'custom':
        ratio = proportional_width / proportional_height
    else:
        a, b = map(int, aspect_ratio.split(':'))
        ratio = a / b
    length = scale_to_length
    if scale_to_side == 'width' or (scale_to_side == 'longest' and ratio > 1) or (scale_to_side == 'shortest' and ratio <= 1):
        size = (length, int(length / ratio))
    elif scale_to_side == 'height' or scale_to_side in ('longest', 'shortest'):
        size = (int(length * ratio), length)
    else:
        raise ValueError(f'Unsupported scale_to_side: {scale_to_side}')
    if round_to_multiple != 'None':
        multiple = int(round_to_multiple)
        size = tuple(math.ceil(x / multiple) * multiple for x in size)
    resample = getattr(Image.Resampling, method.upper())

    def resize(source, background):
        if fit == 'fill':
            return source.resize(size, resample)
        if fit == 'crop':
            return ImageOps.fit(source, size, method=resample)
        if fit == 'letterbox':
            return ImageOps.pad(source, size, method=resample, color=background)
        raise ValueError(f'Unsupported fit: {fit}')
    result = resize(image.convert('RGB'), background_color)
    result_mask = resize(mask.convert('L'), 0) if mask is not None else Image.new('L', size)
    return result, result_mask, [width, height], *size


def grow_mask(mask, expand, tapered_corners):
    if expand == 0:
        return (mask.copy(),)
    from scipy import ndimage
    c = 0 if tapered_corners else 1
    footprint = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]])
    array = np.asarray(mask)
    for _ in range(abs(expand)):
        op = ndimage.grey_erosion if expand < 0 else ndimage.grey_dilation
        array = op(array, footprint=footprint)
    return (Image.fromarray(array),)


def mask_to_image(mask, color, background, invert):
    alpha = ImageOps.invert(mask) if invert else mask
    foreground = Image.new('RGB', mask.size, ImageColor.getrgb(color))
    backdrop = Image.new('RGB', mask.size, ImageColor.getrgb(background))
    return (Image.composite(foreground, backdrop, alpha),)


def blend_image(background_image, layer_image, invert_mask, blend_mode, opacity,
                x_percent, y_percent, mirror, scale, aspect_ratio, rotate,
                transform_method, anti_aliasing, layer_mask=None):
    # All five serialized subgraphs use this exact identity-transform setting.
    settings = (blend_mode, opacity, x_percent, y_percent, mirror, scale, aspect_ratio, rotate, anti_aliasing)
    if settings != ('normal', 100, 50, 50, 'None', 1, 1, 0, 0):
        raise ValueError(f'Unimplemented ImageBlendAdvance setting: {settings}')
    if layer_mask is None:
        alpha = layer_image.getchannel('A') if layer_image.mode == 'RGBA' else Image.new('L', layer_image.size, 255)
    else:
        alpha = ImageOps.invert(layer_mask) if invert_mask else layer_mask
    rgb = background_image.convert('RGB').copy()
    x = int(rgb.width / 2 - layer_image.width / 2)
    y = int(rgb.height / 2 - layer_image.height / 2)
    rgb.paste(layer_image.convert('RGB'), (x, y), alpha)
    outmask = Image.new('L', rgb.size)
    outmask.paste(alpha, (x, y))
    return rgb, outmask


@dataclass
class CompositionConfig:
    size: tuple
    images: list
    masks: list
    invert_mask: bool
    padding: int


def compositor_config(width, height, padding, normalizeHeight, onConfigChanged,
                      invertMask, initialized, **inputs):
    if normalizeHeight or padding:
        raise ValueError('This workflow uses normalizeHeight=False and padding=0')
    images, masks = [], []
    for i in range(1, 9):
        image = inputs.get(f'image{i}')
        mask = inputs.get(f'mask{i}')
        if image is not None:
            image = image.convert('RGBA')
            if mask is not None:
                image.putalpha(ImageOps.invert(mask) if invertMask else mask)
        images.append(image)
        masks.append(mask)
    config = CompositionConfig((width, height), images, masks, invertMask, padding)
    return config, config


def composite(config, fabricData, imageName=None, extendedConfig=None, tools=None):
    """Reproduce the Fabric.js canvas render instead of reloading a missing temp PNG.

    xwidth/xheight describe the old source dimensions; they are not a requested
    size for replacement images. Fabric transforms act on current source pixels.
    bboxes in this workflow swap height/width; rendering uses transforms instead.
    """
    import json
    transforms = json.loads(fabricData)['transforms']
    canvas = Image.new('RGBA', config.size, (0, 0, 0, 0))
    layers = []
    for i, image in enumerate(config.images):
        if image is None:
            continue
        t = transforms[i]
        if t.get('skewX', 0) or t.get('skewY', 0) or t.get('originX', 'left') != 'left' or t.get('originY', 'top') != 'top':
            raise ValueError('This renderer handles the transforms serialized in pipe.json')
        layer = image.copy()
        if t.get('flipX', False):
            layer = ImageOps.mirror(layer)
        if t.get('flipY', False):
            layer = ImageOps.flip(layer)
        size = (max(1, round(layer.width * t['scaleX'])), max(1, round(layer.height * t['scaleY'])))
        layer = layer.resize(size, Image.Resampling.LANCZOS)
        if t.get('angle', 0):
            raise ValueError('Rotation is zero in this workflow; nonzero rotation requires Fabric origin compensation')
        position = (round(t['left']), round(t['top']))
        placed = Image.new('RGBA', canvas.size)
        placed.alpha_composite(layer, position)
        canvas = Image.alpha_composite(canvas, placed)
        layers.append(placed)
    return fabricData, canvas.convert('RGB'), layers


class SAM2BackgroundRemoval:
    """RMBG replacement requested by the user: local SAM2.1, box prompted."""
    def __init__(self, model_path='/home/hdd/lpz/model/sam2.1', device='cuda'):
        import torch
        from transformers import Sam2Model, Sam2Processor
        self.torch = torch
        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
        self.model = Sam2Model.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.records = []

    def __call__(self, image, model, sensitivity, process_res, mask_blur,
                 mask_offset, invert_output, refine_foreground, background, background_color):
        if (sensitivity, mask_blur, mask_offset, invert_output, refine_foreground, background) != (1, 0, 0, False, False, 'Alpha'):
            raise ValueError('RMBG adapter implements the parameters serialized in this workflow')
        w, h = image.size
        # Qwen prompts leave white margins. A broad box and positive body points
        # select the whole person, including clothing, rather than a face only.
        box = [w * .06, h * .02, w * .94, h * .98]
        points = [[w*.5, h*.30], [w*.5, h*.48], [w*.03, h*.5], [w*.97, h*.5]]
        kwargs = dict(input_boxes=[[box]], input_points=[[points]], input_labels=[[[1, 1, 0, 0]]])
        inputs = self.processor(images=image, return_tensors='pt', **kwargs).to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs, multimask_output=True)
        scores = output.iou_scores[0, 0].float().cpu()
        masks = self.processor.post_process_masks(output.pred_masks.float().cpu(), inputs['original_sizes'])[0]
        # A studio background can receive a higher predicted SAM IoU than the
        # subject. Use the known white-background generation condition to rank
        # SAM's three masks; the exported mask remains SAM's unmodified output.
        candidates = masks[0].numpy().astype(bool)
        proposal = np.asarray(image.convert('RGB')).min(axis=-1) < 220
        similarities = []
        for candidate in candidates:
            intersection = (candidate & proposal).sum()
            union = (candidate | proposal).sum()
            similarities.append(float(intersection / max(1, union)))
        best = int(np.argmax(similarities))
        array = masks[0, best].to(self.torch.uint8).numpy() * 255
        alpha = Image.fromarray(array)
        if alpha.getbbox() is None:
            raise RuntimeError('SAM2 returned an empty foreground')
        rgba = image.convert('RGBA')
        rgba.putalpha(alpha)
        self.records.append(dict(size=[w,h], box=box, points=points, iou_score=float(scores[best]),
                                 candidate_scores=scores.tolist(), candidate_white_bg_ious=similarities,
                                 selected_candidate=best,
                                 foreground_fraction=float((array>0).mean()), backend='SAM2.1'))
        return rgba, alpha, image.convert('RGB')
