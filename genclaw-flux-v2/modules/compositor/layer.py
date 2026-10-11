"""Source assets and editable placement, independent of any diffusion model."""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Tuple
import math

from PIL import Image, ImageOps


@dataclass
class ImageLayer:
    """Position is the top-left of the cropped, scaled asset BEFORE rotation.

    Positive rotation is clockwise about that asset's center. Scale is uniform;
    transparent borders are trimmed by default, not stretched to canvas size.
    Alpha uses ordinary PNG semantics: 255 visible, 0 transparent.
    """

    name: str
    image_path: str
    position: Tuple[int, int] = (0, 0)
    scale: float = 1.0
    rotation: float = 0.0
    z_order: int = 0
    opacity: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    depends_on: list = field(default_factory=list)
    trim_alpha: bool = True
    image: Image.Image = field(init=False, repr=False)
    alpha_mask: Image.Image = field(init=False, repr=False)
    source_size: Tuple[int, int] = field(init=False)
    crop_bbox: Tuple[int, int, int, int] = field(init=False)
    source_sha256: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Layer name must be nonempty")
        if len(self.position) != 2 or any(not isinstance(v, int) for v in self.position):
            raise ValueError("Position must contain two integer pixel coordinates")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Scale must be finite and positive")
        if not math.isfinite(self.rotation):
            raise ValueError("Rotation must be finite")
        if not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValueError("Opacity must be between 0 and 1")
        if not isinstance(self.z_order, int):
            raise ValueError("z_order must be an integer")
        path = Path(self.image_path).resolve()
        self.image_path = str(path)
        self.source_sha256 = sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGBA")
        self.source_size = image.size
        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError(f"Layer {self.name} has no visible pixels")
        self.crop_bbox = bbox if self.trim_alpha else (0, 0, *image.size)
        self.image = image.crop(self.crop_bbox)
        self.alpha_mask = self.image.getchannel("A")

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def scaled_size(self):
        return tuple(max(1, round(v * self.scale)) for v in self.image.size)
