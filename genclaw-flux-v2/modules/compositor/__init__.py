"""Explicit, provenance-preserving RGBA scene composition (no ComfyUI runtime)."""
from .layer import ImageLayer
from .composition import SceneComposition

__all__ = ["ImageLayer", "SceneComposition"]
