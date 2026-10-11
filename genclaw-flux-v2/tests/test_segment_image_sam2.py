"""Fast unit tests for the SAM2 cutout helpers (no model/GPU required)."""
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from segment_image_sam2 import make_cutout, validate_prompts


def test_prompt_validation():
    validate_prompts((100, 80), (10, 5, 50, 60), [], [])
    validate_prompts((100, 80), None, [(40, 30)], [(10, 10)])
    with pytest.raises(ValueError, match="Provide"):
        validate_prompts((100, 80), None, [], [])
    with pytest.raises(ValueError, match="Box"):
        validate_prompts((100, 80), (10, 5, 110, 60), [], [])
    with pytest.raises(ValueError, match="outside"):
        validate_prompts((100, 80), None, [(100, 30)], [])


def test_cutout_has_real_transparency_and_optional_crop():
    image = Image.new("RGB", (20, 10), (120, 50, 30))
    mask = Image.new("L", image.size, 0)
    for y in range(3, 7):
        for x in range(6, 12):
            mask.putpixel((x, y), 255)
    full = make_cutout(image, mask)
    assert full.mode == "RGBA" and full.size == image.size
    assert full.getpixel((0, 0))[3] == 0
    assert full.getpixel((8, 5)) == (120, 50, 30, 255)
    cropped = make_cutout(image, mask, crop=True, margin=1)
    assert cropped.size == (8, 6)
    assert cropped.getpixel((0, 0))[3] == 0


def test_empty_mask_is_rejected():
    image = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="empty mask"):
        make_cutout(image, Image.new("L", image.size, 0))
