"""Character draft crops stay local without losing foreground parts."""
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "draftdag_poc"))
from branch_dag import character_crop_box, crop_character_draft


def test_character_crop_excludes_background_and_contains_subject(tmp_path):
    layout = {"compositional_deconstruction": {"elements": [
        {"bbox": [0, 780, 1600, 900], "desc": "grass band"},
        {"bbox": [700, 120, 900, 820], "desc": "woman"},
    ]}}
    Image.new("RGB", (1600, 900), "white").save(tmp_path / "draft.png")
    (tmp_path / "layout.json").write_text(json.dumps(layout))
    metadata = crop_character_draft(tmp_path)
    box = metadata["crop_box_xyxy"]
    assert box[0] <= 700 and box[1] <= 120 and box[2] >= 900 and box[3] >= 820
    assert metadata["local_size"][0] < 1600
    assert metadata["local_size"][1] < 900
    assert all(side % 32 == 0 for side in metadata["local_size"])
    assert Image.open(tmp_path / "draft.png").size == tuple(metadata["local_size"])
    assert Image.open(tmp_path / "full_draft.png").size == (1600, 900)


def test_character_crop_requires_foreground_bbox():
    with pytest.raises(ValueError, match="foreground"):
        character_crop_box({"compositional_deconstruction": {"elements": [
            {"bbox": [0, 780, 1600, 900], "desc": "grass band"}
        ]}}, (1600, 900))
