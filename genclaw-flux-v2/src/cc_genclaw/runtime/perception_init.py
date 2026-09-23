from __future__ import annotations

from typing import Optional

from ..perception import DEFAULT_PERCEPTION, parse_perception, run_perception
from .perception_state import get_perception_state


def run_perception_init(
    user_input: str,
    user_image_path: Optional[str] = None,
) -> None:
    state = get_perception_state()
    if state.is_initialized():
        return

    raw = None
    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            raw = run_perception(user_input, user_image_path)
            if isinstance(raw, dict):
                break
            import sys

            print(
                f"[perception_init] attempt {attempt + 1}: "
                f"run_perception returned non-dict {type(raw).__name__}",
                file=sys.stderr,
            )
            raw = None
        except Exception as e:
            import sys

            last_error = e
            print(
                f"[perception_init] attempt {attempt + 1} failed: "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )
            raw = None

    if isinstance(raw, dict):
        note = parse_perception(raw)
        _fill_state(state, note, user_image_path, fallback=False)
    else:
        print(
            f"[perception_init] using DEFAULT_PERCEPTION sentinel "
            f"(last_error={last_error!r})",
            file=sys.stderr,
        )
        _fill_state(state, DEFAULT_PERCEPTION, user_image_path, fallback=True)


def _fill_state(state, note, user_image_path: Optional[str], fallback: bool) -> None:
    state.painter_thoughts = getattr(note, "painter_thoughts", "")
    state.image_intent = getattr(note, "image_intent", "")
    state.planning_needs = [
        n.model_dump() if hasattr(n, "model_dump") else dict(n)
        for n in getattr(note, "planning_needs", [])
    ]
    state.notes_for_planner = list(getattr(note, "notes_for_planner", []))
    state.user_image_path = user_image_path
    state._initialized = True
    state._fallback_used = fallback

