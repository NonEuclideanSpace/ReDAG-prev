from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionDirState:

    _path: Optional[str] = None

    def current(self) -> str:
        if self._path is None:
            self._path = self._make_new()
        return self._path

    def reset(self, root: Optional[str] = None) -> str:
        self._path = self._make_new(root)
        return self._path

    @staticmethod
    def _make_new(root: Optional[str] = None) -> str:
        if root is None:
            from cc_genclaw.config import get_session_root
            root = get_session_root()
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        out = Path(root) / sid
        out.mkdir(parents=True, exist_ok=True)
        return str(out)


_GLOBAL_SESSION_DIR = SessionDirState()


def get_session_dir() -> str:
    return _GLOBAL_SESSION_DIR.current()


def reset_session_dir(root: Optional[str] = None) -> str:
    return _GLOBAL_SESSION_DIR.reset(root)


def tool_output_dir(tool_name: str, turn_hint: Optional[int] = None) -> str:
    base = Path(get_session_dir()) / tool_name
    tag = f"turn-{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:4]}"
    out = base / tag
    out.mkdir(parents=True, exist_ok=True)
    return str(out)

