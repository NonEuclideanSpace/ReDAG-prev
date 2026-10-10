from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dag import Node, canonical_spec


class Store:
    def __init__(self, root: str, enabled: bool = True):
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index: Dict[str, Dict[str, Any]] = {}
        if self.index_path.is_file():
            try:
                self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.index = {}
        self.hits = 0
        self.misses = 0
        self.puts = 0

    def _save(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.index_path)

    def key(self, node: Node) -> str:
        return node.spec_key()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        entry = self.index.get(key)
        if not entry:
            self.misses += 1
            return None
        frag_path = self.root / entry["fragment"]
        if not frag_path.is_file():
            self.index.pop(key, None)
            self.misses += 1
            return None
        self.hits += 1
        entry = dict(entry)
        entry["fragment_text"] = frag_path.read_text(encoding="utf-8")
        entry["uses"] = int(entry.get("uses", 0)) + 1
        self.index[key]["uses"] = entry["uses"]
        self.index[key]["last_used"] = time.time()
        self._save()
        return entry

    def put(self, key: str, node: Node, fragment: str, png: Optional[str], checks: Optional[Dict[str, Any]], review: Optional[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        folder = self.root / key[:2]
        folder.mkdir(parents=True, exist_ok=True)
        frag_rel = f"{key[:2]}/{key}.svg"
        (self.root / frag_rel).write_text(fragment, encoding="utf-8")
        png_rel = None
        if png and Path(png).is_file():
            png_rel = f"{key[:2]}/{key}.png"
            shutil.copy2(png, self.root / png_rel)
        self.index[key] = {
            "key": key,
            "type": node.type,
            "spec": canonical_spec(node.spec),
            "desc": node.desc(),
            "fragment": frag_rel,
            "png": png_rel,
            "checks_score": (checks or {}).get("score"),
            "review_passed": (review or {}).get("passed"),
            "created": time.time(),
            "uses": 0,
        }
        self.puts += 1
        self._save()

    def catalog(self, limit: int = 40) -> List[Dict[str, Any]]:
        entries = sorted(self.index.values(), key=lambda e: (-int(e.get("uses", 0)), -float(e.get("created", 0))))
        return [{"key": e["key"], "type": e["type"], "spec": e["spec"], "desc": e.get("desc")} for e in entries[:limit]]

    def candidates(self, node: Node, limit: int = 5) -> List[Dict[str, Any]]:
        want = set(str(node.desc()).lower().split()) | set(node.colors())
        scored = []
        for e in self.index.values():
            if e["type"] != node.type:
                continue
            have = set(str(e.get("desc") or "").lower().split()) | set(e["spec"].get("colors") or [])
            overlap = len(want & have)
            if overlap:
                scored.append((overlap, e))
        scored.sort(key=lambda t: -t[0])
        return [{"key": e["key"], "type": e["type"], "spec": e["spec"], "desc": e.get("desc"), "overlap": s} for s, e in scored[:limit]]

    def png_path(self, key: str) -> Optional[str]:
        entry = self.index.get(key)
        if entry and entry.get("png"):
            p = self.root / entry["png"]
            return str(p) if p.is_file() else None
        return None

    def stats(self) -> Dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "puts": self.puts, "entries": len(self.index)}

    def reset_stats(self) -> None:
        self.hits = self.misses = self.puts = 0
