from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

KINDS = ("llm", "vlm", "render", "diffusion", "reuse")
PHASES = ("plan", "initial", "repair")


class CostMeter:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = {k: 1.0 for k in KINDS}
        if weights:
            self.weights.update({str(k): float(v) for k, v in weights.items()})
        self.events: List[Dict[str, Any]] = []
        self.phase = "initial"
        self.started = time.time()

    def record(
        self,
        kind: str,
        node: Optional[str] = None,
        seconds: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        phase: Optional[str] = None,
        label: str = "",
    ) -> Dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"unknown cost kind {kind!r}")
        event = {
            "kind": kind,
            "node": node,
            "seconds": float(seconds),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "phase": phase or self.phase,
            "label": label,
            "units": float(self.weights.get(kind, 1.0)),
            "t": time.time() - self.started,
        }
        self.events.append(event)
        return event

    def _select(self, phase: Optional[str] = None, node: Optional[str] = None, kind: Optional[str] = None):
        for e in self.events:
            if phase is not None and e["phase"] != phase:
                continue
            if node is not None and e["node"] != node:
                continue
            if kind is not None and e["kind"] != kind:
                continue
            yield e

    def units(self, phase: Optional[str] = None, node: Optional[str] = None, kind: Optional[str] = None) -> float:
        return float(sum(e["units"] for e in self._select(phase, node, kind)))

    def counts(self, phase: Optional[str] = None, node: Optional[str] = None) -> Dict[str, int]:
        out = {k: 0 for k in KINDS}
        for e in self._select(phase, node):
            out[e["kind"]] += 1
        return out

    def tokens(self, phase: Optional[str] = None) -> Dict[str, int]:
        p = sum(e["prompt_tokens"] for e in self._select(phase))
        c = sum(e["completion_tokens"] for e in self._select(phase))
        return {"prompt": int(p), "completion": int(c)}

    def seconds(self, phase: Optional[str] = None) -> float:
        return float(sum(e["seconds"] for e in self._select(phase)))

    def since(self, index: int) -> float:
        return float(sum(e["units"] for e in self.events[index:]))

    def mark(self) -> int:
        return len(self.events)

    def by_node(self) -> Dict[str, float]:
        acc: Dict[str, float] = defaultdict(float)
        for e in self.events:
            acc[str(e["node"])] += e["units"]
        return dict(acc)

    def summary(self) -> Dict[str, Any]:
        return {
            "units": self.units(),
            "counts": self.counts(),
            "tokens": self.tokens(),
            "seconds": round(self.seconds(), 3),
            "by_phase": {p: {"units": self.units(phase=p), "counts": self.counts(phase=p)} for p in PHASES},
            "by_node": self.by_node(),
            "events": len(self.events),
        }
