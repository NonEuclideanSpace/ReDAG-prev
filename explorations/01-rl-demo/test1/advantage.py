from __future__ import annotations

import math
from typing import List, Sequence


def group_advantages(rewards: Sequence[float], eps: float = 1e-6, normalize_std: bool = True) -> List[float]:
    values = [float(r) for r in rewards]
    if not values:
        return []
    mean = sum(values) / len(values)
    if not normalize_std or len(values) < 2:
        return [v - mean for v in values]
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(var)
    return [(v - mean) / (std + eps) for v in values]


def edit_advantage(before: float, after: float, cost_units: float, lam: float) -> float:
    return float(after) - float(before) - float(lam) * float(cost_units)


def clip(value: float, limit: float = 5.0) -> float:
    return max(-limit, min(limit, float(value)))
