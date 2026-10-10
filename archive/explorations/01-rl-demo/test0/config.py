from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

MODES = ("mock", "api")
RENDER_BACKENDS = ("auto", "playwright", "pil")
GENERATE_BACKENDS = ("mock", "remote")

DEFAULTS: Dict[str, Any] = {
    "MODE": "mock",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "OPENAI_API_KEY": "",
    "OPENAI_MODEL_NAME": "gpt-5.5",
    "PLANNER_MODEL": "",
    "EXECUTOR_MODEL": "",
    "REVIEW_MODEL": "",
    "JUDGE_BASE_URL": "",
    "JUDGE_API_KEY": "",
    "JUDGE_MODEL": "",
    "GENERATE_BACKEND": "mock",
    "GENERATE_BASE_URL": "http://127.0.0.1:8188",
    "GENERATE_API_KEY": "",
    "GENERATE_MODEL": "qwen-image-controlnet",
    "GENERATE_TIMEOUT": 900.0,
    "GENERATE_STEPS": 28,
    "GENERATE_GUIDANCE": 3.5,
    "GENERATE_STRENGTH": 0.7,
    "GENERATE_CONTROL_SCALE": 0.8,
    "GENERATE_SEED": 0,
    "RENDER_BACKEND": "auto",
    "RENDER_MAX_SIDE": 1024,
    "RUNS_DIR": str(ROOT / "runs"),
    "STORE_DIR": str(ROOT / "store"),
    "STORE_ENABLED": True,
    "NODE_REVIEW": True,
    "MAX_NODE_ROUNDS": 2,
    "MAX_SCENE_ROUNDS": 2,
    "MAX_FINAL_ROUNDS": 1,
    "LLM_TIMEOUT": 180.0,
    "LLM_RETRIES": 2,
    "LLM_MAX_TOKENS": 8192,
    "VLM_MAX_TOKENS": 1500,
    "MOCK_ERROR_RATE": 0.0,
    "MOCK_BASELINE_ERROR_SCALE": 1.5,
    "MOCK_SEED": 0,
    "COST_WEIGHTS": {"llm": 1.0, "vlm": 1.5, "render": 0.1, "diffusion": 10.0, "reuse": 0.0},
    "BENCH_FAULTS": 2,
}

_ENV_SKIP = {"COST_WEIGHTS"}


class Config:
    def __init__(self, values: Dict[str, Any]):
        self._values = dict(values)

    def __getattr__(self, name: str) -> Any:
        values = self.__dict__.get("_values", {})
        if name in values:
            return values[name]
        raise AttributeError(name)

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def set(self, name: str, value: Any) -> None:
        self._values[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._values)

    def with_overrides(self, **overrides: Any) -> "Config":
        values = dict(self._values)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return Config(values)

    def role_model(self, role: str) -> str:
        key = {"planner": "PLANNER_MODEL", "executor": "EXECUTOR_MODEL", "review": "REVIEW_MODEL"}.get(role, "")
        return str(self._values.get(key) or "") or str(self._values["OPENAI_MODEL_NAME"])

    def role_endpoint(self, role: str):
        if role == "judge":
            base = str(self._values.get("JUDGE_BASE_URL") or "") or str(self._values["OPENAI_BASE_URL"])
            key = str(self._values.get("JUDGE_API_KEY") or "") or str(self._values["OPENAI_API_KEY"])
            model = str(self._values.get("JUDGE_MODEL") or "") or str(self._values["OPENAI_MODEL_NAME"])
            return base, key, model
        return str(self._values["OPENAI_BASE_URL"]), str(self._values["OPENAI_API_KEY"]), self.role_model(role)

    @property
    def mock(self) -> bool:
        return str(self._values.get("MODE", "mock")).lower() == "mock"


def _coerce(default: Any, raw: str) -> Any:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    values: Dict[str, Any] = dict(DEFAULTS)
    candidate = Path(path) if path else Path(os.environ.get("DRAFTDAG_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if candidate.is_file():
        with open(candidate, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        for k, v in loaded.items():
            if isinstance(k, str):
                values[k] = v
    for k in list(values.keys()):
        if k in _ENV_SKIP:
            continue
        raw = os.environ.get(k)
        if raw is not None and raw != "":
            values[k] = _coerce(DEFAULTS.get(k, ""), raw)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                values[k] = v
    weights = dict(DEFAULTS["COST_WEIGHTS"])
    if isinstance(values.get("COST_WEIGHTS"), dict):
        weights.update({str(k): float(v) for k, v in values["COST_WEIGHTS"].items()})
    values["COST_WEIGHTS"] = weights
    if str(values["MODE"]).lower() not in MODES:
        raise ValueError(f"MODE must be one of {MODES}, got {values['MODE']!r}")
    if str(values["RENDER_BACKEND"]).lower() not in RENDER_BACKENDS:
        raise ValueError(f"RENDER_BACKEND must be one of {RENDER_BACKENDS}")
    if str(values["GENERATE_BACKEND"]).lower() not in GENERATE_BACKENDS:
        raise ValueError(f"GENERATE_BACKEND must be one of {GENERATE_BACKENDS}")
    return Config(values)
