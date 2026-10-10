from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "config.yaml")

_FORCE_OVERRIDE_KEYS = {
    "TAVILY_API_KEY",
    "TAVILY_BASE_URL",
    "MAIN_LLM_ACTIVE_PROFILE",
    "MAIN_LLM_API_KEY",
    "MAIN_LLM_BASE_URL",
    "MAIN_LLM_MODEL_NAME",
    "CODE_LLM_ENABLED",
    "CODE_LLM_BASE_URL",
    "CODE_LLM_API_KEY",
    "CODE_LLM_MODEL_NAME",
    "SVG_GEN_MODEL_NAME",
    "SVG_REVIEW_MODEL_NAME",
    "SCENE_DRAFT_GEN_MODEL",
    "SCENE_DRAFT_REVIEW_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL_NAME",
    "QWEN_API_KEY",
    "QWEN_BASE_URL",
    "QWEN_MODEL_NAME",
    "QWEN_IMAGE_API_KEY",
    "QWEN_IMAGE_BASE_URL",
    "QWEN_IMAGE_MODEL_NAME",
}

_MAIN_LLM_PROFILE_KEYS = {
    "base_url": "MAIN_LLM_BASE_URL",
    "api_key": "MAIN_LLM_API_KEY",
    "model": "MAIN_LLM_MODEL_NAME",
    "model_name": "MAIN_LLM_MODEL_NAME",
}


def _apply_main_llm_profile(cfg: Dict[str, Any]) -> None:
    active = (
        os.environ.get("MAIN_LLM_ACTIVE_PROFILE")
        or str(cfg.get("MAIN_LLM_ACTIVE_PROFILE") or "")
    ).strip()
    if not active:
        return

    profiles = cfg.get("MAIN_LLM_PROFILES")
    if not isinstance(profiles, dict):
        raise ValueError("MAIN_LLM_ACTIVE_PROFILE is set but MAIN_LLM_PROFILES is missing or invalid.")

    profile = profiles.get(active)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(str(k) for k in profiles.keys())) or "(none)"
        raise ValueError(
            f"MAIN_LLM_ACTIVE_PROFILE={active!r} not found in MAIN_LLM_PROFILES. "
            f"Available profiles: {available}."
        )

    cfg["MAIN_LLM_ACTIVE_PROFILE"] = active
    for raw_key, value in profile.items():
        env_key = _MAIN_LLM_PROFILE_KEYS.get(str(raw_key).strip().lower())
        if env_key and value is not None:
            cfg[env_key] = str(value)


def load_genclaw_config(yaml_path: Optional[str] = None) -> Dict[str, Any]:
    path = (
        yaml_path
        or os.environ.get("GENCLAW_CONFIG_PATH")
        or DEFAULT_CONFIG_PATH
    )
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"GenClaw config.yaml not found at {p}. "
            f"Set GENCLAW_CONFIG_PATH env var to override."
        )

    with open(p, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    _apply_main_llm_profile(cfg)

    user_env_snapshot = {
        k: os.environ[k]
        for k in _FORCE_OVERRIDE_KEYS
        if k in os.environ
    }

    proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"}
    proxy_on = bool(cfg.get("proxy_on", False))
    for k, v in cfg.items():
        if isinstance(k, str) and k.isupper() and isinstance(v, str):
            if k in proxy_keys and not proxy_on:
                continue
            os.environ.setdefault(k, v)

    for k, v in user_env_snapshot.items():
        os.environ[k] = v
        cfg[k] = v

    qwen_base = os.environ.get("QWEN_BASE_URL")
    qwen_key = os.environ.get("QWEN_API_KEY")
    qwen_model = os.environ.get("QWEN_MODEL_NAME")
    aliases = {
        "OPENAI_BASE_URL": qwen_base,
        "MAIN_LLM_BASE_URL": qwen_base,
        "CODE_LLM_BASE_URL": qwen_base,
        "OPENAI_API_KEY": qwen_key,
        "MAIN_LLM_API_KEY": qwen_key,
        "CODE_LLM_API_KEY": qwen_key,
        "OPENAI_MODEL_NAME": qwen_model,
        "MAIN_LLM_MODEL_NAME": qwen_model,
        "CODE_LLM_MODEL_NAME": qwen_model,
    }
    for key, value in aliases.items():
        if value and not os.environ.get(key):
            os.environ[key] = value
            cfg[key] = value

    main_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("MAIN_LLM_BASE_URL")
    main_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("MAIN_LLM_API_KEY")
    if main_base and not os.environ.get("GEMINI_BASE_URL"):
        os.environ["GEMINI_BASE_URL"] = main_base
        cfg["GEMINI_BASE_URL"] = main_base
    if main_key and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = main_key
        cfg["GEMINI_API_KEY"] = main_key

    return cfg


def get_session_root(cfg: Optional[Dict[str, Any]] = None) -> str:
    env = os.environ.get("CC_GENCLAW_RUNS_DIR")
    if env:
        return env
    return str(Path(__file__).resolve().parents[2] / "runs")
