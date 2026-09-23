from __future__ import annotations

import os
from cc_genclaw.llm import LLMConfig
from .prompts import load_prompt

def _code_llm_enabled() -> bool:
    return (os.environ.get("CODE_LLM_ENABLED") or "").strip().lower() in {"1", "true", "yes", "on"}


def _default_code_model() -> str:
    return LLMConfig.for_code_draft().model


if _code_llm_enabled():
    SVG_MODEL = os.environ.get("SVG_GEN_MODEL_NAME") or os.environ.get("CODE_LLM_MODEL_NAME") or _default_code_model()
    REVIEW_MODEL = os.environ.get("SVG_REVIEW_MODEL_NAME") or os.environ.get("CODE_LLM_MODEL_NAME") or _default_code_model()
else:
    SVG_MODEL = _default_code_model()
    REVIEW_MODEL = _default_code_model()

SVG_DELIMITER = "===SVG_CODE_START==="

GEN_BACKEND = "gemini"

def set_gen_backend(backend: str):
    global GEN_BACKEND
    GEN_BACKEND = backend

