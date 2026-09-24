from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-only", "--flux-only", dest="qwen_only", action="store_true")
    parser.add_argument("--skip-qwen", "--skip-flux", dest="skip_qwen", action="store_true")
    args = parser.parse_args()
    from cc_genclaw.config import load_genclaw_config
    try:
        load_genclaw_config()
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1
    problems = []
    if not args.qwen_only:
        for name in ["chainlit", "openai", "playwright", "PIL", "requests", "yaml"]:
            if importlib.util.find_spec(name) is None:
                problems.append(f"Missing dependency: {name}")
        key = os.environ.get("MAIN_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        endpoint = os.environ.get("MAIN_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
        if not key or "your-openai-compatible" in endpoint or not endpoint:
            problems.append("Fill in the planner/VLM gateway and API key in config.yaml (or environment).")
        if importlib.util.find_spec("playwright"):
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                if not Path(p.chromium.executable_path).is_file():
                    problems.append("Chromium missing: run python -m playwright install chromium.")
    from cc_genclaw.tools._image_gen.core import _load_default_provider, _resolve_model
    cfg = _load_default_provider()
    print("Image provider:", cfg["_name"])
    if cfg.get("protocol") in {"qwen_remote", "flux_remote"} and not args.skip_qwen:
        import requests
        endpoint = str(cfg.get("base_url") or "http://127.0.0.1:8188").rstrip("/")
        headers = {"Authorization": "Bearer " + cfg["api_key"]} if cfg.get("api_key") else {}
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(endpoint + "/health", headers=headers, timeout=10)
                response.raise_for_status()
                result = response.json()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result.get("model_files_ready"):
                problems.append("Worker model files are incomplete.")
            if result.get("backend") != _resolve_model(cfg, False):
                problems.append("Worker backend differs from QWEN_IMAGE_MODEL_NAME (mock is not a real Qwen worker).")
        except (requests.RequestException, ValueError) as exc:
            problems.append(f"Qwen worker health check failed: {type(exc).__name__}. Check the worker, tunnel and token.")
    for item in problems:
        print("NOT READY:", item)
    if not problems:
        print("Selected checks passed. Real generation has not been tested by this command.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
