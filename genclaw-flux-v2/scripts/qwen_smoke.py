from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A red ceramic cup on a wooden table, studio photograph.")
    parser.add_argument("--prompt-file", type=Path, help="Read the image prompt from a UTF-8 text file")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-mock", action="store_true", help="Explicitly allow a labeled transport-test image.")
    args = parser.parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else args.prompt
    if not prompt:
        parser.error("Image prompt cannot be empty")
    from cc_genclaw.config import load_genclaw_config
    from cc_genclaw.tools._image_gen import generate_image
    load_genclaw_config()
    if args.allow_mock:
        os.environ["QWEN_IMAGE_ALLOW_MOCK"] = "1"
    try:
        path = Path(generate_image(prompt, [str(Path(p).resolve()) for p in args.reference]))
        if args.output:
            target = args.output.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target != path:
                shutil.copy2(path, target)
                if path.with_suffix(".json").exists():
                    shutil.copy2(path.with_suffix(".json"), target.with_suffix(".json"))
            path = target
        print(path)
        if path.with_suffix(".json").exists():
            print(path.with_suffix(".json").read_text())
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
