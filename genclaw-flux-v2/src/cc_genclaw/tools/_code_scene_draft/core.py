from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from . import generate as G
from .render import render_to_png

logger = logging.getLogger("cc_genclaw.scene_draft")

_MAX_CROP_ROUNDS = 2
_MAX_REVISE_ROUNDS = 3


def draft_scene(
    prompt: str,
    output_dir: str,
    self_review: bool = True,
) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    total_t0 = time.time()
    result: Dict[str, Any] = {"status": "success", "timing": {}}

    try:
        t0 = time.time()
        gen = G.generate(prompt)
        svg_code = gen["svg_code"]
        svg_description = gen["svg_description"]
        layout = gen.get("layout")
        code_type = gen["code_type"]
        result["timing"]["generate"] = round(time.time() - t0, 1)
        logger.info("[scene_draft] SVG generated: %d chars, type=%s", len(svg_code), code_type)

        svg_ext = "html" if code_type == "html" else "svg"
        (Path(output_dir) / f"step1.{svg_ext}").write_text(svg_code, encoding="utf-8")

        t0 = time.time()
        draft_png_path = str(Path(output_dir) / "step2_draft.png")
        render_to_png(svg_code, draft_png_path, code_type=code_type)
        result["timing"]["render"] = round(time.time() - t0, 1)

        review_passed = True
        review_verdict = "SKIPPED"
        review_feedback = ""

        if self_review:
            t0 = time.time()
            rev = G.review(prompt, draft_png_path, svg_code=svg_code)
            review_passed = rev["passed"]
            review_verdict = rev.get("verdict", "FAIL")
            review_feedback = rev.get("feedback") or ""

            crop_rounds = 0
            while (not review_passed) and review_verdict == "OPTIMIZE" and crop_rounds < _MAX_CROP_ROUNDS:
                crop_rounds += 1
                svg_code = G._apply_viewbox_crop(svg_code, rev["viewbox_params"])
                draft_png_path = str(Path(output_dir) / f"step3_crop{crop_rounds}.png")
                render_to_png(svg_code, draft_png_path, code_type=code_type)
                rev = G.review(prompt, draft_png_path, svg_code=svg_code)
                review_passed = rev["passed"]
                review_verdict = rev.get("verdict", "FAIL")
                review_feedback = rev.get("feedback") or ""
                if review_passed:
                    break
            if (not review_passed) and review_verdict == "OPTIMIZE":
                review_passed = True
                review_feedback = ""

            revise_rounds = 0
            while (not review_passed) and review_verdict == "FAIL" and revise_rounds < _MAX_REVISE_ROUNDS:
                revise_rounds += 1
                logger.info("[scene_draft] FAIL revise round %d: %s", revise_rounds, review_feedback)
                revised = G.revise(prompt, svg_code, review_feedback)
                svg_code = revised["svg_code"]
                svg_description = revised["svg_description"]
                layout = revised.get("layout")
                code_type = revised["code_type"]

                draft_png_path = str(Path(output_dir) / f"step3_revise{revise_rounds}.png")
                render_to_png(svg_code, draft_png_path, code_type=code_type)
                rev = G.review(prompt, draft_png_path, svg_code=svg_code)
                review_passed = rev["passed"]
                review_verdict = rev.get("verdict", "FAIL")
                review_feedback = rev.get("feedback") or ""

                if review_passed:
                    break
                if review_verdict == "OPTIMIZE":
                    svg_code = G._apply_viewbox_crop(svg_code, rev["viewbox_params"])
                    draft_png_path = str(Path(output_dir) / f"step3_revise{revise_rounds}_crop.png")
                    render_to_png(svg_code, draft_png_path, code_type=code_type)
                    review_passed = True
                    review_verdict = "OPTIMIZE"
                    review_feedback = ""
                    break

            result["timing"]["review"] = round(time.time() - t0, 1)

        (Path(output_dir) / f"final.{('html' if code_type == 'html' else 'svg')}").write_text(
            svg_code, encoding="utf-8"
        )
        if svg_description:
            try:
                if layout is not None:
                    (Path(output_dir) / "layout.json").write_text(
                        json.dumps(layout, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                else:
                    (Path(output_dir) / "layout.json").write_text(svg_description, encoding="utf-8")
            except Exception:
                pass

        result.update({
            "svg_code": svg_code,
            "svg_description": svg_description,
            "layout": layout,
            "code_type": code_type,
            "draft_png_path": draft_png_path,
            "review_passed": review_passed,
            "review_verdict": review_verdict,
            "review_feedback": review_feedback,
        })
        result["timing"]["total"] = round(time.time() - total_t0, 1)
        logger.info("[scene_draft] complete in %ss", result["timing"]["total"])

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        logger.error("[scene_draft] failed: %s", e)

    return result

