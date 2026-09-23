from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


class FirstImageReady(Exception):
    pass


class SinglePassPolicy:
    def __init__(self, report):
        self.report = report
        self.drafts = []

    def before(self, ctx):
        from cc_genclaw.hooks import BeforeToolCallResult
        if ctx.tool_name == "code_scene_draft":
            ctx.args["self_review"] = False
        if ctx.tool_name in {"code_scene_draft", "code_text_draft"} and self.drafts:
            return BeforeToolCallResult(block=True, reason="Keep the existing draft. Do not regenerate it for quality; call i2i with it.")
        if ctx.tool_name == "t2i":
            return BeforeToolCallResult(block=True, reason="This experiment requires a GenClaw draft followed by i2i. Direct t2i is not allowed.")
        if ctx.tool_name == "i2i":
            source = str(Path(ctx.args.get("image_path") or "").resolve())
            if source not in self.drafts:
                return BeforeToolCallResult(block=True, reason="First call code_scene_draft or code_text_draft, then pass its real output path as i2i.image_path.")
        return None

    def accept(self, tool_name, payload, is_error):
        if is_error or payload.get("status") != "success":
            return
        if tool_name in {"code_scene_draft", "code_text_draft"}:
            path = payload.get("draft_png_path") or payload.get("final_path")
            if path and Path(path).is_file():
                path = str(Path(path).resolve())
                self.drafts.append(path)
                self.report.setdefault("drafts", []).append({"tool": tool_name, "path": path})
                self.report["draft_logic_executed"] = True


def attach_single_pass(run, report):
    policy = SinglePassPolicy(report)
    run.context.system_prompt += """\n\nUSER EXPERIMENT POLICY (overrides quality repair instructions):
Use the real GenClaw code_scene_draft or code_text_draft tool for every case.
Generate one draft and then call i2i with that draft as the main image_path;
include the user's input image and retrieved references as additional references when relevant.
Do not use direct t2i. Do not repeat a draft or image because its quality is poor.
Set scene-draft self_review=false. The batch runner will review and save the first
Flux result, including failures, then continue to the next original prompt.
"""
    run.config.hooks.register_before_tool_call(policy.before)
    run.context.tools = [t for t in run.context.tools if t.name != "vlm_review"]
    previous_followup = run.config.get_followup_messages

    def followup():
        from cc_genclaw.types import UserMessage
        if policy.drafts:
            return [UserMessage(content="The rendered draft is ready. Call i2i once using this draft as image_path: " + policy.drafts[-1], is_system_reminder=True)]
        return previous_followup() if previous_followup else []

    run.config.get_followup_messages = followup
    return policy


def collect_flux(session):
    collected = []
    for path in sorted(Path(session).rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError, OSError):
            continue
        if isinstance(data, dict) and data.get("backend") in {"mock", "flux-fill", "flux-dev"}:
            if data["backend"] == "mock":
                raise RuntimeError("Mock output is forbidden in paper reproduction")
            collected.append({"metadata_path": str(path), **data})
    return collected


def archive_generated_images(report, folder, case):
    session = report.get("run_dir")
    if not session:
        return
    flux = collect_flux(session)
    report["flux_generations"] = flux
    report["used_flux"] = bool(flux)
    archive = (Path(folder) / case["output"]).parent / "attempts" / case["id"] / Path(session).name
    reviews = {str(Path(r.get("image_path", "")).resolve()): r for r in report.get("reviews", [])}
    images = []
    for index, generation in enumerate(flux, 1):
        source = Path(generation["metadata_path"]).with_suffix(".png")
        if not source.is_file():
            continue
        review = reviews.get(str(source.resolve()))
        verdict = "unreviewed" if review is None else ("pass" if review.get("passed") else "failed")
        target = archive / f"{case['id']}_attempt-{index:02d}_{source.stem}.png"
        archive.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise RuntimeError("Refusing to overwrite a different archived image")
        else:
            shutil.copy2(source, target)
        images.append({"path": str(target.resolve()), "source": str(source.resolve()),
                       "review_status": verdict, "review": review})
    report["saved_attempts"] = images

