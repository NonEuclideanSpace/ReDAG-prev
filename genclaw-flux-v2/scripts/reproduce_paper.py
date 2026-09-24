from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.single_pass_policy import FirstImageReady, attach_single_pass, archive_generated_images


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_digest(case, folder):
    value = dict(case)
    if case.get("reference_image"):
        value["reference_sha256"] = digest(folder / case["reference_image"])
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def completed(case, folder):
    output = folder / case["output"]
    meta = folder / "records" / (case["id"] + ".json")
    if not output.is_file() or not meta.is_file():
        return False
    try:
        report = json.loads(meta.read_text())
        return (report.get("status") == "completed"
                and report.get("case_sha256") == case_digest(case, folder)
                and report.get("image_sha256") == digest(output))
    except (ValueError, OSError):
        return False


def run_case(case, folder, max_turns, seed):
    from PIL import Image
    from cc_genclaw.runtime.agent_runner import (
        prepare_agent_run, initialize_perception, run_prompt, persist_event,
    )
    from cc_genclaw.runtime.completion_guard import get_completion_state
    from cc_genclaw.runtime.perception_state import get_perception_state
    from cc_genclaw.tools.vlm_review import VLMReviewTool

    if case["source_status"] != "complete":
        raise ValueError("A complete source prompt is required")
    report_path = folder / "records" / (case["id"] + ".json")
    output = folder / case["output"]
    if output.exists():
        raise FileExistsError("Existing final image is never overwritten; move it before rerunning")
    os.environ["CC_GENCLAW_RUNS_DIR"] = str(folder / "runs" / case["id"])
    os.environ["QWEN_IMAGE_SEED"] = str(seed)
    os.environ["CC_GENCLAW_IMAGE_PROVIDER"] = "qwen_remote"
    os.environ.pop("QWEN_IMAGE_ALLOW_MOCK", None)
    os.environ["GENCLAW_REPRO_SINGLE_PASS"] = "1"
    report = {"id": case["id"], "title": case["title"], "prompt": case["prompt"],
              "case_sha256": case_digest(case, folder), "status": "running",
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": seed,
              "workflow": "full_genclaw_qwen", "tools": [], "reviews": [],
              "quality_retry_policy": "record_only", "draft_logic_executed": False}
    started = time.monotonic()
    write_json(report_path, report)
    try:
        run = prepare_agent_run(max_turns=max_turns)
        policy = attach_single_pass(run, report)
        session = Path(run.session.dir).resolve()
        report.update(model=run.model, run_dir=str(session))
        reference = str((folder / case["reference_image"]).resolve()) if case.get("reference_image") else None
        report["reference_image"] = reference
        initialize_perception(case["prompt"], user_image_path=reference, session=run.session)
        if get_perception_state()._fallback_used:
            raise RuntimeError("Perception failed; refusing to label the fallback as a complete run")

        def emit(event):
            persist_event(run.session, event)
            if event.type == "message_end" and getattr(event, "message", None) is not None:
                run.session.append_message(event.message)
            if event.type == "tool_execution_start":
                print(f"[{case['id']}] {event.tool_name}", flush=True)
            if event.type != "tool_execution_end":
                return
            result = getattr(event, "result", None)
            item = {"name": event.tool_name, "is_error": bool(getattr(result, "is_error", False))}
            payload = {}
            try:
                payload = json.loads(result.content)
                item["status"] = payload.get("status")
                if event.tool_name in {"t2i", "i2i"} and item["is_error"]:
                    error = str(payload.get("error", ""))
                    if any(token in error for token in ("Cannot connect to Qwen", "timed out", "Qwen diffusion HTTP 429", "Qwen diffusion HTTP 503")):
                        report.setdefault("infrastructure_errors", []).append(event.tool_name)
                if event.tool_name == "vlm_review" and payload.get("status") == "success":
                    report["reviews"].append(payload)
            except (ValueError, AttributeError, TypeError):
                pass
            report["tools"].append(item)
            policy.accept(event.tool_name, payload, item["is_error"])
            write_json(report_path, report)
            if report.get("infrastructure_errors"):
                raise RuntimeError("Qwen image infrastructure failed; stop before fallback generation")
            if event.tool_name == "i2i" and not item["is_error"] and payload.get("status") == "success":
                if not report["draft_logic_executed"]:
                    raise RuntimeError("A real GenClaw draft is required before image generation")
                archive_generated_images(report, folder, case)
                write_json(report_path, report)
                raise FirstImageReady()

        try:
            messages = run_prompt(case["prompt"], run, emit=emit)
        except FirstImageReady:
            messages = []
            run.session.append_event("single_pass_stop", {"reason": "User requested retaining first output without quality retries"})
        if any(getattr(m, "stop_reason", None) in {"error", "max_turns"} for m in messages):
            raise RuntimeError("Agent stopped with an error or reached the turn limit")
        if report.get("infrastructure_errors"):
            raise RuntimeError("Qwen image infrastructure failed during this run; a fallback image is not accepted")
        state = get_completion_state()
        if not state.image_produced or not state.latest_final_path:
            raise RuntimeError("Agent did not produce a final image")
        final = Path(state.latest_final_path).resolve()
        if not final.is_relative_to(session):
            raise RuntimeError("Final image must belong to this fresh session")
        reviews = [r for r in report["reviews"] if Path(r.get("image_path", "")).resolve() == final]
        if not reviews:
            args = {"image_path": str(final), "original_user_prompt": case["prompt"]}
            if reference:
                args["reference_image_paths"] = [reference]
            result = VLMReviewTool().execute(args)
            review = json.loads(result.content)
            run.session.append_event("batch_final_review", {"args": args, "result": review})
            if result.is_error or review.get("status") != "success":
                raise RuntimeError("Final VLM review failed; image remains in the session directory")
            report["reviews"].append(review)
            reviews = [review]
        report["final_review"] = reviews[-1]
        report["source_final_path"] = str(final)
        if not report["draft_logic_executed"]:
            raise RuntimeError("The GenClaw draft was not executed")
        qwen = []
        for path in session.rglob("*.json"):
            try:
                data = json.loads(path.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(data, dict) and data.get("backend") in {"mock", "qwen-image", "qwen-image-edit", "flux-fill", "flux-dev"}:
                if data["backend"] == "mock":
                    raise RuntimeError("Mock output is forbidden in paper reproduction")
                qwen.append({"metadata_path": str(path), **data})
        report["qwen_generations"] = qwen
        report["used_qwen"] = bool(qwen)
        if not qwen:
            raise RuntimeError("A real Qwen image output is required for this reproduction")
        report["quality_passed"] = bool(reviews[-1].get("passed"))
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(final) as img:
            img.load()
            tmp = output.with_suffix(".tmp.png")
            img.convert("RGB").save(tmp, format="PNG")
        tmp.replace(output)
        report.update(status="completed", output=str(output), image_sha256=digest(output))
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        if type(exc) in {RuntimeError, ValueError, FileExistsError}:
            report["error"] = str(exc)[:400]
        print(f"[{case['id']}] failed: {type(exc).__name__}", flush=True)
    finally:
        archive_generated_images(report, folder, case)
        report["seconds"] = round(time.monotonic() - started, 2)
        write_json(report_path, report)
    return 0 if report["status"] == "completed" else 1


def preflight(cases, folder):
    from cc_genclaw.config import load_genclaw_config
    load_genclaw_config()
    issues = []
    for case in cases:
        if case.get("reference_image") and not (folder / case["reference_image"]).is_file():
            issues.append(f"Missing reference image for {case['id']}")
    if any(c.get("needs_search") for c in cases) and not os.environ.get("TAVILY_API_KEY"):
        issues.append("TAVILY_API_KEY is missing")
    check = subprocess.run([sys.executable, str(ROOT / "scripts/check_setup.py")], cwd=ROOT)
    if check.returncode:
        issues.append("Model/browser/Qwen setup check failed")
    for item in issues:
        print(item, flush=True)
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/paper_cases.json")
    parser.add_argument("--ids", help="Comma-separated case IDs in execution order, e.g. 07,08,03")
    parser.add_argument("--check", action="store_true", help="Check configuration without generating")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--seed", type=int, default=260500)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".browsers"))
    os.environ.setdefault("CHAINLIT_NO_TELEMETRY", "true")
    manifest = args.manifest.resolve()
    folder = manifest.parent
    data = json.loads(manifest.read_text())
    all_cases = data["cases"]
    if args.worker_case:
        case = next(c for c in all_cases if c["id"] == args.worker_case)
        return run_case(case, folder, args.max_turns, args.seed + int(case["id"]))
    wanted = list(dict.fromkeys(x.strip().zfill(2) for x in args.ids.split(","))) if args.ids else None
    by_id = {c["id"]: c for c in all_cases}
    if wanted and set(wanted) - set(by_id):
        parser.error("Unknown case ID")
    ordered = [by_id[case_id] for case_id in wanted] if wanted else all_cases
    cases = [c for c in ordered if c["source_status"] == "complete"]
    if not cases:
        parser.error("Selected entries have no complete original prompt")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".batch.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another batch is already running")
        issues = preflight(cases, folder)
        state = {"workflow": "full_genclaw_qwen", "status": "blocked" if issues else "ready",
                 "issues": issues, "selected": [c["id"] for c in cases],
                 "excluded": [{"id": c["id"], "reason": c["source_status"]} for c in all_cases if c["source_status"] != "complete"]}
        status_file = folder / "reproduction_status.json"
        write_json(status_file, state)
        if issues or args.check:
            return 1 if issues else 0
        for case in cases:
            if completed(case, folder):
                print(f"[{case['id']}] existing verified result; skipping", flush=True)
                continue
            if (folder / case["output"]).exists():
                state.update(status="blocked", issues=[f"Existing unverified image for {case['id']}; refusing overwrite"])
                write_json(status_file, state)
                return 1
            state.update(status="running", current=case["id"])
            write_json(status_file, state)
            print(f"[{case['id']}] starting {case['title']}", flush=True)
            code = subprocess.call([sys.executable, "-u", str(Path(__file__).resolve()),
                "--manifest", str(manifest), "--worker-case", case["id"],
                "--max-turns", str(args.max_turns), "--seed", str(args.seed)], cwd=ROOT)
            if code:
                state.update(status="stopped_on_error", current=case["id"])
                write_json(status_file, state)
                return code
            print(f"[{case['id']}] saved {case['output']}", flush=True)
        state.update(status="completed", current=None)
        write_json(status_file, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
