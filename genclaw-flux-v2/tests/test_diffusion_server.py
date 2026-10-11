import base64
import io
import json
import socket
import sys
import threading
import time
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import uvicorn
from fastapi.testclient import TestClient
from PIL import Image

from server.diffusion_server import DiffusionEngine, GenerationRequest, create_app, fill_canvas


def encoded_image(color="red", size=(320, 192)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def worker():
    engine = DiffusionEngine("mock", "/unused")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(create_app(engine), log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.02)
    assert server.started
    try:
        yield "http://127.0.0.1:" + str(sock.getsockname()[1]), engine
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


@pytest.fixture
def configured(monkeypatch, tmp_path, worker):
    from cc_genclaw.runtime.session_dir import reset_session_dir
    for key, value in {"CC_GENCLAW_IMAGE_PROVIDER": "qwen_remote", "QWEN_IMAGE_BASE_URL": worker[0],
                       "QWEN_IMAGE_MODEL_NAME": "qwen-image", "QWEN_IMAGE_API_KEY": "", "QWEN_IMAGE_WIDTH": "320",
                       "QWEN_IMAGE_HEIGHT": "320", "QWEN_IMAGE_STEPS": "2", "QWEN_IMAGE_SEED": "123",
                       "QWEN_IMAGE_ALLOW_MOCK": "1", "QWEN_IMAGE_TIMEOUT": "5"}.items():
        monkeypatch.setenv(key, value)
    reset_session_dir(str(tmp_path / "runs"))
    return worker


def test_actual_http_transport_and_artifacts(configured):
    from cc_genclaw.tools.t2i import T2ITool
    result = T2ITool().execute({"prompt": "one red cup"})
    assert not result.is_error
    path = Path(json.loads(result.content)["final_path"])
    assert path.is_absolute() and Image.open(path).size == (320, 320)
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["backend"] == "mock" and metadata["seed"] == 123


def test_reference_bytes_and_aspect_ratio(configured, tmp_path):
    from cc_genclaw.tools._image_gen import generate_image
    engine = configured[1]
    calls = []
    original = engine.generate

    def capture(req, images, seed):
        calls.append((req, images))
        return original(req, images, seed)

    engine.generate = capture
    reference = tmp_path / "reference.png"
    reference.write_bytes(base64.b64decode(encoded_image()))
    output = generate_image("render this composition", [str(reference)])
    assert calls[0][1][0].getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(output).size == (320, 192)
    assert json.loads(Path(output).with_suffix(".json").read_text())["reference_count"] == 1


def test_qwen_pipeline_receives_all_reference_images(monkeypatch):
    calls = []
    engine = DiffusionEngine("qwen-image", "/unused")
    engine.load = lambda: None
    engine.pipe = lambda **kwargs: calls.append(kwargs) or SimpleNamespace(images=[Image.new("RGB", (320, 192))])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: seed), inference_mode=nullcontext))
    images = [Image.new("RGB", (320, 192), "red"), Image.new("RGB", (320, 192), "blue")]
    output, mode = engine.generate(GenerationRequest(prompt="merge both", width=320, height=192), images, 1)
    assert mode == "img2img" and output.size == (320, 192)
    assert len(calls[0]["image"]) == 2
    assert [image.getpixel((0, 0)) for image in calls[0]["image"]] == [(255, 0, 0), (0, 0, 255)]


def test_text_layout_uses_same_remote_backend(configured, tmp_path):
    from cc_genclaw.tools._code_text_draft.backends import t2i_generate, i2i_refine
    first = tmp_path / "poster" / "background.png"
    second = tmp_path / "poster" / "refined.png"
    t2i_generate("paper texture", str(first))
    i2i_refine(str(first), "poster", "centered", str(second))
    assert first.is_file() and second.is_file()
    assert json.loads(second.with_suffix(".json").read_text())["backend"] == "mock"


def test_no_silent_mock_fallback(configured, monkeypatch):
    from cc_genclaw.tools._image_gen import generate_image
    monkeypatch.delenv("QWEN_IMAGE_ALLOW_MOCK")
    with pytest.raises(RuntimeError, match="mock mode"):
        generate_image("cup")


def test_worker_failure_is_not_retried(configured):
    from cc_genclaw.tools._image_gen import generate_image
    calls = []

    def fail(*args):
        calls.append(1)
        raise RuntimeError("CUDA out of memory")

    configured[1].generate = fail
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        generate_image("cup")
    assert len(calls) == 1


def test_agent_loop_can_call_remote_qwen(configured):
    from cc_genclaw.loop import LoopConfig, run_agent_loop
    from cc_genclaw.types import AgentContext, UserMessage, ToolResultMessage
    from cc_genclaw.tools.t2i import T2ITool
    responses = iter([
        {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "t2i",
             "arguments": json.dumps({"prompt": "a cup"})}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [{"message": {"role": "assistant", "content": "Completed."}, "finish_reason": "stop"}]},
    ])
    context = AgentContext(system_prompt="Test tool routing", messages=[], tools=[T2ITool()])
    messages = run_agent_loop([UserMessage(content="Draw a cup")], context,
                              LoopConfig(llm_driver=lambda **kwargs: next(responses), max_turns=2))
    results = [m for m in messages if isinstance(m, ToolResultMessage)]
    assert len(results) == 1 and not results[0].is_error
    assert Path(json.loads(results[0].content)["final_path"]).is_file()


def test_full_runner_loads_prompts_and_tools(configured, monkeypatch, tmp_path):
    from cc_genclaw.runtime.agent_runner import prepare_agent_run
    monkeypatch.setenv("OPENAI_API_KEY", "startup-test-only")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("CC_GENCLAW_RUNS_DIR", str(tmp_path / "runner"))
    run = prepare_agent_run()
    assert {"t2i", "i2i", "code_scene_draft", "code_text_draft", "vlm_review"} <= {t.name for t in run.context.tools}
    assert all(tool.description.strip() for tool in run.context.tools)
    assert len(run.context.system_prompt) > 100


def test_fill_mask_protects_reference_and_crops_only_output():
    reference = Image.new("RGB", (320, 192), "red")
    canvas, mask, crop = fill_canvas([reference], 320, 192)
    assert canvas.size == (640, 192)
    assert canvas.getpixel((20, 20)) == (255, 0, 0)
    assert mask.getpixel((20, 20)) == 0
    assert mask.getpixel((400, 20)) == 255
    assert canvas.crop(crop).size == (320, 192)
    _, mask, crop = fill_canvas([], 320, 192)
    assert mask.getextrema() == (255, 255) and crop == (0, 0, 320, 192)


def test_auth_and_bad_input_rejected_before_inference():
    client = TestClient(create_app(DiffusionEngine("mock", "/unused"), token="test-token"))
    assert client.get("/health").status_code == 401
    headers = {"Authorization": "Bearer test-token"}
    assert client.get("/health", headers=headers).status_code == 200
    for body in [{"prompt": " "}, {"prompt": "cup", "images": ["not-base64"]},
                 {"prompt": "cup", "width": 321}, {"prompt": "cup", "model": "unknown"}]:
        assert client.post("/v1/generate", json=body, headers=headers).status_code == 422


def test_dev_backend_rejects_extra_references_and_wrong_model():
    client = TestClient(create_app(DiffusionEngine("qwen-image-edit", "/missing")))
    assert client.post("/v1/generate", json={"prompt": "cup", "model": "qwen-image"}).status_code == 409
    response = client.post("/v1/generate", json={"prompt": "cup", "model": "qwen-image-edit",
                                                "images": [encoded_image(), encoded_image()]})
    assert response.status_code == 422


def test_health_does_not_load_models_and_reports_missing_shards(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir()
    (transformer / "weights.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    engine = DiffusionEngine("qwen-image", str(tmp_path))
    response = TestClient(create_app(engine)).get("/health").json()
    assert engine.pipe is None
    assert not response["model_files_ready"]
    assert "transformer/missing.safetensors" in response["missing_files"]


def test_busy_worker_does_not_start_second_generation():
    engine = DiffusionEngine("mock", "/unused")
    start, finish = threading.Event(), threading.Event()
    original = engine.generate

    def slow(*args):
        start.set()
        assert finish.wait(timeout=5)
        return original(*args)

    engine.generate = slow
    client = TestClient(create_app(engine))
    thread = threading.Thread(target=lambda: client.post("/v1/generate", json={"prompt": "cup"}))
    thread.start()
    try:
        assert start.wait(timeout=5)
        assert client.post("/v1/generate", json={"prompt": "second"}).status_code == 429
    finally:
        finish.set()
        thread.join(timeout=5)
