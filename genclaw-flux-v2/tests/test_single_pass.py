import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from cc_genclaw.hooks import Hooks, BeforeToolCallContext
from scripts.single_pass_policy import SinglePassPolicy


def test_image_requires_the_actual_generated_draft(tmp_path):
    report = {}
    policy = SinglePassPolicy(report)
    draft = tmp_path / 'draft.png'
    Image.new('RGB', (32, 32)).save(draft)
    image_call = BeforeToolCallContext('i2i', {'image_path': str(draft)}, 'a')
    assert policy.before(image_call).block
    policy.accept('code_scene_draft', {'status': 'success', 'draft_png_path': str(draft)}, False)
    assert policy.before(image_call) is None
    assert report['draft_logic_executed']
    assert policy.before(BeforeToolCallContext('t2i', {}, 'b')).block
    assert policy.before(BeforeToolCallContext('i2i', {'image_path': str(tmp_path/'unrelated.png')}, 'c')).block
    assert policy.before(BeforeToolCallContext('code_scene_draft', {}, 'd')).block


def test_failed_visual_review_is_saved_without_retry(monkeypatch, tmp_path):
    import cc_genclaw.runtime.agent_runner as runner
    import cc_genclaw.runtime.perception_state as perception
    import cc_genclaw.runtime.completion_guard as completion
    from cc_genclaw.tools.vlm_review import VLMReviewTool
    from scripts.reproduce_paper import run_case

    session_dir = tmp_path/'session'
    session_dir.mkdir()
    draft = session_dir/'draft.png'
    final = session_dir/'gen_test.png'
    for path in (draft, final):
        Image.new('RGB', (32, 32), 'blue').save(path)
    final.with_suffix('.json').write_text(json.dumps({'backend': 'qwen-image', 'seed': 1}))
    session = SimpleNamespace(dir=str(session_dir), append_event=lambda *a: None, append_message=lambda *a: None)
    fake = SimpleNamespace(session=session, model='test', context=SimpleNamespace(system_prompt='', tools=[]),
                           config=SimpleNamespace(hooks=Hooks(), get_followup_messages=None))
    monkeypatch.setattr(runner, 'prepare_agent_run', lambda **kw: fake)
    monkeypatch.setattr(runner, 'initialize_perception', lambda *a, **kw: None)
    monkeypatch.setattr(perception, 'get_perception_state', lambda: SimpleNamespace(_fallback_used=False))
    monkeypatch.setattr(completion, 'get_completion_state', lambda: SimpleNamespace(image_produced=True, latest_final_path=str(final)))
    reached_retry = []

    def fake_run(prompt, run, emit):
        def tool(name, data):
            emit(SimpleNamespace(type='tool_execution_end', tool_name=name,
                                 result=SimpleNamespace(is_error=False, content=json.dumps(data))))
        tool('code_scene_draft', {'status': 'success', 'draft_png_path': str(draft)})
        tool('i2i', {'status': 'success', 'final_path': str(final)})
        reached_retry.append(True)
        raise AssertionError('The planner must stop before requesting a quality retry')

    monkeypatch.setattr(runner, 'run_prompt', fake_run)
    monkeypatch.setattr(VLMReviewTool, 'execute', lambda *a: SimpleNamespace(is_error=False, content=json.dumps(
        {'status': 'success', 'passed': False, 'verdict': 'NEEDS_FIX', 'image_path': str(final), 'issues': [{'description': 'wrong count'}]})))
    case = {'id': '07', 'title': 'fixture', 'prompt': 'four blue apples', 'source_status': 'complete',
            'reference_image': None, 'output': 'images/07.png'}
    assert run_case(case, tmp_path, 10, 1) == 0
    report = json.loads((tmp_path/'records/07.json').read_text())
    assert report['status'] == 'completed' and report['quality_passed'] is False
    assert report['draft_logic_executed'] and report['used_qwen']
    assert not reached_retry
    assert (tmp_path/'images/07.png').is_file()
    assert len(report['saved_attempts']) == 1
    assert report['saved_attempts'][0]['review_status'] == 'failed'
    assert Path(report['saved_attempts'][0]['path']).is_file()

