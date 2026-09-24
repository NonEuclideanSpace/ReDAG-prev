import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from cc_genclaw.llm import chat_completion


def test_gpt5_request_parameters():
    calls=[]
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:calls.append(kw))))
    chat_completion(client,model='gpt-5.5',messages=[],max_tokens=1200,temperature=0)
    assert calls[0]['max_completion_tokens']==8192
    assert 'max_tokens' not in calls[0] and 'temperature' not in calls[0]
    chat_completion(client,model='gpt-4.1',messages=[],max_tokens=1200,temperature=0)
    assert calls[1]['max_tokens']==1200 and calls[1]['temperature']==0


def test_explicit_completion_limit_preserved():
    calls=[]
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:calls.append(kw))))
    chat_completion(client,model='gpt-5.5',max_tokens=1200,max_completion_tokens=16000)
    assert calls[0]['max_completion_tokens']==16000 and 'max_tokens' not in calls[0]


def test_resume_checks_image_and_prompt(tmp_path):
    from scripts.reproduce_paper import completed, case_digest, digest, write_json
    case={'id':'01','prompt':'a red cup','reference_image':None,'output':'images/01.png'}
    target=tmp_path/case['output'];target.parent.mkdir();target.write_bytes(b'fixture-not-a-real-result')
    meta={'status':'completed','case_sha256':case_digest(case,tmp_path),'image_sha256':digest(target)}
    write_json(tmp_path/'records/01.json',meta)
    assert completed(case,tmp_path)
    assert not completed({**case,'prompt':'a green cup'},tmp_path)
    target.write_bytes(b'changed')
    assert not completed(case,tmp_path)


def test_reference_change_invalidates_resume(tmp_path):
    from scripts.reproduce_paper import case_digest
    ref=tmp_path/'input.png';ref.write_bytes(b'reference-a')
    case={'prompt':'edit','reference_image':'input.png'}
    first=case_digest(case,tmp_path)
    ref.write_bytes(b'reference-b')
    assert first!=case_digest(case,tmp_path)


def test_qwen_outage_cannot_be_counted_as_completed(monkeypatch,tmp_path):
    from scripts.reproduce_paper import run_case
    import cc_genclaw.runtime.agent_runner as runner
    import cc_genclaw.runtime.perception_state as perception
    session_dir=tmp_path/'session';session_dir.mkdir()
    session=SimpleNamespace(dir=str(session_dir),append_event=lambda *a:None)
    from cc_genclaw.hooks import Hooks
    monkeypatch.setattr(runner,'prepare_agent_run',lambda **kw:SimpleNamespace(session=session,model='test',context=SimpleNamespace(system_prompt='',tools=[]),config=SimpleNamespace(hooks=Hooks(),get_followup_messages=None)))
    monkeypatch.setattr(runner,'initialize_perception',lambda *a,**kw:None)
    monkeypatch.setattr(perception,'get_perception_state',lambda:SimpleNamespace(_fallback_used=False))
    def fake_run(prompt,run,emit):
        emit(SimpleNamespace(type='tool_execution_end',tool_name='i2i',result=SimpleNamespace(is_error=True,content=json.dumps({'status':'failed','error':'Cannot connect to Qwen'}))))
        return []
    monkeypatch.setattr(runner,'run_prompt',fake_run)
    case={'id':'03','title':'fixture','prompt':'fixture','source_status':'complete','reference_image':None,'output':'images/03.png'}
    assert run_case(case,tmp_path,10,1)==1
    report=json.loads((tmp_path/'records/03.json').read_text())
    assert report['status']=='failed' and report['infrastructure_errors']==['i2i']
    assert not (tmp_path/'images/03.png').exists()

