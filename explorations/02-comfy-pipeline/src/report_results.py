"""Record real artifacts, node coverage, and sampler results after execution."""
import json
import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from workflow import ROOT, load_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--select', default='final_no_lora_8step.png')
    args = parser.parse_args()
    run = ROOT / 'runs/001-autumn-lounge'
    shutil.copyfile(run/args.select,run/'final_scene.png')
    selected_workflow = run/(Path(args.select).stem+'.workflow.json')
    shutil.copyfile(selected_workflow,run/'final_scene.workflow.json')
    workflow = load_workflow()
    expected = {('main',n['id']) for n in workflow['nodes'] if n['type'] != 'MarkdownNote'}
    subgraphs = {g['id']:g for g in workflow['definitions']['subgraphs']}
    for node in workflow['nodes']:
        if node['type'] in subgraphs:
            expected.update((f"main.{node['id']}",n['id']) for n in subgraphs[node['type']]['nodes'])
    records = [json.loads(line) for line in (run/'execution.jsonl').read_text().splitlines()]
    seen = {(n['scope'],n['id']) for n in records}
    artifacts = []
    images = []
    for name,label in [('composite_new_scene.png','Layer composite'),
                       ('final_no_lora_8step.png','Qwen Edit 2511 / no LoRA / 8 steps / CFG 1'),
                       ('final_no_lora_40step.png','Qwen Edit 2511 / no LoRA / 40 steps / CFG 4'),
                       ('final_qwen21.png','Qwen Image 2.1 / no LoRA / 30 steps / CFG 1')]:
        with Image.open(run/name) as image:
            artifacts.append(dict(file=name,size=list(image.size),mode=image.mode))
            thumbnail = image.convert('RGB')
            thumbnail.thumbnail((1200,552))
        images.append((label,thumbnail))
    sheet = Image.new('RGB',(1200,len(images)*590),'#f5f5f5')
    draw = ImageDraw.Draw(sheet)
    for i,(label,image) in enumerate(images):
        draw.text((12,i*590+10),label,fill='black')
        sheet.paste(image,(0,i*590+34))
    sheet.save(run/'comparison.jpg',quality=95)
    report = dict(computation_nodes=len(expected),executed_computation_nodes=len(expected&seen),
                  selected_result=args.select, final_result='final_scene.png',
                  selection_reason='Visual inspection: 8-step CFG1 preserves natural light and faces; 40-step CFG4 shows dark grain and texture artifacts in this scene',
                  missing_nodes=sorted(expected-seen),removed_lora_nodes=[110,53],
                  source_reproduction='ComfyUI Qwen-Image-Edit-2511 GGUF, exact conditioning and sampler nodes',
                  source_modifications=['RMBG INSPYRENET replaced with requested SAM2.1',
                                        'Missing browser temporary canvas replaced with saved-transform renderer',
                                        'Five generated inputs and one generated background replace absent original input files',
                                        'New scene seating transforms saved separately',
                                        '40-step comparison changes steps/end and CFG; 8-step result preserves sampler settings'],
                  reference_latent_shapes=[[1,16,1,87,189],[1,16,1,115,250]],
                  seed=603311763670135,artifacts=artifacts,
                  sampler_elapsed_seconds=[n['elapsed_seconds'] for n in records if n['type']=='KSamplerAdvanced'],
                  model_integrity={})
    for path in (ROOT/'provenance').glob('hf_*.json'):
        remote = json.loads(path.read_text())
        name = Path(remote['rfilename']).name
        local = json.loads((ROOT/'provenance'/f'{name}.json').read_text())
        report['model_integrity'][name] = dict(sha256=local['sha256'],matches_hf_lfs=local['sha256']==remote['lfs']['sha256'])
    (run/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
