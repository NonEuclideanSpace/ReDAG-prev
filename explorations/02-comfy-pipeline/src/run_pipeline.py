"""Run the entire original graph with SAM2 masks and both LoRAs removed."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from workflow import ROOT, Workflow, load_workflow, conditioning_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/001-autumn-lounge')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--cfg', type=float)
    parser.add_argument('--layout', type=Path)
    parser.add_argument('--output')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--compare-no-lora', action='store_true', help='Save 8-step CFG1 baseline, then 40-step CFG4 result')
    args = parser.parse_args()
    args.output = args.output or ('final_no_lora_40step.png' if args.compare_no_lora else 'final_original_8step.png')
    workflow = load_workflow()
    sampler = next(n for n in workflow['nodes'] if n['id'] == 23)
    if args.steps is not None:
        sampler['widgets_values'][3] = args.steps
        sampler['widgets_values'][8] = args.steps
    if args.cfg is not None:
        sampler['widgets_values'][4] = args.cfg
    if args.layout:
        node = next(n for n in workflow['nodes'] if n['id'] == 184)
        node['widgets_values'][0] = args.layout.read_text()
    executor = Workflow(workflow, args.run)
    # Reuse audited SAM results from compose_scene.py so SAM and the diffusion
    # model need not stay in VRAM together. Cache only the subgraph outputs.
    for node_id in (122,132,139,147,155):
        directory = args.run / 'nodes/main'
        rgb = Image.open(directory / f'{node_id}_0.png').copy()
        mask = Image.open(directory / f'{node_id}_1.png').copy()
        executor.cache[('main', node_id)] = (rgb, mask)
    from model_nodes import ComfyModelNodes
    executor.model_backend = ComfyModelNodes()
    with torch.inference_mode():
        if args.prepare_only:
            executor.execute(86)
            executor.execute(60)
            cached = {node_id: executor.cache[('main', node_id)] for node_id in (20,60,85,86)}
            torch.save(cached, args.run / 'conditioning_cache.pt')
            (args.run / 'conditioning_layout.json').write_text(next(n for n in workflow['nodes'] if n['id'] == 184)['widgets_values'][0])
            (args.run / 'conditioning_fingerprint.txt').write_text(conditioning_fingerprint(workflow,args.run))
            print('Prepared exact upstream conditioning and VAE latent', flush=True)
            return
        cache_path = args.run / 'conditioning_cache.pt'
        fingerprint_path = args.run / 'conditioning_fingerprint.txt'
        if cache_path.exists() and fingerprint_path.exists() and fingerprint_path.read_text() == conditioning_fingerprint(workflow,args.run):
            cached = torch.load(cache_path, weights_only=False, map_location='cpu')
            for node_id, result in cached.items():
                executor.cache[('main', node_id)] = result
        if args.compare_no_lora:
            sampler['widgets_values'][3] = 8
            sampler['widgets_values'][4] = 1
            sampler['widgets_values'][8] = 30
            executor.execute(25)[0].save(args.run / 'final_no_lora_8step.png')
            (args.run / 'final_no_lora_8step.workflow.json').write_text(json.dumps(workflow,ensure_ascii=False,indent=2))
            sampler['widgets_values'][3] = 40
            sampler['widgets_values'][4] = 4
            sampler['widgets_values'][8] = 40
            for node_id in (23,24,25):
                executor.cache.pop(('main', node_id))
        result = executor.execute(25)
    result[0].save(args.run / args.output)
    saved = args.run / (Path(args.output).stem + '.workflow.json')
    saved.write_text(json.dumps(workflow, ensure_ascii=False, indent=2))
    print(f'FINAL: {args.run / args.output}', flush=True)


if __name__ == '__main__':
    main()
