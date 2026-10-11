import argparse
import json
from pathlib import Path

from image_nodes import SAM2BackgroundRemoval
from workflow import ROOT, Workflow, load_workflow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'runs/001-autumn-lounge')
    args = parser.parse_args()
    workflow = load_workflow()
    backend = SAM2BackgroundRemoval()
    executor = Workflow(workflow, args.run, rmbg=backend)
    output = executor.execute(184)
    output[1].save(args.run / 'composite_original_layout.png')
    layout = (ROOT / 'configs/layout_autumn_lounge.json').read_text()
    (args.run / 'layout_new_scene.json').write_text(layout)
    next(n for n in workflow['nodes'] if n['id'] == 184)['widgets_values'][0] = layout
    executor.cache.pop(('main',184))
    executor.execute(184)[1].save(args.run / 'composite_new_scene.png')
    (args.run / 'sam2_masks.json').write_text(json.dumps(backend.records, indent=2))
    assets = args.run / 'transparent_assets'
    assets.mkdir(exist_ok=True)
    for node_id, name in [(122, 'woman_red'), (132, 'man_blue'), (139, 'woman_green'), (147, 'man_ochre'), (155, 'woman_lilac')]:
        rgb, alpha = executor.cache[('main', node_id)]
        # Save straight RGB from the pre-blend RMBG output; ImageBlendAdvance's
        # RGB is already matted against a mask-image background.
        rgba = next(executor.cache[(f'main.{node_id}', n['id'])][0]
                    for g in workflow['definitions']['subgraphs'] if g['id'] == next(n['type'] for n in workflow['nodes'] if n['id'] == node_id)
                    for n in g['nodes'] if n['type'] == 'RMBG')
        rgba.save(assets / (name + '.png'))
    (args.run / 'workflow_no_lora.json').write_text(json.dumps(load_workflow(), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
