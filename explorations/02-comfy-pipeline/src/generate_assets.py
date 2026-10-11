"""Generate scene inputs using the explicitly requested local Qwen Image 2.1."""
import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL = '/home/hdd4/lpz/Qwen-Image-2.1'
PEOPLE = [
    ('woman_red', 'an adult East Asian woman with shoulder length black hair, wearing a burgundy knit sweater and cream trousers, seated elegantly, legs naturally crossed, hands on lap'),
    ('man_blue', 'an adult East Asian man with short black hair, wearing a navy blue cardigan and beige trousers, seated with a warm smile, hands on lap'),
    ('woman_green', 'an adult East Asian woman with long dark hair, wearing a sage green blouse and dark trousers, seated elegantly, legs naturally crossed, hands on lap'),
    ('man_ochre', 'an adult East Asian man with wavy dark hair, wearing an ochre sweater and charcoal trousers, seated casually with legs resting naturally, hands on thighs'),
    ('woman_lilac', 'an adult East Asian woman with a short bob haircut, wearing a lilac cardigan and navy trousers, sitting upright with knees together, hands on lap'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, default=ROOT / 'runs/001-autumn-lounge')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--edit', type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--prompt')
    args = ap.parse_args()
    target = args.run / 'inputs'
    target.mkdir(parents=True, exist_ok=True)
    pipe = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to('cuda')
    pipe.vae.enable_tiling()
    records = []

    def generate(name, prompt, w, h, seed, image=None, output=None):
        destination = output or target / (name + '.png')
        if destination.exists():
            print(f'RESUME: {destination}', flush=True)
            return
        started = time.time()
        print(f'GENERATING {name}: {w}x{h}, seed={seed}', flush=True)
        with torch.inference_mode():
            result = pipe(prompt=prompt, image=image, width=w, height=h,
                          output_resolution=max(w, h), num_inference_steps=args.steps,
                          true_cfg_scale=1.0, generator=torch.Generator('cuda').manual_seed(seed)).images[0]
        result.save(destination)
        record = dict(name=name, prompt=prompt, seed=seed, steps=args.steps,
                      requested_size=[w, h], actual_size=list(result.size), mode=result.mode,
                      elapsed_seconds=round(time.time()-started, 2), output=str(destination))
        records.append(record)
        with (args.run / 'generation.jsonl').open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if args.edit:
        from PIL import Image
        image = Image.open(args.edit).convert('RGB')
        generate('final_qwen21', args.prompt, 1536, 704, 202610118, image,
                 args.output or args.run / 'final_qwen21.png')
        return
    generate('background',
             'Photorealistic architectural interior photograph of a cozy autumn conservatory lounge, '
             'straight-on eye-level camera, wide horizontal composition. A long cream upholstered sofa '
             'with exactly four generous seats across the left and center, and one matching lounge chair '
             'on the right. Front of the sofa and chair clearly visible. Warm natural afternoon light '
             'from tall windows, maple leaves and trees outside, terracotta tile floor, a few indoor plants '
             'at the far edges. Spacious foreground. Empty room without people, no table blocking the seats, '
             'no text, no watermark. Editorial interior photography, realistic textures.',
             1536, 704, 202610110)
    for i, (name, description) in enumerate(PEOPLE):
        generate(name,
                 'Photorealistic full body studio portrait of exactly one person: ' + description + '. '
                 'Facing the camera, viewed straight on at eye level. Entire body from top of head to '
                 'shoes fully visible, feet visible. Sitting pose, an invisible seat, no furniture, '
                 'no chair, no props, isolated against a plain white studio background. '
                 'Warm daylight from the upper left. Natural anatomy, realistic hands and clothing. '
                 'Leave clear white margins around the entire silhouette. No text, no watermark.',
                 704, 1024, 202610111+i)


if __name__ == '__main__':
    main()
