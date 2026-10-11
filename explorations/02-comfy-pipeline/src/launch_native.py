"""Wait for the explicit download, then run both no-LoRA sampler settings."""
import os
import subprocess
import time

from workflow import ROOT
from download_models import FILES

if __name__ == '__main__':
    print('Waiting for verified model downloads', flush=True)
    while not all((ROOT / 'models' / name).exists() for name in FILES):
        time.sleep(10)
    print('All model files ready. Starting full native workflow.', flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1')
    subprocess.run([
        str(ROOT / '.venv/bin/python'), str(ROOT / 'src/run_pipeline.py'),
        '--compare-no-lora', '--layout', str(ROOT / 'runs/001-autumn-lounge/layout_new_scene.json'),
        '--output', 'final_no_lora_40step.png',
    ], env=env, check=True)
