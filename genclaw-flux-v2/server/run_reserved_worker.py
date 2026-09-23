from pathlib import Path
import os
import runpy
import torch

fraction = float(os.environ.get('FLUX_GPU_MEMORY_FRACTION', '0.55'))
if not 0 < fraction <= 1:
    raise ValueError('FLUX_GPU_MEMORY_FRACTION must be in (0, 1]')
torch.cuda.set_per_process_memory_fraction(fraction)
keep_cache = os.environ.get('FLUX_KEEP_CACHE') == '1'
original_empty_cache = torch.cuda.empty_cache
if keep_cache:
    torch.cuda.empty_cache = lambda: None
try:
    runpy.run_path(str(Path(__file__).with_name('flux_server.py')), run_name='__main__')
finally:
    torch.cuda.empty_cache = original_empty_cache
    original_empty_cache()

