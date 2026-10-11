"""Explicit workflow model downloads, stored on the large data disk."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'qwen-image-edit-2511-Q4_K_M.gguf': ('unsloth/Qwen-Image-Edit-2511-GGUF', 'qwen-image-edit-2511-Q4_K_M.gguf'),
    'qwen_2.5_vl_7b_fp8_scaled.safetensors': ('Comfy-Org/Qwen-Image_ComfyUI', 'split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors'),
    'qwen_image_vae.safetensors': ('Comfy-Org/Qwen-Image_ComfyUI', 'split_files/vae/qwen_image_vae.safetensors'),
}


def download(item):
    name, (repo, remote) = item
    destination = ROOT / 'models' / name
    url = f'https://huggingface.co/{repo}/resolve/main/{remote}'
    response = urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=60)
    size = int(response.headers['Content-Length'])
    if destination.exists() and destination.stat().st_size == size:
        return {'name': name, 'url': url, 'bytes': size, 'resumed': True}
    part = destination.with_suffix(destination.suffix + '.partial')
    block = 128 * 1024 * 1024
    with part.open('wb') as f:
        f.truncate(size)

    def chunk(start):
        end = min(size - 1, start + block - 1)
        for attempt in range(5):
            try:
                request = urllib.request.Request(url + f'?chunk={start}', headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f'Range response was {r.status}')
                    data = r.read()
                if len(data) != end-start+1:
                    raise RuntimeError('Incomplete range')
                with part.open('r+b') as f:
                    f.seek(start)
                    f.write(data)
                print(f'{name} {end+1}/{size}', flush=True)
                return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(attempt + 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(chunk, range(0, size, block)))
    part.rename(destination)
    digest = hashlib.sha256()
    with destination.open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(data)
    record = {'name': name, 'url': url, 'bytes': size, 'sha256': digest.hexdigest()}
    (ROOT / 'provenance' / (name + '.json')).write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for record in pool.map(download, FILES.items()):
            print(json.dumps(record), flush=True)
