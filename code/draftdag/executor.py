import json
import os
import time

from .views import load_image, to_rgb


class Executor:
    """Draws whole images from a prompt and edits a canvas with an instruction. Images are PIL RGB, same size as the canvas.
    last_sec: when not None after draw(), the loop books this many seconds instead of the measured wall time."""

    name = "base"
    calls = 0
    sec = 0.0
    last_sec = None
    pid = None

    def draw(self, prompt, n, seed, size=1024):
        raise NotImplementedError

    def edit(self, canvas, instruction, n, seed):
        raise NotImplementedError

    def commit_hook(self, canvas, cand, composite, info):
        return composite

    def stats(self):
        return dict(name=self.name, calls=self.calls, sec=round(self.sec, 1))


class PreDrawn(Executor):
    """Serves drafts saved by scripts/draw.py (runs/<run>/drafts.jsonl) in their stored order, keyed by prompt text, and
    books their recorded generation time; falls back to the wrapped executor when a prompt's pool is exhausted. Lets every
    mode of a comparison start from the same drafts (paired), and lets a draft pool be scored once the evaluator exists."""

    name = "predrawn"

    def __init__(self, inner, drafts_jsonl, size=None):
        self.inner = inner
        self.pool, self.sec_per, self.used = {}, {}, {}
        self.served = 0
        self.size = size
        with open(drafts_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                paths = [os.path.normpath(p if os.path.isabs(p) else os.path.join(os.path.dirname(os.path.abspath(drafts_jsonl)), p)) for p in rec["drafts"]]
                per = float(rec.get("sec", 0.0)) / max(1, int(rec.get("n", len(paths))))
                for key in ("pid:" + str(rec["pid"]), "prompt:" + rec["prompt"]):
                    self.pool.setdefault(key, []).extend(paths)
                    self.sec_per.setdefault(key, per)
                if size is not None and int(rec.get("size", size)) != int(size):
                    raise ValueError(f"draft pool was drawn at {rec.get('size')} but the loop runs at {size}")

    def _key(self, prompt):
        """The loop sets ex.pid before drawing; a pool entry for that pid wins over the prompt text (prompts may repeat)."""
        if self.pid is not None and ("pid:" + str(self.pid)) in self.pool:
            return "pid:" + str(self.pid)
        return "prompt:" + prompt

    def draw(self, prompt, n, seed, size=1024):
        key = self._key(prompt)
        paths = self.pool.get(key, [])
        k = self.used.get(key, 0)
        take = paths[k:k + n]
        self.used[key] = k + len(take)
        ims = [to_rgb(load_image(p)) for p in take]
        for im in ims:
            if max(im.size) != int(size):
                raise ValueError(f"pre-drawn draft {im.size} does not match size {size}")
        sec = self.sec_per.get(key, 0.0) * len(take)
        self.served += len(take)
        if len(take) < n:
            if self.inner is None:
                raise RuntimeError(f"draft pool exhausted for prompt {prompt[:60]!r} and no fallback executor")
            t0 = time.time()
            ims += self.inner.draw(prompt, n - len(take), seed, size)
            sec += time.time() - t0
        self.calls += len(ims)
        self.sec += sec
        self.last_sec = sec
        return ims

    def edit(self, canvas, instruction, n, seed):
        if self.inner is None:
            raise RuntimeError("no executor for edits")
        self.last_sec = None
        return self.inner.edit(canvas, instruction, n, seed)

    def commit_hook(self, canvas, cand, composite, info):
        return self.inner.commit_hook(canvas, cand, composite, info) if self.inner is not None else composite

    def load_lora(self, path):
        return self.inner.load_lora(path)

    def stats(self):
        d = dict(name=self.name, served=self.served, calls=self.calls, sec=round(self.sec, 1))
        if self.inner is not None:
            d["inner"] = self.inner.stats()
        return d
