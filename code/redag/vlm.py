import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw

from .paths import gates_dir
from .views import data_url, image_key, resize_min_side, to_rgb

TEMPLATES = {
    "a": "{q} Answer yes or no.",
    "b": "{q} Please answer with yes or no.",
    # three-way answer: "partly" takes the mass that a forced yes/no pushes into yes when the thing is there but a detail is off
    "c": "{q} Answer with exactly one word: yes, partly, or no.",
}
PARTLY = ("partly", "partly.", "partially", "partial")
GRADED = "{q} Reply with a single integer from 0 to 100: the probability that the correct answer is yes. Output only the number."
YES = ("yes", "yes.", "y")
NO = ("no", "no.", "n")
REASONING_RE = re.compile(r"^(gpt-5|gpt-6|o[1-9])", re.I)
GROUND = 'Locate {desc} in the image. Output only one bounding box for the best match as JSON: {{"bbox_2d": [x1, y1, x2, y2]}}.'
BBOX_RE = re.compile(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]')
# Count conditions (1009 count probe on the 166 count items of seed_v4): the inverted yes/no "Are there N+1 or more X?" says
# yes on 56% of the items and fails 5 of the 14 human-y items; asking for the number fails 1 of 14 and agrees with the human
# on 83% (yes/no: 74%); listing one box per instance agrees on 83% too and is the cheapest second opinion.
COUNT_Q = "How many {plural} are in this image? Answer with a single number."
DETECT_Q = 'Locate every {cls} in the image. Output a JSON list with one {{"bbox_2d": [x1, y1, x2, y2]}} per {cls} you can see; output [] if there is none.'
NUMW = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def parse_count(txt):
    m = re.search(r"\d+", txt or "")
    if m:
        return int(m.group(0))
    for w, v in NUMW.items():
        if re.search(r"\b" + w + r"\b", (txt or "").lower()):
            return v
    return None


def digit_dist(lp):
    """P over the first digit of the answer from the first-token top logprobs (Qwen tokenizes digits one by one)."""
    d = {}
    for e in lp or []:
        t = _norm_tok(e["token"])
        if len(t) == 1 and t.isdigit():
            d[int(t)] = d.get(int(t), 0.0) + math.exp(float(e["logprob"]))
        elif t in NUMW:
            d[NUMW[t]] = d.get(NUMW[t], 0.0) + math.exp(float(e["logprob"]))
    z = sum(d.values())
    return {k: v / z for k, v in d.items()} if z > 0 else {}


def _norm_tok(t):
    return t.strip().strip('"').strip("'").lower()


class VLMError(RuntimeError):
    pass


class VLM:
    """OpenAI-compatible chat client for a vision-language model.

    mode="logprob": P(yes) from the first-token top logprobs (vLLM-served open models).
    mode="graded": P(yes) from an integer 0-100 the model writes (API models without logprobs).
    """

    def __init__(self, base_url, model, api_key=None, mode="logprob", workers=8, timeout=300, coord_mode=None, disable_thinking=True, cache_path=None, require_gate=True, max_side=1024):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("VLM_API_KEY") or "EMPTY"
        self.mode = mode
        self.workers = int(workers)
        self.timeout = timeout
        self.coord_mode = coord_mode
        self.disable_thinking = disable_thinking
        self.require_gate = require_gate
        self.max_side = int(max_side)
        self.fmt_fail = 0
        self.calls = 0
        self.sec = 0.0
        self.lock = threading.Lock()
        self.cache = {}
        self.cache_path = cache_path
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["k"]] = r["p"]
                    except Exception:
                        pass
        self.reasoning = bool(REASONING_RE.match(model.split("/")[-1]))
        self.session = requests.Session()
        self.session.trust_env = os.environ.get("P0_TRUST_ENV", "1") != "0"

    def _limits(self, n):
        """OpenAI reasoning models (gpt-5*, gpt-6*, o*) reject max_tokens and temperature and spend tokens on reasoning first."""
        if self.reasoning:
            return dict(max_completion_tokens=max(int(n), 2000), reasoning_effort=os.environ.get("VLM_REASONING_EFFORT", "low"))
        return dict(max_tokens=int(n), temperature=0.0)

    def tag(self):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model)

    def gate_path(self):
        return os.path.join(gates_dir(), f"evaluator_{self.tag()}.json")

    def check_gate(self):
        if not self.require_gate:
            return True
        p = self.gate_path()
        if not os.path.exists(p):
            raise VLMError(f"evaluator gate missing: {p}; run scripts/calibrate_eval.py first")
        g = json.load(open(p, encoding="utf-8"))
        if not g.get("pass"):
            raise VLMError(f"evaluator gate failed for {self.model}: {g.get('summary')}")
        if g.get("coord_mode"):
            self.coord_mode = g["coord_mode"]
        return True

    def _post(self, body):
        t0 = time.time()
        r = self.session.post(self.base_url + "/chat/completions", headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}, json=body, timeout=self.timeout)
        with self.lock:
            self.calls += 1
            self.sec += time.time() - t0
        if r.status_code != 200:
            raise VLMError(f"vlm http {r.status_code}: {r.text[:300]}")
        return r.json()

    def _msg(self, images, text):
        content = [{"type": "image_url", "image_url": {"url": data_url(resize_min_side(to_rgb(im), self.max_side) if min(im.size) > self.max_side else im)}} for im in images]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _extra(self):
        if self.disable_thinking and self.mode == "logprob":
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {}

    def _yes_prob_logprob(self, images, text, three_way=False):
        body = dict(model=self.model, messages=self._msg(images, text), max_tokens=2, temperature=0.0, logprobs=True, top_logprobs=20, **self._extra())
        d = self._post(body)
        try:
            lp = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise VLMError("no logprobs in response; the server must be started with logprobs enabled")
        ly, ln, lpart = None, None, None
        for e in lp:
            t = _norm_tok(e["token"])
            if t in YES and ly is None:
                ly = float(e["logprob"])
            elif t in NO and ln is None:
                ln = float(e["logprob"])
            elif three_way and (t in PARTLY or t == "part") and lpart is None:
                lpart = float(e["logprob"])
        if ly is None and ln is None and lpart is None:
            with self.lock:
                self.fmt_fail += 1
            return 0.5, 0.0, False
        ly = -30.0 if ly is None else ly
        ln = -30.0 if ln is None else ln
        if three_way and lpart is not None:  # "partly" counts against the condition
            ln = math.log(math.exp(ln) + math.exp(lpart))
        z = ly - ln
        return 1.0 / (1.0 + math.exp(-z)), z, True

    def _yes_prob_graded(self, images, text):
        body = dict(model=self.model, messages=self._msg(images, text), **self._limits(8))
        d = self._post(body)
        txt = d["choices"][0]["message"]["content"] or ""
        m = re.search(r"\d+(\.\d+)?", txt)
        if not m:
            with self.lock:
                self.fmt_fail += 1
            return 0.5, 0.0, False
        p = min(100.0, max(0.0, float(m.group(0)))) / 100.0
        p = min(0.999, max(0.001, p))
        return p, math.log(p / (1 - p)), True

    def yesno_one(self, image, question, tpl="a", inv=False):
        text = TEMPLATES[tpl].format(q=question) if self.mode == "logprob" else GRADED.format(q=question)
        key = f"{self.model}|{image_key(image)}|{image.size[0]}x{image.size[1]}|{tpl}|{question}"
        with self.lock:
            hit = self.cache.get(key)
        if hit is not None:
            p, z = hit
        else:
            p, z, ok = self._yes_prob_logprob([image], text, three_way=(tpl == "c")) if self.mode == "logprob" else self._yes_prob_graded([image], text)
            with self.lock:
                self.cache[key] = (p, z)
                if self.cache_path:
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(k=key, p=[p, z])) + "\n")
        if inv:
            return 1.0 - p, -z
        return p, z

    def yesno(self, items):
        """items: list of dict(image=PIL, question=str, tpl='a'|'b', inv=bool). Returns list of (p, z)."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda it: self.yesno_one(it["image"], it["question"], it.get("tpl", "a"), bool(it.get("inv", False))), items))

    def _count_answer(self, image, text):
        """Greedy integer answer plus the first-digit distribution (logprob mode) for a counting question."""
        body = dict(model=self.model, messages=self._msg([image], text), **self._limits(8), **self._extra())
        if self.mode == "logprob":
            body.update(logprobs=True, top_logprobs=20)
        d = self._post(body)
        ch = d["choices"][0]
        txt = ch.get("message", {}).get("content") or ""
        lp = None
        try:
            lp = ch["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            lp = None
        return parse_count(txt), digit_dist(lp)

    def count_one(self, image, plural, n):
        """P(at most n of the class in the image). The greedy integer settles answers of ten or more; below that the
        first-digit distribution gives the probability. Returns (p, z, k_hat)."""
        n = int(n)
        key = f"{self.model}|{image_key(image)}|{image.size[0]}x{image.size[1]}|count{n}|{plural}"
        with self.lock:
            hit = self.cache.get(key)
        if hit is not None:
            return hit[0], hit[1], hit[2] if len(hit) > 2 else None
        k, dist = self._count_answer(image, COUNT_Q.format(plural=plural))
        if k is None and not dist:
            with self.lock:
                self.fmt_fail += 1
            return 0.5, 0.0, None
        if k is not None and k >= 10:
            p = 0.999 if k <= n else 0.001
        elif dist:
            p = sum(v for dgt, v in dist.items() if dgt <= n)
        else:
            p = 0.95 if k <= n else 0.05
        p = min(0.999, max(0.001, p))
        z = math.log(p / (1.0 - p))
        with self.lock:
            self.cache[key] = (p, z, k)
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(k=key, p=[p, z, k])) + "\n")
        return p, z, k

    def count_many(self, items):
        """items: list of dict(image=PIL, plural=str, n=int). Returns list of (p, z, k_hat)."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda it: self.count_one(it["image"], it["plural"], it["n"]), items))

    def detect_count_one(self, image, cls, n):
        """Second opinion on a count: one box per instance, pass when at most n are listed. Returns (p, z, k)."""
        key = f"{self.model}|{image_key(image)}|{image.size[0]}x{image.size[1]}|detect{int(n)}|{cls}"
        with self.lock:
            hit = self.cache.get(key)
        if hit is not None:
            return hit[0], hit[1], hit[2] if len(hit) > 2 else None
        im = to_rgb(image)
        sent = resize_min_side(im, self.max_side) if min(im.size) > self.max_side else im
        txt = self.generate_one([sent], DETECT_Q.format(cls=cls), max_tokens=400)
        k = len(BBOX_RE.findall(txt))
        if k == 0 and not re.search(r"\[\s*\]", txt) and "bbox" not in txt:
            with self.lock:
                self.fmt_fail += 1
        p = 0.95 if k <= int(n) else 0.05
        z = math.log(p / (1.0 - p))
        with self.lock:
            self.cache[key] = (p, z, k)
            if self.cache_path:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(k=key, p=[p, z, k])) + "\n")
        return p, z, k

    def detect_count_many(self, items):
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda it: self.detect_count_one(it["image"], it["cls"], it["n"]), items))

    def generate_one(self, images, text, max_tokens=300, json_mode=False):
        body = dict(model=self.model, messages=self._msg(images, text), **self._limits(max_tokens), **self._extra())
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        d = self._post(body)
        return d["choices"][0]["message"]["content"] or ""

    def generate(self, items, max_tokens=300, json_mode=False):
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda it: self.generate_one(it["images"], it["text"], max_tokens, json_mode), items))

    def parse_bbox(self, text, sent_size, orig_size, min_side=12):
        m = BBOX_RE.search(text)
        if not m:
            return None
        x0, y0, x1, y1 = [float(g) for g in m.groups()]
        sw, sh = sent_size
        ow, oh = orig_size
        if self.coord_mode == "norm1000":
            x0, x1, y0, y1 = x0 * ow / 1000.0, x1 * ow / 1000.0, y0 * oh / 1000.0, y1 * oh / 1000.0
        else:
            x0, x1, y0, y1 = x0 * ow / sw, x1 * ow / sw, y0 * oh / sh, y1 * oh / sh
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(ow), x1), min(float(oh), y1)
        if x1 - x0 < min_side or y1 - y0 < min_side:
            return None
        return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]

    def ground(self, image, desc, max_area=0.7):
        if self.coord_mode is None:
            raise VLMError("coord_mode unknown; run calibrate_coords() first")
        im = to_rgb(image)
        sent = resize_min_side(im, self.max_side) if min(im.size) > self.max_side else im
        txt = self.generate_one([sent], GROUND.format(desc=desc.strip().rstrip(".")), max_tokens=60)
        b = self.parse_bbox(txt, sent.size, im.size)
        if b is not None and (b[2] - b[0]) * (b[3] - b[1]) > max_area * im.size[0] * im.size[1]:
            return None
        return b

    def ground_many(self, pairs):
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda pr: self.ground(pr[0], pr[1]), pairs))

    def calibrate_coords(self):
        W, H = 1024, 768
        im = Image.new("RGB", (W, H), (120, 130, 140))
        d = ImageDraw.Draw(im)
        truth = [600, 200, 800, 400]
        d.rectangle(truth, fill=(230, 20, 20))
        txt = self.generate_one([im], GROUND.format(desc="the red square"), max_tokens=60)
        m = BBOX_RE.search(txt)
        if not m:
            raise VLMError("grounding format not parsable: " + txt[:200])
        vals = [float(g) for g in m.groups()]
        from .schema import box_iou

        as_abs = vals
        as_norm = [vals[0] * W / 1000.0, vals[1] * H / 1000.0, vals[2] * W / 1000.0, vals[3] * H / 1000.0]
        ia, inn = box_iou(as_abs, truth), box_iou(as_norm, truth)
        self.coord_mode = "abs" if ia >= inn else "norm1000"
        return dict(coord_mode=self.coord_mode, iou_abs=ia, iou_norm=inn, raw=txt[:200])

    def stats(self):
        return dict(calls=self.calls, sec=round(self.sec, 1), fmt_fail=self.fmt_fail, model=self.model)
