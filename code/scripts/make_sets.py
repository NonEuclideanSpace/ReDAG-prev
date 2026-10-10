"""Prompt sets.

corebench: import T2I-CoReBench composition subsets (C-MI, C-MA, C-MR) as sets; the official checklists are stored
separately under data/checklists/ and are used for final scoring only, never inside the loop.
synthetic: write new scene descriptions at four complexity tiers with the planner API, together with an independent
requirement list (used as the coverage reference for the planner gate). Prompts are split dev/test by hash.
longbench: import LongBench-T2I (500 long prompts, median about 530 words) as set lb_t2i; its per-prompt labels (nine
evaluation dimensions) go to data/checklists/lb_t2i.json and, like the CoReBench checklists, are for final scoring only.
legacy: import a first-phase prompt file (id, prompt, optional cat/setting/style) as a set, e.g. the 60 xl_dev prompts whose
drafts were labelled on 10-08; same hash split, nothing else carried over."""
import argparse
import hashlib
import json
import os
import random

from common import P0

from draftdag.planner import DEFAULT_MODEL, api_chat, parse_json

TIERS = {"t02": (2, 1, 1, 1), "t03": (3, 2, 1, 1), "t05": (5, 3, 2, 1), "t10": (6, 3, 2, 1), "t20": (10, 6, 4, 2), "t30": (14, 9, 6, 2), "t40": (18, 12, 8, 3)}
SETTINGS = ["a busy restaurant kitchen", "a village market at dawn", "a hospital corridor", "a mountain campsite", "a subway platform", "a school science fair", "a shipyard dock", "a theatre backstage", "a farm during harvest", "a rooftop party at night", "a fire station", "a library reading room", "a beach volleyball tournament", "an airport gate", "a pottery workshop", "a snowy town square", "a construction site", "a wedding reception", "a botanical greenhouse", "a train dining car"]
STYLES = ["photorealistic photo", "oil painting", "watercolor illustration", "flat vector illustration", "anime style", "cinematic film still", "pencil sketch with light color wash"]

WRITE = """Write one scene description for an image in the style "{style}", set in {setting}. It must contain exactly {ne} distinct people, animals or objects that a viewer can tell apart (give each a short identifying look), {na} stated properties or poses of those entities, {nr} interactions between pairs of them (handing, pointing, looking at, sitting on, next to), and {nab} things that must NOT appear (say "there is no ..."). Mention left/right/center positions for at least {npos} entities. Give exactly these numbers of entities, properties, interactions and exclusions and nothing more: no extra objects, poses, expressions or actions beyond them. Refer to people and things only by their look (a man in a red scarf, a brown dog), never by names, letters or labels. Write it as natural flowing prose of {words} words or fewer, no lists, and no commentary about composition, mood, lighting or technique beyond naming the style once.

Then list every visual requirement you wrote, one fact per line as a short phrase: each entity, each property or pose, each interaction, each exclusion, each stated position. No headings, no grouping, no lines about format. Output one JSON object and nothing else: {{"prompt": "<the description>", "requirements": ["...", "..."]}}"""


def split_of(pid):
    return "test" if int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) % 4 == 0 else "dev"


def write_set(name, rows):
    os.makedirs(os.path.join(P0, "data", "sets"), exist_ok=True)
    p = os.path.join(P0, "data", "sets", name + ".jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), "->", p)


def corebench(args):
    src = os.path.join(P0, "data_ext", "corebench")
    os.makedirs(os.path.join(P0, "data", "checklists"), exist_ok=True)
    for sub, name in (("C-MI", "cb_mi"), ("C-MA", "cb_ma"), ("C-MR", "cb_mr")):
        d = json.load(open(os.path.join(src, sub + ".json"), encoding="utf-8"))
        rows, checks = [], {}
        for k, v in d.items():
            pid = k.replace("-", "_").lower()
            rows.append(dict(id=pid, cat=name, prompt=v["Prompt"], split=split_of(pid), n_check=len(v.get("Checklist", []))))
            checks[pid] = v.get("Checklist", [])
        write_set(name, rows)
        with open(os.path.join(P0, "data", "checklists", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(checks, f, ensure_ascii=False)


def synthetic(args):
    rng = random.Random(args.seed)
    for tier, (ne, na, nr, nab) in TIERS.items():
        if args.tier and tier != args.tier:
            continue
        rows = []
        reqs = {}
        if args.append:  # keep every existing prompt as it is and add args.n new ones after it (new ids, same hash split)
            rows = [json.loads(l) for l in open(os.path.join(P0, "data", "sets", tier + ".jsonl"), encoding="utf-8") if l.strip()]
            reqs = json.load(open(os.path.join(P0, "data", "requirements", tier + ".json"), encoding="utf-8"))
        target = len(rows) + args.n
        i = len(rows) if args.append else 0
        fails = 0
        while len(rows) < target and fails < 3 * args.n:
            setting = rng.choice(SETTINGS)
            style = rng.choice(STYLES)
            words = 60 + 25 * ne
            out = api_chat([{"role": "user", "content": WRITE.format(style=style, setting=setting, ne=ne, na=na, nr=nr, nab=nab, words=words, npos=min(3, ne))}], model=args.model, tag=f"write:{tier}:{i}", max_tokens=2500, temperature=0.9)
            i += 1
            try:
                d = parse_json(out)
                prompt = str(d["prompt"]).strip()
                req = [str(x) for x in d.get("requirements", [])]
                assert len(prompt.split()) >= 25 and len(req) >= ne
            except Exception:
                fails += 1
                continue
            pid = f"{tier}_{len(rows):04d}"
            rows.append(dict(id=pid, cat=tier, setting=setting, style=style, prompt=prompt, split=split_of(pid), n_req=len(req)))
            reqs[pid] = req
            print(pid, len(prompt.split()), "words", len(req), "reqs", flush=True)
            save_tier(tier, rows, reqs)  # after every prompt: a run stopped half way keeps what it wrote
        save_tier(tier, rows, reqs)


def save_tier(tier, rows, reqs):
    """Write the set and its requirement lists through temporary files, so a reader never sees half a file."""
    for sub, name, payload in (("sets", tier + ".jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)),
                               ("requirements", tier + ".json", json.dumps(reqs, ensure_ascii=False, indent=1))):
        d = os.path.join(P0, "data", sub)
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "." + name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, os.path.join(d, name))


def longbench(args):
    src = args.src or os.path.join(P0, "data_ext", "longbench_t2i", "instruction.jsonl")
    rows, checks = [], {}
    with open(src, encoding="utf-8") as f:
        for i, line in enumerate(l for l in f if l.strip()):
            r = json.loads(line)
            pid = f"lb_{i:04d}"
            rows.append(dict(id=pid, cat="lb_t2i", prompt=r["instruction"].strip(), split=split_of(pid), n_words=len(r["instruction"].split())))
            checks[pid] = r.get("label", [])
    write_set("lb_t2i", rows)
    os.makedirs(os.path.join(P0, "data", "checklists"), exist_ok=True)
    with open(os.path.join(P0, "data", "checklists", "lb_t2i.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False)


def legacy(args):
    rows = []
    with open(args.src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(dict(id=r["id"], cat=args.name, prompt=r["prompt"], split=split_of(r["id"]), **{k: r[k] for k in ("setting", "style") if k in r}))
    write_set(args.name, rows)


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
sub.add_parser("corebench")
s2 = sub.add_parser("synthetic")
s2.add_argument("--n", type=int, default=120)
s2.add_argument("--tier", default=None)
s2.add_argument("--model", default=DEFAULT_MODEL)
s2.add_argument("--seed", type=int, default=0, help="use a new seed with --append (the first 100-150 of each tier used seed 0)")
s2.add_argument("--append", action="store_true", help="add --n prompts after the existing ones of the tier instead of writing the set anew")
s4 = sub.add_parser("longbench")
s4.add_argument("--src", default=None, help="default data_ext/longbench_t2i/instruction.jsonl")
s3 = sub.add_parser("legacy")
s3.add_argument("--src", required=True, help="first-phase jsonl with id and prompt")
s3.add_argument("--name", required=True)
args = ap.parse_args()
{"corebench": corebench, "synthetic": synthetic, "legacy": legacy, "longbench": longbench}[args.cmd](args)
