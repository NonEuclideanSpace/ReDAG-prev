"""Label sets for the evaluator, the diagnosis and the acceptance rules.

items.jsonl: one item per line. kinds:
  cond: {id, kind:'cond', image, cid, ntype, question, region, pid, run}
  edit: {id, kind:'edit', before, after, op, node, ntype, instruction, region, rules, pid, run}
  diag: {id, kind:'diag', image, node, ntype, question, desc, pid, run}
labels_<who>.jsonl: {id, label, note}
  cond labels: y (satisfied), u (present but detail wrong), n (not satisfied)
  edit labels: {fixed: y/n, damaged: y/n, duplicate: y/n}
  diag labels: missing | wrong | extra | merged | fine
"""
import html
import json
import os

from .schema import box_intersects


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def items_from_run(run_dir, run_name, kinds=("cond", "edit", "diag"), max_per_pid=None):
    items = []
    img_root = os.path.join(run_dir, "img")
    if not os.path.isdir(img_root):
        return items
    for pid in sorted(os.listdir(img_root)):
        tp = os.path.join(img_root, pid, "trace.json")
        if not os.path.exists(tp):
            continue
        tr = json.load(open(tp, encoding="utf-8"))
        s1 = tr.get("stage1", {})
        if "cond" in kinds:
            for di, col in enumerate(s1.get("columns", [])):
                for cid, st in col.get("status", {}).items():
                    if st in ("pass", "fail"):
                        items.append(dict(id=f"{run_name}:{pid}:d{di}:{cid}", kind="cond", image=col["image"], cid=cid, ntype=col.get("ntype", {}).get(cid), question=None, region=None, pid=pid, run=run_name, score=col.get("score", {}).get(cid), status=st))
        for rd in tr.get("rounds", []):
            for br in rd.get("branches", []):
                if "diag" in kinds:
                    items.append(dict(id=f"{run_name}:{pid}:r{rd['round']}:{br['node']}:diag", kind="diag", image=rd["canvas"], node=br["node"], ntype=br["ntype"], op=br["op"], pid=pid, run=run_name))
                if "edit" in kinds:
                    for ci, cj in enumerate(br.get("candidates", [])):
                        items.append(dict(id=f"{run_name}:{pid}:r{rd['round']}:{br['node']}:c{ci}", kind="edit", before=rd["canvas"], after=cj["path"], raw=cj.get("raw"), op=br["op"], node=br["node"], ntype=br["ntype"], instruction=br["instruction"], region=br.get("region"), accept=cj["accept"], rules={k: v.get("ok") for k, v in cj["rules"].items()}, pid=pid, run=run_name))
    return items


def attach_questions(items, plans):
    """plans: dict pid -> Plan with generated nodes. Fills question/region text for cond and diag items."""
    for it in items:
        p = plans.get(it["pid"])
        if p is None:
            continue
        if it["kind"] == "cond":
            v = it["cid"].split(":", 1)[1]
            if v in p.nodes:
                it["question"] = p.nodes[v]["question"]
                it["desc"] = p.nodes[v]["desc"]
        elif it["kind"] == "diag":
            v = it["node"]
            if v in p.nodes:
                it["question"] = p.nodes[v]["question"]
                it["desc"] = p.nodes[v]["desc"]
    return items


def make_sheet(items, out_html, title="labels", rel=None, embed=False, side=768):
    """HTML sheet: radio buttons per item, export button writes labels JSON. embed=True inlines the images (downscaled
    to `side`, JPEG) so the page can be sent to someone without the run directories; otherwise images are linked relative
    to the page."""
    rel = rel or os.path.dirname(os.path.abspath(out_html))

    def src(p):
        if p is None:
            return ""
        if embed:
            import base64
            import io

            from .views import load_image, to_rgb

            im = to_rgb(load_image(p))
            if max(im.size) > side:
                im = im.resize((int(im.size[0] * side / max(im.size)), int(im.size[1] * side / max(im.size))))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=88)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return html.escape(os.path.relpath(p, rel))

    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font-family:sans-serif;margin:16px}} .it{{border-bottom:1px solid #ccc;padding:10px 0}} img{{max-width:520px;max-height:420px;margin-right:8px;border:1px solid #999}} .q{{font-weight:bold}} .meta{{color:#666;font-size:12px}}</style></head><body>",
             f"<h2>{html.escape(title)}</h2><p>y = 完全成立；u = 东西在但细节不对；n = 没有或明显不对。编辑对子：fixed 目标修好了没有，damaged 别处弄坏了没有，duplicate 多出一份没有。诊断：missing / wrong / extra / merged / fine。</p>",
             "<button onclick='exportLabels()'>导出标注 JSON</button> <span id='n'></span><div id='items'>"]
    for it in items:
        iid = html.escape(it["id"])
        if it["kind"] == "cond":
            parts.append(f"<div class='it' data-id='{iid}'><div class='q'>{html.escape(it.get('question') or it['cid'])}</div><div class='meta'>{iid} · {html.escape(str(it.get('ntype')))} · score {it.get('score')}</div><img src='{src(it['image'])}'>"
                         + "".join(f"<label><input type='radio' name='{iid}' value='{v}'> {v}</label> " for v in ("y", "u", "n")) + f"<input type='text' name='{iid}:note' placeholder='备注' size='30'></div>")
        elif it["kind"] == "edit":
            parts.append(f"<div class='it' data-id='{iid}'><div class='q'>[{html.escape(it['op'])}] {html.escape(it['instruction'])}</div><div class='meta'>{iid} · accept={it.get('accept')} · rules={html.escape(json.dumps(it.get('rules')))}</div><img src='{src(it['before'])}'><img src='{src(it['after'])}'>"
                         + "".join(f"<span> {k}: " + "".join(f"<label><input type='radio' name='{iid}:{k}' value='{v}'> {v}</label>" for v in ("y", "n")) + "</span>" for k in ("fixed", "damaged", "duplicate")) + f"<input type='text' name='{iid}:note' placeholder='备注' size='30'></div>")
        else:
            parts.append(f"<div class='it' data-id='{iid}'><div class='q'>{html.escape(it.get('desc') or it['node'])}</div><div class='meta'>{iid} · {html.escape(str(it.get('ntype')))} · system said {html.escape(str(it.get('op')))}</div><img src='{src(it['image'])}'>"
                         + "".join(f"<label><input type='radio' name='{iid}' value='{v}'> {v}</label> " for v in ("missing", "wrong", "extra", "merged", "fine")) + f"<input type='text' name='{iid}:note' placeholder='备注' size='30'></div>")
    parts.append("</div><script>function exportLabels(){const out=[];document.querySelectorAll('.it').forEach(d=>{const id=d.dataset.id;const r={id:id};const rs=d.querySelectorAll('input[type=radio]:checked');if(!rs.length)return;rs.forEach(x=>{const k=x.name===id?'label':x.name.slice(id.length+1);r[k]=x.value;});const n=d.querySelector('input[type=text]');if(n&&n.value)r.note=n.value;out.push(r);});const blob=new Blob([out.map(x=>JSON.stringify(x)).join('\\n')+'\\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='labels_human.jsonl';a.click();document.getElementById('n').textContent=out.length+' 条';}</script></body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return out_html


def agreement(a, b, kind="cond"):
    """Agreement between two label files for cond items (3-way) or edit items (per field)."""
    want = ("label",) if kind != "edit" else ("fixed", "damaged", "duplicate")
    la = {r["id"]: r for r in a if any(r.get(k) is not None for k in want)}
    lb = {r["id"]: r for r in b if any(r.get(k) is not None for k in want)}
    common = [i for i in la if i in lb]
    if kind == "cond":
        agree = sum(1 for i in common if la[i].get("label") == lb[i].get("label"))
        strict = sum(1 for i in common if (la[i].get("label") == "y") == (lb[i].get("label") == "y"))
        return dict(n=len(common), exact=agree / max(1, len(common)), y_vs_rest=strict / max(1, len(common)))
    out = dict(n=len(common))
    for k in ("fixed", "damaged", "duplicate"):
        out[k] = sum(1 for i in common if la[i].get(k) == lb[i].get(k)) / max(1, len(common))
    return out


def effective_label(item, label):
    """Labels answer the item's question literally. For an inverted question (count "N+1 or more?", absent "any X?",
    quality) a literal y means the condition FAILS, so the condition-level label is flipped; u is kept."""
    if item.get("inv") and label in ("y", "n"):
        return "n" if label == "y" else "y"
    return label


def eval_metrics(items, ref, scores, thr=0.5, size_bins=(0.02, 0.08)):
    """Evaluator metrics against reference labels for cond items (condition-level labels, see effective_label).
    scores: dict item id -> final score (after two-stage). Returns false-fail rate on y, detection on n, detection on u, by type and region size."""
    by_id = {it["id"]: it for it in items}
    ref = {r["id"]: effective_label(by_id.get(r["id"], {}), r.get("label")) for r in ref}
    tot = dict(y=[0, 0], u=[0, 0], n=[0, 0])
    by_type, by_size = {}, {}
    for it in items:
        if it["kind"] != "cond" or it["id"] not in ref or it["id"] not in scores:
            continue
        lab = ref[it["id"]]
        if lab not in tot:
            continue
        pred_fail = scores[it["id"]] < thr
        t = it.get("ntype") or "?"
        key = ("fail" if pred_fail else "pass")
        tot[lab][0] += 1
        tot[lab][1] += int(pred_fail)
        bt = by_type.setdefault(t, dict(y=[0, 0], u=[0, 0], n=[0, 0]))
        bt[lab][0] += 1
        bt[lab][1] += int(pred_fail)
        r = it.get("region")
        if r and it.get("image_size"):
            W, H = it["image_size"]
            a = max(0.0, (r[2] - r[0]) * (r[3] - r[1])) / float(W * H)
            sb = "small" if a < size_bins[0] else ("medium" if a < size_bins[1] else "large")
            bs = by_size.setdefault(sb, dict(y=[0, 0], u=[0, 0], n=[0, 0]))
            bs[lab][0] += 1
            bs[lab][1] += int(pred_fail)

    def rates(d):
        return dict(false_fail=(d["y"][1] / d["y"][0]) if d["y"][0] else None, detect_n=(d["n"][1] / d["n"][0]) if d["n"][0] else None, detect_u=(d["u"][1] / d["u"][0]) if d["u"][0] else None, n_y=d["y"][0], n_u=d["u"][0], n_n=d["n"][0])

    return dict(overall=rates(tot), by_type={k: rates(v) for k, v in by_type.items()}, by_size={k: rates(v) for k, v in by_size.items()})


def acceptance_metrics(items, ref):
    """Precision and recall of the system's acceptance (and of each rule) against edit labels. A good edit: fixed=y, damaged=n, duplicate=n."""
    ref = {r["id"]: r for r in ref}
    out = {}
    names = ["accept"] + sorted({k for it in items if it["kind"] == "edit" for k in (it.get("rules") or {})})
    for name in names:
        tp = fp = fn = tn = 0
        for it in items:
            if it["kind"] != "edit" or it["id"] not in ref:
                continue
            r = ref[it["id"]]
            good = r.get("fixed") == "y" and r.get("damaged") == "n" and r.get("duplicate") == "n"
            pred = it.get("accept") if name == "accept" else (it.get("rules") or {}).get(name)
            if pred is None:
                continue
            if pred and good:
                tp += 1
            elif pred and not good:
                fp += 1
            elif (not pred) and good:
                fn += 1
            else:
                tn += 1
        out[name] = dict(precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None, tp=tp, fp=fp, fn=fn, tn=tn)
    return out


def diagnosis_metrics(items, ref):
    ref = {r["id"]: r.get("label") for r in ref}
    m = {}
    conf = {}
    for it in items:
        if it["kind"] != "diag" or it["id"] not in ref:
            continue
        truth = ref[it["id"]]
        sysop = {"add": "missing", "fix": "wrong", "remove": "extra", "split": "merged"}.get(it.get("op"), it.get("op"))
        conf.setdefault(truth, {}).setdefault(sysop, 0)
        conf[truth][sysop] += 1
    n = sum(sum(v.values()) for v in conf.values())
    correct = sum(conf.get(k, {}).get(k, 0) for k in conf)
    wrong_as_missing = conf.get("wrong", {}).get("missing", 0) / max(1, sum(conf.get("wrong", {}).values()))
    fine_targeted = sum(conf.get("fine", {}).values()) / max(1, n)
    return dict(n=n, accuracy=correct / max(1, n), wrong_as_missing=wrong_as_missing, fine_targeted=fine_targeted, confusion=conf)


COND_PROMPT = ("You are labelling an image for a dataset. Question about the image: \"{question}\"\n"
               "Answer exactly one label: y = clearly true in the image; u = the thing asked about is present but a detail does not match "
               "(wrong attribute, pose, state, place, or it is unclear); n = not in the image or clearly false. "
               "The second image, when given, is a close-up of the region where it should be; judge from both. "
               "Reply with one JSON object: {{\"label\": \"y|u|n\", \"note\": \"<at most 15 words>\"}}.")
EDIT_PROMPT = ("Two versions of the same image: BEFORE (first) and AFTER (second). An image editor was told: \"{instruction}\"\n"
               "Judge three things. fixed: is the requested change now true in AFTER (y/n)? damaged: did anything outside the requested change "
               "get worse, removed, distorted, or altered in AFTER (y/n)? duplicate: does AFTER contain an extra copy of a person or object "
               "that BEFORE had once (y/n)? Reply with one JSON object: {{\"fixed\": \"y|n\", \"damaged\": \"y|n\", \"duplicate\": \"y|n\", \"note\": \"<at most 15 words>\"}}.")
DIAG_PROMPT = ("You are labelling why an image fails a requirement. Requirement: \"{desc}\". Checker question: \"{question}\"\n"
               "Choose exactly one: missing = the required thing is not in the image at all; wrong = something of that kind is there but does not match; "
               "extra = there are more of them than required; merged = two required individuals appear as one; fine = the requirement is actually met. "
               "Reply with one JSON object: {{\"label\": \"missing|wrong|extra|merged|fine\", \"note\": \"<at most 15 words>\"}}.")


def _parse_json(text):
    import re

    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def label_items_with_model(items, vlm, side=1024, crop_min=448, workers=None, progress=None):
    """Independent labelling of a label set by a model (an API model of another family, for the second opinion;
    never the evaluator under test). Never looks at any score. Returns labels_<who> rows: {id, label|fields, note, raw}."""
    from .views import crop_box, load_image, resize_min_side, to_rgb

    jobs, meta = [], []
    for it in items:
        k = it["kind"]
        if k == "cond":
            if not it.get("question"):
                continue
            im = to_rgb(load_image(it["image"]))
            full = resize_min_side(im, side) if min(im.size) > side else im
            ims = [full]
            if it.get("region"):
                ims.append(crop_box(im, it["region"], 0.25, min_side=crop_min, max_side=side)[0])
            jobs.append(dict(images=ims, text=COND_PROMPT.format(question=it["question"].strip())))
        elif k == "edit":
            b = to_rgb(load_image(it["before"]))
            a = to_rgb(load_image(it["after"]))
            b = resize_min_side(b, side) if min(b.size) > side else b
            a = resize_min_side(a, side) if min(a.size) > side else a
            jobs.append(dict(images=[b, a], text=EDIT_PROMPT.format(instruction=it["instruction"].strip())))
        elif k == "diag":
            if not it.get("question"):
                continue
            im = to_rgb(load_image(it["image"]))
            full = resize_min_side(im, side) if min(im.size) > side else im
            jobs.append(dict(images=[full], text=DIAG_PROMPT.format(desc=(it.get("desc") or "").strip(), question=it["question"].strip())))
        else:
            continue
        meta.append(it)
    outs = vlm.generate(jobs, max_tokens=80)
    rows = []
    for it, txt in zip(meta, outs):
        d = _parse_json(txt)
        k = it["kind"]
        if k == "cond":
            lab = str(d.get("label", "")).strip().lower()[:1]
            rows.append(dict(id=it["id"], label=lab if lab in ("y", "u", "n") else None, note=d.get("note"), raw=(txt or "")[:200]))
        elif k == "edit":
            r = dict(id=it["id"], note=d.get("note"), raw=(txt or "")[:200])
            for f in ("fixed", "damaged", "duplicate"):
                v = str(d.get(f, "")).strip().lower()[:1]
                r[f] = v if v in ("y", "n") else None
            rows.append(r)
        else:
            lab = str(d.get("label", "")).strip().lower()
            rows.append(dict(id=it["id"], label=lab if lab in ("missing", "wrong", "extra", "merged", "fine") else None, note=d.get("note"), raw=(txt or "")[:200]))
        if progress:
            progress(len(rows), len(meta))
    return rows
