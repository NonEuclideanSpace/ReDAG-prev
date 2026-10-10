from .conditions import region_of
from .schema import union_box
from .views import crop_box, load_image, pair_image, to_rgb

AUDIT_HEAD = 'The left half is an original picture and the right half is the same picture after an edit. The edit was supposed to do this: "{instr}" '
AUDIT = dict(
    done=AUDIT_HEAD + "Is the requested change clearly visible in the right half?",
    intact=AUDIT_HEAD + "Apart from that change, does the right half still show all the people and objects of the left half, with nobody and nothing removed, replaced or distorted?",
    plausible=AUDIT_HEAD + "In the right half, does the new or changed element have a realistic size, position and appearance relative to the rest of the scene?",
)


class AcceptConfig:
    def __init__(self, margin=0.05, drop_tol=0.15, audit_rules=("intact", "done"), audit_thr=0.5, local_margin=0.0):
        self.margin = float(margin)
        self.drop_tol = float(drop_tol)
        self.audit_rules = tuple(audit_rules)
        self.audit_thr = float(audit_thr)
        self.local_margin = float(local_margin)


def target_cids(plan, target):
    op, t, v = target["op"], target["ntype"], target["node"]
    if op == "split":
        return [target["cid"]] + ["n:" + u for u in target["participants"]]
    return [target["cid"]]


def judge(plan, conds, vlm, canvas, col0, committed, col1, info, target, affected, cfg=None):
    """Five acceptance rules on a committed candidate. Every rule's outcome is recorded; `accept` is their conjunction."""
    cfg = cfg or AcceptConfig()
    rules = {}
    tc = target_cids(plan, target)
    main = target["cid"]
    s0, s1 = col0.score(main, 0.0), col1.score(main, 0.0)
    r1 = all(col1.passes(c) for c in tc) and s1 >= s0 + cfg.margin
    rules["r1_established"] = dict(ok=bool(r1), before=s0, after=s1, cids=tc)

    c = next(x for x in conds if x.cid == main)
    region = union_box([target.get("region")] + list(info.get("boxes", [])))
    if region is None:
        region = region_of(plan, c, col1.boxes)
    before, after = to_rgb(load_image(canvas)), to_rgb(load_image(committed))
    if region is not None and c.ntype not in ("count", "absent", "quality"):
        va, _ = crop_box(before, region, 0.25)
        vb, _ = crop_box(after, region, 0.25)
        (pa, _), (pb, _) = vlm.yesno([dict(image=va, question=c.question, tpl="a", inv=c.inv), dict(image=vb, question=c.question, tpl="a", inv=c.inv)])
    else:
        pa, pb = s0, s1
    r2 = pb >= pa + cfg.local_margin
    rules["r2_local_gain"] = dict(ok=bool(r2), before=float(pa), after=float(pb))

    drops = {}
    worst = 0.0
    desc_of_target = set(plan.descendants(target["node"])) | set(tc)
    for x in conds:
        if x.cid in desc_of_target or x.kind == "quality":
            continue
        if not col0.passes(x.cid):
            continue
        if x.cid not in affected:
            continue
        d = col1.score(x.cid, 0.0) - col0.score(x.cid, 0.0)
        drops[x.cid] = round(float(d), 3)
        worst = min(worst, d)
    r3 = worst >= -cfg.drop_tol
    rules["r3_no_damage"] = dict(ok=bool(r3), worst=float(worst), drops=drops)

    cnt_ok = True
    details = {}
    for x in conds:
        if x.kind == "count" or (x.kind == "pair" and col0.passes(x.cid)):
            ok = col1.passes(x.cid) or col1.status(x.cid) == "unasked"
            details[x.cid] = bool(ok)
            if x.kind == "count" and target["op"] in ("add", "split", "fix") and not ok:
                cnt_ok = False
            if x.kind == "pair" and not ok and target["op"] != "split":
                cnt_ok = False
    rules["r4_no_duplicate"] = dict(ok=bool(cnt_ok), details=details)

    instr = target.get("instruction", "").replace(" Keep everything else in the image unchanged.", "").strip()
    pr = pair_image(before, after, region)
    items = [dict(image=pr, question=AUDIT[k].format(instr=instr), tpl="a", inv=False) for k in ("done", "intact", "plausible")]
    res = vlm.yesno(items)
    audit = {k: float(p) for k, (p, _) in zip(("done", "intact", "plausible"), res)}
    r5 = all(audit[k] >= cfg.audit_thr for k in cfg.audit_rules)
    rules["r5_audit"] = dict(ok=bool(r5), scores=audit, used=list(cfg.audit_rules))

    accept = bool(r1 and r2 and r3 and cnt_ok and r5)
    return dict(accept=accept, rules=rules, region=region)
