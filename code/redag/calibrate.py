import json
import os
import time

from .conditions import build_conditions
from .labels import effective_label, eval_metrics, read_jsonl
from .schema import Plan, add_generated_nodes
from .verify import Verifier, VerifyConfig

GATE = dict(false_fail_max=0.02, detect_n_min=0.90, detect_u_min=0.70, dup_detect_min=0.80, fmt_fail_max=0.01, min_items=300)


def score_items_with_production_path(vlm, items, plans, vcfg):
    """Group cond items by image, run the production verification on each image once, read the scores."""
    by_img = {}
    for it in items:
        if it["kind"] == "cond":
            by_img.setdefault((it["pid"], it["image"]), []).append(it)
    scores, views = {}, {}
    t0 = time.time()
    for (pid, img), its in by_img.items():
        plan = plans[pid]
        conds = build_conditions(plan)
        ver = Verifier(vlm, plan, conds, vcfg)
        col = ver.verify(img, tag="calib")
        for it in its:
            r = col.recs.get(it["cid"])
            if r is None or r.s is None:
                continue
            scores[it["id"]] = float(r.s)
            views[it["id"]] = dict(v1=r.v1, v2=r.v2, s1=r.s1, s2=r.s2, status=col.status(it["cid"]))
    return scores, views, time.time() - t0


def load_plans(plan_dir, pids):
    out = {}
    for pid in pids:
        p = os.path.join(plan_dir, pid + ".json")
        if os.path.exists(p):
            out[pid] = add_generated_nodes(Plan.load(p))
    return out


def gate_decision(metrics, fmt_fail_rate, n_items, dup_detect=None, thresholds=GATE):
    o = metrics["overall"]
    checks = dict(
        enough_items=n_items >= thresholds["min_items"],
        false_fail=(o["false_fail"] is not None and o["false_fail"] <= thresholds["false_fail_max"]),
        detect_n=(o["detect_n"] is not None and o["detect_n"] >= thresholds["detect_n_min"]),
        detect_u=(o["detect_u"] is not None and o["detect_u"] >= thresholds["detect_u_min"]),
        dup_detect=(dup_detect is None or dup_detect >= thresholds["dup_detect_min"]),
        fmt=fmt_fail_rate <= thresholds["fmt_fail_max"],
    )
    return all(checks.values()), checks


def run_calibration(vlm, items_path, ref_path, plan_dir, vcfg_kw=None, out_path=None, thresholds=GATE):
    items = read_jsonl(items_path)
    ref = read_jsonl(ref_path)
    pids = sorted({it["pid"] for it in items})
    plans = load_plans(plan_dir, pids)
    items = [it for it in items if it["pid"] in plans]
    vcfg = VerifyConfig(**(vcfg_kw or {}))
    coord = None
    try:
        coord = vlm.calibrate_coords()
    except Exception as ex:
        coord = dict(error=str(ex))
    scores, views, sec = score_items_with_production_path(vlm, items, plans, vcfg)
    metrics = eval_metrics(items, ref, scores, thr=vcfg.thr)
    by_id = {it["id"]: it for it in items}
    refd = {r["id"]: effective_label(by_id.get(r["id"], {}), r.get("label")) for r in ref}
    dup_items = [it for it in items if it["kind"] == "cond" and it.get("ntype") == "count" and refd.get(it["id"]) == "n" and it["id"] in scores]
    dup_detect = (sum(1 for it in dup_items if scores[it["id"]] < vcfg.thr) / len(dup_items)) if dup_items else None
    st = vlm.stats()
    fmt_rate = st.get("fmt_fail", 0) / max(1, st.get("calls", 1))
    ok, checks = gate_decision(metrics, fmt_rate, len(scores), dup_detect, thresholds)
    gate = dict(model=vlm.model, settings=vars(vcfg), metrics=metrics, dup_detect=dup_detect, n_scored=len(scores), fmt_fail_rate=fmt_rate, coord_mode=coord.get("coord_mode") if isinstance(coord, dict) else None, coord_check=coord, checks=checks, thresholds=thresholds, sec=round(sec, 1), time=time.strftime("%F %T"), summary=f"false_fail={metrics['overall']['false_fail']} detect_n={metrics['overall']['detect_n']} detect_u={metrics['overall']['detect_u']} dup={dup_detect} fmt={fmt_rate:.4f} n={len(scores)}")
    gate["pass"] = bool(ok)
    out_path = out_path or vlm.gate_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gate, f, ensure_ascii=False, indent=1)
    with open(out_path.replace(".json", "_scores.json"), "w", encoding="utf-8") as f:
        json.dump(dict(scores=scores, views=views), f, ensure_ascii=False)
    return gate
