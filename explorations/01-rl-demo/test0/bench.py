from __future__ import annotations

import copy
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import svg as S
from .attribution import attribute
from .backends import Backends, build_backends
from .baseline import Baseline, element_dag
from .bench_cases import CASES, case_by_id
from .config import Config, load_config
from .cost import CostMeter
from .dag import ACTION, BACKGROUND, DAG
from .judge import score as judge_score
from .pipeline import run_prompt, write_json
from .render import overlay_boxes
from .report import build_report, to_markdown
from .scheduler import Scheduler
from .store import Store

FAULT_KINDS = ("color", "remove")


def _recolor(elem: ET.Element, avoid: Sequence[str], rng: random.Random) -> None:
    palette = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown"]
    for shape, _m in S.walk_shapes(elem):
        fill = shape.get("fill")
        rgb = S.parse_color(fill)
        if rgb is None:
            continue
        options = [c for c in palette if not any(S.color_matches(S.parse_color(c), a) for a in avoid) and S.color_distance(S.parse_color(c), rgb) > 120]
        if not options:
            options = ["purple"]
        shape.set("fill", "#%02x%02x%02x" % S.parse_color(rng.choice(options)))


def inject_fault_dag(composed_svg: str, dag: DAG, rng: random.Random, kind: str) -> Optional[Tuple[str, str]]:
    root = S.parse_svg_text(composed_svg)
    consumed = dag.consumed()
    candidates = [n for n in dag.draft_nodes() if n.type != BACKGROUND and n.type != ACTION and root.find(f".//g[@data-node='{n.id}']") is not None]
    if kind == "color":
        candidates = [n for n in candidates if n.colors()]
    if not candidates:
        return None
    victim = rng.choice(candidates)
    group = root.find(f".//g[@data-node='{victim.id}']")
    if kind == "color":
        _recolor(group, victim.colors(), rng)
    else:
        parent_map = {child: parent for parent in root.iter() for child in parent}
        parent_map[group].remove(group)
    truth = consumed.get(victim.id, victim.id)
    return S.serialize(root), truth


def inject_fault_baseline(svg_text: str, elements: List[Dict[str, Any]], rng: random.Random, kind: str) -> Optional[Tuple[str, str]]:
    root = S.parse_svg_text(svg_text)
    parent_map = {child: parent for parent in root.iter() for child in parent}
    candidates = [el for el in elements if el.get("type") != "background"]
    if kind == "color":
        candidates = [el for el in candidates if any(S.normalize_color_name(t) in S.NAMED_COLORS for t in el["desc"].lower().split())]
    if not candidates:
        return None
    victim = rng.choice(candidates)
    b = victim["bbox"]
    inside = []
    for shape, m in S.walk_shapes(root):
        sb = S.shape_bounds(shape, m)
        if sb is None:
            continue
        cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and (sb[2] - sb[0]) <= 1.2 * (b[2] - b[0]) and (sb[3] - sb[1]) <= 1.2 * (b[3] - b[1]):
            inside.append(shape)
    if not inside:
        return None
    avoid = [t for t in victim["desc"].lower().split() if S.normalize_color_name(t) in S.NAMED_COLORS]
    if kind == "color":
        for shape in inside:
            rgb = S.parse_color(shape.get("fill"))
            if rgb is None:
                continue
            options = [c for c in ("red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown") if not any(S.color_matches(S.parse_color(c), a) for a in avoid) and S.color_distance(S.parse_color(c), rgb) > 120] or ["purple"]
            shape.set("fill", "#%02x%02x%02x" % S.parse_color(rng.choice(options)))
    else:
        for shape in inside:
            parent = parent_map.get(shape)
            if parent is not None and shape in list(parent):
                parent.remove(shape)
    return S.serialize(root), victim["id"]


def localization_dag(scheduler: Scheduler, result: Dict[str, Any], rng: random.Random, faults: int, run_dir: Path) -> List[Dict[str, Any]]:
    out = []
    composed = result.get("_composed_svg") or ""
    if not composed:
        return out
    for k in range(faults):
        kind = FAULT_KINDS[k % len(FAULT_KINDS)]
        injected = inject_fault_dag(composed, scheduler.dag, rng, kind)
        if injected is None:
            continue
        svg_text, truth = injected
        png = run_dir / f"fault_{k}_{kind}.png"
        scheduler.b.renderer.render(svg_text, str(png), node="fault", phase="repair")
        overlay = run_dir / f"fault_{k}_{kind}_overlay.png"
        overlay_boxes(str(png), str(overlay), [{"bbox": item["bbox"], "label": item["node"]} for item in result["legend"] if item["type"] != BACKGROUND], tuple(scheduler.dag.canvas))
        scheduler.cost.phase = "repair"
        review = scheduler.b.reviewer.scene(scheduler.dag, scheduler.layout, svg_text, str(png), str(overlay))
        blamed = attribute(review.get("issues", []), scheduler.dag, "scene")
        blamed_ids = [nid for nid in blamed.keys() if nid != scheduler.dag.scene_id() or len(blamed) == 1]
        hit = truth in blamed
        out.append({"kind": kind, "truth": truth, "blamed": sorted(blamed.keys()), "detected": not review.get("passed", False), "hit": hit, "precision": (1.0 / len(blamed)) if hit and blamed else 0.0, "issues": len(review.get("issues", []))})
    return out


def localization_baseline(baseline: Baseline, result: Dict[str, Any], rng: random.Random, faults: int, run_dir: Path) -> List[Dict[str, Any]]:
    out = []
    svg_text = result.get("state_svg") or ""
    if not svg_text or not result.get("elements"):
        return out
    pseudo = element_dag(result["prompt"], result["canvas"], result["elements"])
    for k in range(faults):
        kind = FAULT_KINDS[k % len(FAULT_KINDS)]
        injected = inject_fault_baseline(svg_text, result["elements"], rng, kind)
        if injected is None:
            continue
        faulty, truth = injected
        review = baseline.review_svg(result["prompt"], result, faulty, run_dir, f"fault_{k}_{kind}")
        blamed = attribute(review.get("issues", []), pseudo, "scene")
        hit = truth in blamed
        out.append({"kind": kind, "truth": truth, "blamed": sorted(blamed.keys()), "detected": not review.get("passed", False), "hit": hit, "precision": (1.0 / len(blamed)) if hit and blamed else 0.0, "issues": len(review.get("issues", []))})
    return out


def _metrics(result: Dict[str, Any], judge: Dict[str, Any], localization: List[Dict[str, Any]]) -> Dict[str, Any]:
    cost = result["cost"]
    by_phase = cost["by_phase"]
    reuse = result["reuse"]
    demanded = max(1, int(reuse.get("demanded", 0)))
    return {
        "judge_pass_rate": judge["pass_rate"],
        "judge_passed": judge["passed"],
        "judge_total": judge["total"],
        "scene_passed": bool(result["scene_passed"]),
        "final_passed": bool(result["final_passed"]),
        "scene_rounds": int(result["scene_rounds"]),
        "final_rounds": int(result["final_rounds"]),
        "cost_total": float(cost["units"]),
        "cost_initial": float(by_phase["initial"]["units"]) + float(by_phase["plan"]["units"]),
        "cost_repair": float(by_phase["repair"]["units"]),
        "calls": cost["counts"],
        "tokens": cost["tokens"],
        "reuse_rate": float(reuse.get("store_hits", 0)) / demanded,
        "store_hits": int(reuse.get("store_hits", 0)),
        "demanded": int(reuse.get("demanded", 0)),
        "skipped_clean": int(reuse.get("skipped_clean", 0)),
        "repair_scope": _repair_scope(result),
        "node_retries": int(sum(int(r.get("rounds") or 0) for r in result.get("records", []))),
        "localization_hits": sum(1 for l in localization if l["hit"]),
        "localization_trials": len(localization),
        "localization_precision": (sum(l["precision"] for l in localization) / len(localization)) if localization else None,
        "localization_detected": sum(1 for l in localization if l["detected"]),
        "seconds": float(result.get("seconds", 0.0)),
    }


def _repair_scope(result: Dict[str, Any]) -> Optional[float]:
    total_nodes = None
    if result.get("dag"):
        total_nodes = len([n for n in result["dag"]["nodes"] if n["type"] in ("background", "entity", "object", "action")])
    elif result.get("elements"):
        total_nodes = len(result["elements"])
    if not total_nodes:
        return None
    scopes = []
    for rep in result.get("repairs", []):
        affected = [a for a in rep.get("affected", []) if a not in ("scene", "final")]
        scopes.append(len(affected) / float(total_nodes))
    return (sum(scopes) / len(scopes)) if scopes else None


def run_case(case: Dict[str, Any], system: str, cfg: Config, backends: Backends, out_dir: Path, faults: int, seed: int, oracle_plan: bool, drafter: Any = None, editor: Any = None) -> Dict[str, Any]:
    run_dir = out_dir / case["id"] / system
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    backends.cost.events.clear()
    backends.judge_cost.events.clear()
    if backends.store is not None:
        backends.store.reset_stats()
    hint = case.get("hint") if (cfg.mock or oracle_plan) else None
    if system == "dag":
        result = run_prompt(case["prompt"], cfg, backends, run_dir=str(run_dir), hint=hint, drafter=drafter, editor=editor)
        planned = DAG.from_dict(result["dag"])
        scheduler = Scheduler(planned, backends, cfg, str(run_dir))
        scheduler.layout = result["layout"]
        scheduler.composed = {"svg": Path(result["draft_svg"]).read_text(encoding="utf-8") if result.get("draft_svg") else "", "legend": result["legend"]}
        result["_composed_svg"] = scheduler.composed["svg"]
        localization = localization_dag(scheduler, result, rng, faults, run_dir) if faults > 0 else []
        result.pop("_composed_svg", None)
    else:
        baseline = Baseline(cfg, backends)
        result = baseline.run(case["prompt"], str(run_dir), hint)
        localization = localization_baseline(baseline, result, rng, faults, run_dir) if faults > 0 else []
    result["cost"] = backends.cost.summary()
    judge_raw = backends.judge.judge(case["prompt"], case["checks"], result.get("final_png"), result.get("structure") or {})
    judge = judge_score(judge_raw.get("results", []), case["checks"])
    judge["results"] = judge_raw.get("results", [])
    metrics = _metrics(result, judge, localization)
    record = {
        "case": case["id"],
        "title": case["title"],
        "system": system,
        "prompt": case["prompt"],
        "run_dir": str(run_dir),
        "metrics": metrics,
        "judge": judge,
        "localization": localization,
        "final_png": result.get("final_png"),
        "draft_png": result.get("draft_png"),
        "judge_cost": backends.judge_cost.summary()["units"],
    }
    write_json(run_dir / "bench_record.json", record)
    return record


def run_bench(cfg: Config, case_ids: Optional[Sequence[str]] = None, systems: Sequence[str] = ("dag", "baseline"), faults: int = 2, out_dir: Optional[str] = None, seed: int = 0, oracle_plan: bool = False, backends: Optional[Backends] = None, drafter: Any = None, editor: Any = None, quiet: bool = False) -> Dict[str, Any]:
    cases = [case_by_id(c) for c in case_ids] if case_ids else list(CASES)
    out = Path(out_dir) if out_dir else Path(str(cfg.RUNS_DIR)) / f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    own_backends = backends is None
    if backends is None:
        store = Store(str(out / "store"), enabled=bool(cfg.STORE_ENABLED))
        backends = build_backends(cfg, store=store)
    records: List[Dict[str, Any]] = []
    try:
        for i, case in enumerate(cases):
            for system in systems:
                record = run_case(case, system, cfg, backends, out, faults, seed + i, oracle_plan, drafter=drafter, editor=editor)
                records.append(record)
                if not quiet:
                    print(f"[{case['id']}/{system}] judge={record['metrics']['judge_pass_rate']:.2f} cost={record['metrics']['cost_total']:.1f} repair={record['metrics']['cost_repair']:.1f} reuse={record['metrics']['reuse_rate']:.2f} loc={record['metrics']['localization_hits']}/{record['metrics']['localization_trials']}", flush=True)
                write_json(out / "bench.json", {"config": cfg.to_dict(), "records": records})
    finally:
        if own_backends:
            backends.close()
    report = build_report(records)
    write_json(out / "report.json", report)
    (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    report["out_dir"] = str(out)
    return report
