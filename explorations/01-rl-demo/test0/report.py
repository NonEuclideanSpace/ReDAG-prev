from __future__ import annotations

from typing import Any, Dict, List, Optional


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _fmt(value: Optional[float], digits: int = 2, pct: bool = False) -> str:
    if value is None:
        return "-"
    if pct:
        return f"{100.0 * value:.0f}%"
    return f"{value:.{digits}f}"


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ms = [r["metrics"] for r in records]
    if not ms:
        return {}
    loc_trials = sum(m["localization_trials"] for m in ms)
    loc_hits = sum(m["localization_hits"] for m in ms)
    demanded = sum(m["demanded"] for m in ms)
    hits = sum(m["store_hits"] for m in ms)
    repaired = [m for m in ms if m["cost_repair"] > 0]
    return {
        "cases": len(ms),
        "judge_pass_rate": _mean([m["judge_pass_rate"] for m in ms]),
        "judge_all_pass": sum(1 for m in ms if m["judge_total"] and m["judge_passed"] == m["judge_total"]) / len(ms),
        "scene_pass_rate": sum(1 for m in ms if m["scene_passed"]) / len(ms),
        "final_pass_rate": sum(1 for m in ms if m["final_passed"]) / len(ms),
        "localization_hit_rate": (loc_hits / loc_trials) if loc_trials else None,
        "localization_precision": _mean([m["localization_precision"] for m in ms]),
        "localization_detect_rate": (sum(m["localization_detected"] for m in ms) / loc_trials) if loc_trials else None,
        "cost_total": _mean([m["cost_total"] for m in ms]),
        "cost_initial": _mean([m["cost_initial"] for m in ms]),
        "cost_repair": _mean([m["cost_repair"] for m in ms]),
        "cost_repair_share": _mean([(m["cost_repair"] / m["cost_total"]) if m["cost_total"] else 0.0 for m in ms]),
        "repair_scope": _mean([m["repair_scope"] for m in ms]),
        "cases_repaired": len(repaired),
        "node_retries": _mean([m.get("node_retries", 0) for m in ms]),
        "reuse_rate": (hits / demanded) if demanded else 0.0,
        "skipped_clean": sum(m["skipped_clean"] for m in ms),
        "scene_rounds": _mean([m["scene_rounds"] for m in ms]),
        "final_rounds": _mean([m["final_rounds"] for m in ms]),
        "llm_calls": _mean([m["calls"]["llm"] for m in ms]),
        "vlm_calls": _mean([m["calls"]["vlm"] for m in ms]),
        "render_calls": _mean([m["calls"]["render"] for m in ms]),
        "diffusion_calls": _mean([m["calls"]["diffusion"] for m in ms]),
        "seconds": _mean([m["seconds"] for m in ms]),
    }


def build_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    systems = sorted({r["system"] for r in records})
    summary = {s: aggregate([r for r in records if r["system"] == s]) for s in systems}
    per_case: Dict[str, Dict[str, Any]] = {}
    for r in records:
        per_case.setdefault(r["case"], {"title": r["title"]})[r["system"]] = r["metrics"]
    return {"systems": systems, "summary": summary, "per_case": per_case, "records": len(records)}


ROWS = [
    ("check pass rate (judge)", "judge_pass_rate", True),
    ("all checks passed", "judge_all_pass", True),
    ("scene review passed", "scene_pass_rate", True),
    ("final review passed", "final_pass_rate", True),
    ("fault localization hit rate", "localization_hit_rate", True),
    ("fault localization precision", "localization_precision", True),
    ("fault detected by review", "localization_detect_rate", True),
    ("cost total (units)", "cost_total", False),
    ("cost initial (units)", "cost_initial", False),
    ("cost repair (units)", "cost_repair", False),
    ("repair share of cost", "cost_repair_share", True),
    ("repair scope (fraction of nodes redrawn)", "repair_scope", True),
    ("node-level retry rounds", "node_retries", False),
    ("reuse rate (store hits / node demands)", "reuse_rate", True),
    ("nodes skipped as clean during repairs", "skipped_clean", False),
    ("scene repair rounds", "scene_rounds", False),
    ("final repair rounds", "final_rounds", False),
    ("llm calls", "llm_calls", False),
    ("vlm calls", "vlm_calls", False),
    ("render calls", "render_calls", False),
    ("diffusion calls", "diffusion_calls", False),
    ("seconds per case", "seconds", False),
]


def to_markdown(report: Dict[str, Any]) -> str:
    systems = report.get("systems", [])
    lines = ["# DraftDAG bench report", "", f"records: {report.get('records', 0)}", "", "## Summary", ""]
    lines.append("| metric | " + " | ".join(systems) + " |")
    lines.append("|---|" + "---|" * len(systems))
    for label, key, pct in ROWS:
        cells = []
        for s in systems:
            value = report["summary"].get(s, {}).get(key)
            cells.append(_fmt(value, 2, pct) if isinstance(value, (int, float)) or value is None else str(value))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Per case", ""])
    lines.append("| case | " + " | ".join(f"{s} judge" for s in systems) + " | " + " | ".join(f"{s} cost" for s in systems) + " | " + " | ".join(f"{s} repair" for s in systems) + " | " + " | ".join(f"{s} loc" for s in systems) + " |")
    lines.append("|---|" + "---|" * (4 * len(systems)))
    for case_id, item in report.get("per_case", {}).items():
        judge = [_fmt(item.get(s, {}).get("judge_pass_rate"), 2, True) for s in systems]
        cost = [_fmt(item.get(s, {}).get("cost_total"), 1) for s in systems]
        repair = [_fmt(item.get(s, {}).get("cost_repair"), 1) for s in systems]
        loc = [f"{item.get(s, {}).get('localization_hits', 0)}/{item.get(s, {}).get('localization_trials', 0)}" if s in item else "-" for s in systems]
        lines.append(f"| {case_id} | " + " | ".join(judge) + " | " + " | ".join(cost) + " | " + " | ".join(repair) + " | " + " | ".join(loc) + " |")
    return "\n".join(lines) + "\n"
