from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .backends import Backends, build_backends
from .config import Config, load_config
from .cost import CostMeter
from .dag import DAG
from .scheduler import Scheduler


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def new_run_dir(cfg: Config, name: str = "run") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(str(cfg.RUNS_DIR))
    candidate = root / f"{stamp}-{name}"
    k = 2
    while candidate.exists():
        candidate = root / f"{stamp}-{name}-{k}"
        k += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def plan_dag(prompt: str, backends: Backends, hint: Optional[Dict[str, Any]] = None) -> DAG:
    backends.cost.phase = "plan"
    catalog = backends.store.catalog() if backends.store is not None else []
    dag = backends.planner.plan(prompt, catalog, hint)
    return dag


def run_prompt(
    prompt: str,
    cfg: Config,
    backends: Backends,
    run_dir: Optional[str] = None,
    hint: Optional[Dict[str, Any]] = None,
    dag: Optional[DAG] = None,
    drafter: Any = None,
    layouter: Any = None,
    editor: Optional[Callable] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    run_path = Path(run_dir) if run_dir else new_run_dir(cfg)
    run_path.mkdir(parents=True, exist_ok=True)
    events_path = run_path / "events.jsonl"
    handle = events_path.open("a", encoding="utf-8")

    def emit(event: Dict[str, Any]) -> None:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        if on_event is not None:
            on_event(event)

    started = time.time()
    backends.cost.events.clear()
    if backends.store is not None:
        backends.store.reset_stats()
    try:
        if dag is None:
            dag = plan_dag(prompt, backends, hint)
        write_json(run_path / "plan.json", dag.to_dict())
        if hasattr(backends.reviewer, "reference_dag"):
            backends.reviewer.reference_dag = dag.clone()
        scheduler = Scheduler(dag, backends, cfg, str(run_path), on_event=emit, drafter=drafter, layouter=layouter, editor=editor)
        result = scheduler.run()
        result["run_dir"] = str(run_path)
        result["seconds"] = round(time.time() - started, 3)
        result["status"] = "completed"
        write_json(run_path / "result.json", result)
        write_json(run_path / "dag.json", dag.to_dict())
        return result
    finally:
        handle.close()


def run_cli(prompt: str, config_path: Optional[str], overrides: Dict[str, Any], run_dir: Optional[str] = None, hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_config(config_path, overrides)
    backends = build_backends(cfg)
    try:
        return run_prompt(prompt, cfg, backends, run_dir=run_dir, hint=hint)
    finally:
        backends.close()
