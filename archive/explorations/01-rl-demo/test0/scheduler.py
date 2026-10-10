from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from . import svg as S
from .attribution import attribute, summarize_attribution
from .checks import check_fragment, check_layout, issues_from_checks
from .compose import compose
from .dag import ACTION, BACKGROUND, DAG, DRAFT_TYPES, ENTITY, FINAL, OBJECT, SCENE, Node
from .generate import generation_prompt
from .render import overlay_boxes


class Op:
    name = "op"

    def apply(self, dag: DAG) -> Set[str]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name}


class EditSpec(Op):
    name = "edit_spec"

    def __init__(self, node_id: str, new_spec: Dict[str, Any], feedback: Optional[Sequence[str]] = None, reason: str = ""):
        self.node_id = node_id
        self.new_spec = new_spec
        self.feedback = list(feedback or [])
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        node = dag[self.node_id]
        node.spec = json.loads(json.dumps(self.new_spec))
        node.feedback = list(dict.fromkeys(list(node.feedback) + self.feedback))[-6:]
        node.store_key = ""
        return dag.mark_dirty([self.node_id])

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "node": self.node_id, "reason": self.reason, "feedback": self.feedback}


class AdjustFinal(Op):
    name = "adjust_final"

    def __init__(self, node_id: str, reason: str = ""):
        self.node_id = node_id
        self.reason = reason

    def apply(self, dag: DAG) -> Set[str]:
        node = dag[self.node_id]
        node.spec["seed"] = int(node.spec.get("seed") or 0) + 1
        node.spec["control_scale"] = min(1.0, float(node.spec.get("control_scale") or 0.8) + 0.1)
        node.spec["strength"] = min(0.95, float(node.spec.get("strength") or 0.7) + 0.1)
        node.dirty = True
        return {self.node_id}

    def describe(self) -> Dict[str, Any]:
        return {"op": self.name, "node": self.node_id, "reason": self.reason}


class Scheduler:
    def __init__(
        self,
        dag: DAG,
        backends: Any,
        cfg: Any,
        run_dir: str,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        drafter: Any = None,
        layouter: Any = None,
        editor: Optional[Callable[["Scheduler", Dict[str, Any], Dict[str, List[Dict[str, Any]]], str], List[Op]]] = None,
    ):
        self.dag = dag
        self.b = backends
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event
        self.drafter = drafter or backends.drafter
        self.layouter = layouter or backends.layouter
        self.editor = editor
        self.cost = backends.cost
        self.store = backends.store
        self.events: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []
        self.scene_reviews: List[Dict[str, Any]] = []
        self.final_reviews: List[Dict[str, Any]] = []
        self.repairs: List[Dict[str, Any]] = []
        self.reuse = {"demanded": 0, "store_hits": 0, "executed": 0, "skipped_clean": 0}
        self._pending_repair = False
        self.layout: Dict[str, Any] = {"placements": []}
        self.composed: Dict[str, Any] = {"svg": "", "legend": []}
        self.draft_svg_path: Optional[str] = None
        self.draft_png: Optional[str] = None
        self.overlay_png: Optional[str] = None
        self.final_png: Optional[str] = None
        self.final_meta: Dict[str, Any] = {}
        self.scene_rounds = 0
        self.final_rounds = 0
        self.max_node_rounds = int(cfg.MAX_NODE_ROUNDS)

    def emit(self, kind: str, **payload: Any) -> None:
        event = {"t": round(time.time(), 3), "event": kind}
        event.update(payload)
        self.events.append(event)
        if self.on_event is not None:
            self.on_event(event)

    def phase_of(self, node: Node) -> str:
        return "initial" if node.version == 0 else "repair"

    def store_key(self, node: Node) -> str:
        base = node.spec_key()
        if node.type == ACTION:
            parts = [base]
            for pid in node.participants():
                out = self.dag[pid].output if pid in self.dag else ""
                parts.append(hashlib.sha1(str(out or "").encode("utf-8")).hexdigest())
            return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        if node.type == BACKGROUND:
            aspect = round(self.dag.canvas[0] / float(self.dag.canvas[1]), 2)
            return hashlib.sha1(f"{base}|{aspect}".encode("utf-8")).hexdigest()
        return base

    def fragment_document(self, fragment: str, node_id: str) -> str:
        root = S.parse_fragment(fragment, node_id)
        bbox = S.declared_bbox(root)
        if bbox is None:
            b = S.bounds(root) or (0.0, 0.0, 100.0, 100.0)
            bbox = (b[0], b[1], max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1]))
        x, y, w, h = bbox
        body = f'<g transform="translate({-x:.2f} {-y:.2f})">{S.serialize(root)}</g>'
        return S.svg_document(w, h, body)

    def render_fragment(self, node: Node, fragment: str, tag: str, phase: Optional[str] = None) -> Optional[str]:
        try:
            doc = self.fragment_document(fragment, node.id)
        except (ValueError, Exception):
            return None
        out = self.run_dir / "nodes" / f"{node.id}_v{node.version}_{tag}.png"
        self.b.renderer.render(doc, str(out), node=node.id, phase=phase or self.phase_of(node))
        return str(out)

    def deps_outputs(self, node: Node) -> Dict[str, str]:
        return {d: self.dag[d].output for d in node.deps if d in self.dag and isinstance(self.dag[d].output, str)}

    def execute_node(self, node_id: str) -> Dict[str, Any]:
        node = self.dag[node_id]
        phase = self.phase_of(node)
        self.cost.phase = phase
        self.reuse["demanded"] += 1
        t0 = time.time()
        key = self.store_key(node)
        entry = self.store.get(key) if self.store is not None else None
        record: Dict[str, Any] = {"node": node_id, "type": node.type, "phase": phase, "version": node.version, "rounds": 0}
        if entry is not None and float(entry.get("checks_score") or 0.0) >= 0.999 and entry.get("review_passed") is not False:
            node.output = entry["fragment_text"]
            node.png = self.store.png_path(key)
            node.checks = check_fragment(node.output, node, self.dag)
            node.review = {"passed": True, "issues": [], "model": "store"}
            node.source = "store"
            node.store_key = key
            self.reuse["store_hits"] += 1
            self.cost.record("reuse", node=node_id, phase=phase, label="store_hit")
            record.update({"source": "store", "passed": True, "score": node.checks["score"]})
        else:
            feedback = list(node.feedback)
            rounds = 0
            passed = False
            fragment = ""
            checks: Dict[str, Any] = {}
            review: Optional[Dict[str, Any]] = None
            png: Optional[str] = None
            while True:
                attempt_phase = phase if rounds == 0 else "repair"
                self.cost.phase = attempt_phase
                fragment = self.drafter.draft(node, self.dag, self.deps_outputs(node), feedback, phase=attempt_phase)
                checks = check_fragment(fragment, node, self.dag)
                review = None
                png = None
                if not checks["fatal"]:
                    png = self.render_fragment(node, fragment, f"r{rounds}", attempt_phase)
                    if checks["passed"] and bool(self.cfg.NODE_REVIEW):
                        self.cost.phase = attempt_phase
                        review = self.b.reviewer.node(node, fragment, png, self.dag)
                passed = bool(checks["passed"] and (review is None or review.get("passed")))
                self.emit("node_attempt", node=node_id, round=rounds, checks_score=checks["score"], checks_passed=checks["passed"], review_passed=(review or {}).get("passed"), phase=attempt_phase)
                if passed or rounds >= self.max_node_rounds:
                    break
                feedback = [f"{f['name']}: {f['detail']}" for f in checks.get("failures", [])]
                if review is not None:
                    feedback += [str(i.get("fix_instruction") or i.get("description")) for i in review.get("issues", []) if i.get("fix_instruction") or i.get("description")]
                rounds += 1
            node.output = fragment
            node.png = png
            node.checks = checks
            node.review = review or {"passed": checks["passed"], "issues": issues_from_checks(checks, node_id), "model": "checks"}
            node.source = "draft"
            node.store_key = key
            self.reuse["executed"] += 1
            if passed and self.store is not None:
                self.store.put(key, node, fragment, png, checks, review)
            record.update({"source": "draft", "passed": passed, "score": checks.get("score"), "rounds": rounds})
        node.dirty = False
        node.version += 1
        node.feedback = []
        record["seconds"] = round(time.time() - t0, 3)
        self.records.append(record)
        self.emit("node_done", **record)
        return record

    def execute_drafts(self) -> List[Dict[str, Any]]:
        done = []
        count_skips = self._pending_repair
        self._pending_repair = False
        for nid in self.dag.topological():
            node = self.dag[nid]
            if node.type not in DRAFT_TYPES:
                continue
            if not node.dirty and node.output:
                if count_skips:
                    self.reuse["skipped_clean"] += 1
                continue
            done.append(self.execute_node(nid))
        return done

    def execute_scene(self) -> Dict[str, Any]:
        scene = self.dag[self.dag.scene_id()]
        phase = self.phase_of(scene)
        self.cost.phase = phase
        feedback = list(scene.feedback)
        layout = self.layouter.layout(self.dag, feedback, phase=phase)
        report = check_layout(layout, self.dag)
        rounds = 0
        while not report["passed"] and rounds < self.max_node_rounds:
            feedback = feedback + [f"{f['name']}: {f['detail']}" for f in report.get("failures", [])]
            self.cost.phase = "repair"
            layout = self.layouter.layout(self.dag, feedback, phase="repair")
            report = check_layout(layout, self.dag)
            rounds += 1
        self.cost.phase = phase
        scene.output = layout
        scene.checks = report
        scene.source = "layout"
        self.layout = layout
        self.recompose(phase)
        scene.dirty = False
        scene.version += 1
        scene.feedback = []
        record = {"node": scene.id, "type": SCENE, "phase": phase, "version": scene.version - 1, "rounds": rounds, "passed": report["passed"], "score": report["score"], "source": "layout"}
        self.records.append(record)
        self.emit("scene_done", **record)
        return record

    def recompose(self, phase: Optional[str] = None) -> Dict[str, Any]:
        scene = self.dag[self.dag.scene_id()]
        composed = compose(self.dag, self.layout)
        self.composed = composed
        tag = f"v{scene.version}"
        svg_path = self.run_dir / f"draft_{tag}.svg"
        svg_path.write_text(composed["svg"], encoding="utf-8")
        png_path = self.run_dir / f"draft_{tag}.png"
        self.b.renderer.render(composed["svg"], str(png_path), node=scene.id, phase=phase or self.phase_of(scene))
        overlay_path = self.run_dir / f"overlay_{tag}.png"
        overlay_boxes(str(png_path), str(overlay_path), [{"bbox": item["bbox"], "label": item["node"]} for item in composed["legend"] if item["type"] != BACKGROUND], self.dag.canvas)
        self.draft_svg_path = str(svg_path)
        self.draft_png = str(png_path)
        self.overlay_png = str(overlay_path)
        return composed

    def review_scene(self) -> Dict[str, Any]:
        self.cost.phase = "initial" if not self.scene_reviews else "repair"
        review = self.b.reviewer.scene(self.dag, self.layout, self.composed["svg"], self.draft_png, self.overlay_png)
        review = dict(review)
        review["round"] = len(self.scene_reviews)
        self.scene_reviews.append(review)
        self.emit("scene_review", round=review["round"], passed=review["passed"], issues=len(review.get("issues", [])))
        return review

    def final_params(self) -> Dict[str, Any]:
        final = self.dag[self.dag.final_id()]
        return {
            "seed": int(final.spec.get("seed") or self.cfg.GENERATE_SEED),
            "strength": float(final.spec.get("strength") or self.cfg.GENERATE_STRENGTH),
            "control_scale": float(final.spec.get("control_scale") or self.cfg.GENERATE_CONTROL_SCALE),
            "steps": int(self.cfg.GENERATE_STEPS),
            "guidance": float(self.cfg.GENERATE_GUIDANCE),
        }

    def execute_final(self) -> Dict[str, Any]:
        final = self.dag[self.dag.final_id()]
        phase = self.phase_of(final)
        self.cost.phase = phase
        if not self.draft_png:
            self.recompose(phase)
        params = self.final_params()
        prompt = generation_prompt(self.dag, self.composed["legend"])
        out = self.run_dir / f"final_v{final.version}.png"
        t0 = time.time()
        meta = self.b.generator.generate(self.draft_png, prompt, params, str(out))
        if getattr(self.b.generator, "mock", False):
            self.cost.record("diffusion", node=final.id, seconds=time.time() - t0, phase=phase, label="mock")
        final.output = str(out)
        final.png = str(out)
        final.source = str(meta.get("backend") or "generator")
        self.final_png = str(out)
        self.final_meta = meta
        final.dirty = False
        final.version += 1
        record = {"node": final.id, "type": FINAL, "phase": phase, "version": final.version - 1, "rounds": 0, "passed": True, "score": 1.0, "source": final.source, "params": params}
        self.records.append(record)
        self.emit("final_done", **record)
        return record

    def review_final(self) -> Dict[str, Any]:
        self.cost.phase = "initial" if not self.final_reviews else "repair"
        review = self.b.reviewer.final(self.dag, self.layout, self.composed["svg"], self.draft_png, self.final_png, len(self.final_reviews))
        review = dict(review)
        review["round"] = len(self.final_reviews)
        self.final_reviews.append(review)
        self.emit("final_review", round=review["round"], passed=review["passed"], issues=len(review.get("issues", [])))
        return review

    def default_ops(self, review: Dict[str, Any], blamed: Dict[str, List[Dict[str, Any]]], stage: str) -> List[Op]:
        ops: List[Op] = []
        final_id = self.dag.final_id()
        for nid, issues in blamed.items():
            if nid not in self.dag:
                continue
            major = [i for i in issues if i.get("severity") == "major"] or issues
            if nid == final_id:
                ops.append(AdjustFinal(nid, reason="; ".join(str(i.get("description")) for i in major)[:300]))
                continue
            self.cost.phase = "repair"
            new_spec = self.b.planner.revise_spec(self.dag[nid], major, self.dag)
            feedback = [str(i.get("fix_instruction") or i.get("description") or "") for i in major]
            ops.append(EditSpec(nid, new_spec, feedback=[f for f in feedback if f], reason=f"{stage} review"))
        return ops

    def repair(self, review: Dict[str, Any], stage: str) -> Dict[str, Any]:
        blamed = attribute(review.get("issues", []), self.dag, stage)
        if self.editor is not None:
            ops = self.editor(self, review, blamed, stage)
        else:
            ops = self.default_ops(review, blamed, stage)
        affected: Set[str] = set()
        applied = []
        for op in ops:
            affected |= set(op.apply(self.dag))
            applied.append(op.describe())
        info = {"stage": stage, "blamed": summarize_attribution(blamed), "ops": applied, "affected": sorted(affected), "issues": len(review.get("issues", []))}
        self.repairs.append(info)
        self._pending_repair = True
        self.emit("repair", **info)
        return info

    def run(self) -> Dict[str, Any]:
        self.cost.phase = "initial"
        scene_id = self.dag.scene_id()
        final_id = self.dag.final_id()
        self.scene_rounds = 0
        while True:
            self.execute_drafts()
            if self.dag[scene_id].dirty:
                self.execute_scene()
            review = self.review_scene()
            if review["passed"] or self.scene_rounds >= int(self.cfg.MAX_SCENE_ROUNDS):
                break
            self.repair(review, "scene")
            self.scene_rounds += 1
        self.final_rounds = 0
        while True:
            self.execute_drafts()
            if self.dag[scene_id].dirty:
                self.execute_scene()
            if self.dag[final_id].dirty:
                self.execute_final()
            review = self.review_final()
            if review["passed"] or self.final_rounds >= int(self.cfg.MAX_FINAL_ROUNDS):
                break
            self.repair(review, "final")
            self.final_rounds += 1
        return self.result()

    def structure(self) -> Dict[str, Any]:
        return structure_from_dag(self.dag, self.composed.get("legend", []), self.composed.get("svg", ""))

    def result(self) -> Dict[str, Any]:
        scene_passed = bool(self.scene_reviews and self.scene_reviews[-1]["passed"])
        final_passed = bool(self.final_reviews and self.final_reviews[-1]["passed"])
        return {
            "prompt": self.dag.prompt,
            "canvas": list(self.dag.canvas),
            "dag": self.dag.to_dict(),
            "scene_passed": scene_passed,
            "final_passed": final_passed,
            "scene_rounds": self.scene_rounds,
            "final_rounds": self.final_rounds,
            "draft_svg": self.draft_svg_path,
            "draft_png": self.draft_png,
            "overlay_png": self.overlay_png,
            "final_png": self.final_png,
            "final_meta": self.final_meta,
            "legend": self.composed.get("legend", []),
            "layout": self.layout,
            "scene_reviews": self.scene_reviews,
            "final_reviews": self.final_reviews,
            "repairs": self.repairs,
            "records": self.records,
            "reuse": dict(self.reuse),
            "cost": self.cost.summary(),
            "structure": self.structure(),
        }


def structure_from_dag(dag: DAG, legend: List[Dict[str, Any]], composed_svg: str = "") -> Dict[str, Any]:
    units: List[Dict[str, Any]] = []
    placed_by = {item["node"]: item for item in legend}
    consumed = dag.consumed()
    composed_root = None
    if composed_svg:
        try:
            composed_root = S.parse_svg_text(composed_svg)
        except (ValueError, Exception):
            composed_root = None
    for node in dag.draft_nodes():
        owner = consumed.get(node.id)
        placed = placed_by.get(owner or node.id)
        unit: Dict[str, Any] = {"id": node.id, "type": node.type, "desc": node.desc(), "present": placed is not None, "valid": bool((node.checks or {}).get("passed", True)), "bbox": placed["bbox"] if placed else None, "count": node.count() if node.type in (ENTITY, OBJECT) else 1, "colors": [], "pose": node.spec.get("pose")}
        source = None
        if composed_root is not None:
            group = composed_root.find(f".//g[@data-node='{node.id}']")
            if group is None:
                unit["present"] = False
            else:
                source = S.serialize(group)
                unit["pose"] = group.get("data-pose") or unit["pose"]
        elif isinstance(node.output, str):
            source = node.output
        if owner and owner in dag:
            unit["valid"] = unit["valid"] and bool((dag[owner].checks or {}).get("passed", True))
        if source:
            try:
                root = S.parse_fragment(source, node.id)
                unit["colors"] = [list(c) for c in S.fills(root)]
                inst = len(S.instances(root))
                unit["count"] = inst if inst else 1
                report = check_fragment(source, node, dag)
                unit["valid"] = unit["valid"] and bool(report["passed"])
            except (ValueError, Exception):
                unit["valid"] = False
        units.append(unit)
    return {"units": units, "canvas": list(dag.canvas)}
