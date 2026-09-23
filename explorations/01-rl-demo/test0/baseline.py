from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import svg as S
from .attribution import attribute
from .compose import legend_text
from .dag import DAG, Node, OBJECT, SCENE
from .generate import generation_prompt
from .llm import parse_json
from .prompts import BASELINE_DRAFT_SYSTEM, BASELINE_REVIEW_SYSTEM, BASELINE_REVISE_SYSTEM, FINAL_REVIEW_SYSTEM
from .render import overlay_boxes
from .review import normalize_review
from .scheduler import structure_from_dag

_PLANNING = "===PLANNING_START==="
_LEGEND = "===LEGEND_START==="
_SVG = "===SVG_CODE_START==="


def parse_three_parts(raw: str) -> Dict[str, Any]:
    text = str(raw or "")
    legend_json: Any = None
    svg_code = text
    if _SVG in text:
        before, svg_code = text.split(_SVG, 1)
        if _LEGEND in before:
            legend_part = before.split(_LEGEND, 1)[1]
            try:
                legend_json = parse_json(legend_part)
            except ValueError:
                legend_json = None
    svg_code = S.extract_code(svg_code)
    svg_code = re.sub(r"<text[^>]*>.*?</text>", "", svg_code, flags=re.DOTALL | re.IGNORECASE)
    return {"svg": svg_code, "legend": legend_json if isinstance(legend_json, dict) else {}}


def elements_from_legend(legend: Dict[str, Any], canvas: Sequence[int]) -> List[Dict[str, Any]]:
    out = []
    comp = legend.get("compositional_deconstruction") if isinstance(legend, dict) else None
    raw = comp.get("elements") if isinstance(comp, dict) else None
    if not isinstance(raw, list):
        return out
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        try:
            b = [float(v) for v in bbox] if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else [0.0, 0.0, float(canvas[0]), float(canvas[1])]
        except (TypeError, ValueError):
            b = [0.0, 0.0, float(canvas[0]), float(canvas[1])]
        out.append({"id": f"e{i + 1}", "type": str(item.get("type") or "obj"), "bbox": b, "desc": str(item.get("desc") or ""), "node": None})
    return out


def element_dag(prompt: str, canvas: Sequence[int], elements: List[Dict[str, Any]]) -> DAG:
    dag = DAG(prompt, (int(canvas[0]), int(canvas[1])))
    for el in elements:
        colors = [t for t in re.findall(r"[a-z]+", el["desc"].lower()) if S.normalize_color_name(t) in S.NAMED_COLORS]
        dag.add(Node(id=el["id"], type=OBJECT, spec={"desc": el["desc"], "count": 1, "colors": colors[:2]}, deps=[]))
    dag.add(Node(id="scene", type=SCENE, spec={"desc": prompt, "relations": []}, deps=[el["id"] for el in elements]))
    dag.normalize()
    return dag


class Baseline:
    def __init__(self, cfg: Any, backends: Any):
        self.cfg = cfg
        self.b = backends
        self.cost = backends.cost

    def _draft(self, prompt: str, feedback: Sequence[str], hint: Optional[Dict[str, Any]], phase: str) -> Dict[str, Any]:
        self.cost.phase = phase
        if self.b.monolithic is not None:
            out = self.b.monolithic.draft(prompt, feedback, hint)
            return {"svg": out["svg"], "elements": out["elements"], "canvas": out["canvas"], "dag": out["dag"]}
        if feedback:
            user = f"Original prompt: {prompt}\n\nPrevious SVG code:\n{self._last_svg[:12000]}\n\nReview feedback:\n" + "\n".join(f"- {f}" for f in feedback) + "\n\nGenerate the corrected three parts."
            text = self.b.llm.chat("executor", BASELINE_REVISE_SYSTEM, user, node="baseline", phase=phase, temperature=0.7)
        else:
            text = self.b.llm.chat("executor", BASELINE_DRAFT_SYSTEM, f"Generate SVG code for:\n{prompt}", node="baseline", phase=phase, temperature=0.7)
        parts = parse_three_parts(text)
        svg = parts["svg"]
        if "<svg" not in svg.lower():
            svg = S.svg_document(1024, 1024, svg)
        m = re.search(r'viewBox\s*=\s*["\']\s*[-0-9.]+\s+[-0-9.]+\s+([0-9.]+)\s+([0-9.]+)', svg)
        canvas = [int(float(m.group(1))), int(float(m.group(2)))] if m else [1024, 1024]
        elements = elements_from_legend(parts["legend"], canvas)
        return {"svg": svg, "elements": elements, "canvas": canvas, "dag": None}

    def _review(self, prompt: str, state: Dict[str, Any], png: str, overlay: str, phase: str) -> Dict[str, Any]:
        self.cost.phase = phase
        if self.b.monolithic is not None and state.get("dag") is not None:
            dag = state["dag"]
            review = self.b.reviewer.scene(dag, dag[dag.scene_id()].output, state["svg"], png, overlay)
            return self._map_nodes(review, state)
        legend = [{"node": el["id"], "type": "element", "desc": el["desc"], "count": 1, "bbox": el["bbox"], "participants": []} for el in state["elements"]]
        user = f"USER PROMPT: {prompt}\nCANVAS: {state['canvas'][0]}x{state['canvas'][1]}\n\nLEGEND (element ids to use for attribution; use \"scene\" for global layout problems):\n{legend_text(legend)}\n\nImage 1 is the clean draft, image 2 is the draft with element ids overlaid. Return the JSON verdict."
        text = self.b.llm.chat("review", BASELINE_REVIEW_SYSTEM, user, images=[png, overlay], json_mode=True, node="baseline")
        try:
            data = parse_json(text)
        except ValueError:
            data = None
        return normalize_review(data, ["scene"], self.cfg.role_model("review"))

    def _map_nodes(self, review: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        node_to_el = {el["node"]: el["id"] for el in state["elements"] if el.get("node")}
        dag = state.get("dag")
        consumed = dag.consumed() if dag is not None else {}
        mapped = dict(review)
        issues = []
        for issue in review.get("issues", []):
            item = dict(issue)
            nodes = []
            for n in item.get("nodes") or []:
                owner = consumed.get(n, n)
                if owner in node_to_el:
                    nodes.append(node_to_el[owner])
                elif n == (dag.scene_id() if dag is not None else "scene"):
                    nodes.append("scene")
                elif n == "final":
                    nodes.append("final")
            item["nodes"] = nodes
            issues.append(item)
        mapped["issues"] = issues
        return mapped

    def _final_review(self, prompt: str, state: Dict[str, Any], draft_png: str, final_png: str, round_index: int, phase: str) -> Dict[str, Any]:
        self.cost.phase = phase
        if self.b.monolithic is not None and state.get("dag") is not None:
            dag = state["dag"]
            review = self.b.reviewer.final(dag, dag[dag.scene_id()].output, state["svg"], draft_png, final_png, round_index)
            return self._map_nodes(review, state)
        legend = [{"node": el["id"], "type": "element", "desc": el["desc"], "count": 1, "bbox": el["bbox"], "participants": []} for el in state["elements"]]
        user = f"USER PROMPT: {prompt}\n\nLEGEND of the structural draft (element ids):\n{legend_text(legend)}\n\nImage 1 is the structural draft, image 2 is the final generated image. Return the JSON verdict."
        text = self.b.llm.chat("review", FINAL_REVIEW_SYSTEM, user, images=[draft_png, final_png], json_mode=True, node="final")
        try:
            data = parse_json(text)
        except ValueError:
            data = None
        return normalize_review(data, ["final"], self.cfg.role_model("review"))

    def _render(self, state: Dict[str, Any], run_dir: Path, tag: str, phase: str) -> None:
        self.cost.phase = phase
        svg_path = run_dir / f"draft_{tag}.svg"
        svg_path.write_text(state["svg"], encoding="utf-8")
        png_path = run_dir / f"draft_{tag}.png"
        self.b.renderer.render(state["svg"], str(png_path), node="baseline", phase=phase)
        overlay = run_dir / f"overlay_{tag}.png"
        overlay_boxes(str(png_path), str(overlay), [{"bbox": el["bbox"], "label": el["id"]} for el in state["elements"] if el["type"] != "background"], tuple(state["canvas"]))
        state["svg_path"] = str(svg_path)
        state["png"] = str(png_path)
        state["overlay"] = str(overlay)

    def run(self, prompt: str, run_dir: str, hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self._last_svg = ""
        scene_reviews: List[Dict[str, Any]] = []
        final_reviews: List[Dict[str, Any]] = []
        repairs: List[Dict[str, Any]] = []
        records: List[Dict[str, Any]] = []
        feedback: List[str] = []
        state: Dict[str, Any] = {}
        scene_round = 0
        while True:
            phase = "initial" if scene_round == 0 else "repair"
            state = self._draft(prompt, feedback, hint, phase)
            self._last_svg = state["svg"]
            self._render(state, run_path, f"v{scene_round}", phase)
            records.append({"node": "baseline", "type": "monolithic", "phase": phase, "version": scene_round, "passed": None})
            review = self._review(prompt, state, state["png"], state["overlay"], phase)
            review["round"] = scene_round
            scene_reviews.append(review)
            if review["passed"] or scene_round >= int(self.cfg.MAX_SCENE_ROUNDS):
                break
            feedback = [str(i.get("fix_instruction") or i.get("description")) for i in review.get("issues", []) if i.get("fix_instruction") or i.get("description")]
            repairs.append({"stage": "scene", "ops": [{"op": "regenerate_all"}], "affected": [el["id"] for el in state["elements"]], "issues": len(review.get("issues", []))})
            scene_round += 1
        params = {"seed": int(self.cfg.GENERATE_SEED), "strength": float(self.cfg.GENERATE_STRENGTH), "control_scale": float(self.cfg.GENERATE_CONTROL_SCALE), "steps": int(self.cfg.GENERATE_STEPS), "guidance": float(self.cfg.GENERATE_GUIDANCE)}
        legend = [{"node": el["id"], "type": el["type"], "desc": el["desc"], "count": 1, "bbox": el["bbox"], "z": i, "participants": []} for i, el in enumerate(state["elements"])]
        pseudo = element_dag(prompt, state["canvas"], state["elements"])
        gen_prompt = generation_prompt(pseudo, legend)
        final_round = 0
        final_png = None
        final_meta: Dict[str, Any] = {}
        while True:
            phase = "initial" if final_round == 0 else "repair"
            self.cost.phase = phase
            out = run_path / f"final_v{final_round}.png"
            t0 = time.time()
            final_meta = self.b.generator.generate(state["png"], gen_prompt, params, str(out))
            if getattr(self.b.generator, "mock", False):
                self.cost.record("diffusion", node="final", seconds=time.time() - t0, phase=phase, label="mock")
            final_png = str(out)
            review = self._final_review(prompt, state, state["png"], final_png, final_round, phase)
            review["round"] = final_round
            final_reviews.append(review)
            if review["passed"] or final_round >= int(self.cfg.MAX_FINAL_ROUNDS):
                break
            params["seed"] += 1
            params["control_scale"] = min(1.0, params["control_scale"] + 0.1)
            params["strength"] = min(0.95, params["strength"] + 0.1)
            repairs.append({"stage": "final", "ops": [{"op": "regenerate_final"}], "affected": ["final"], "issues": len(review.get("issues", []))})
            final_round += 1
        if state.get("dag") is not None:
            dag = state["dag"]
            structure = structure_from_dag(dag, [{"node": el["node"], "type": el["type"], "desc": el["desc"], "bbox": el["bbox"]} for el in state["elements"]], state["svg"])
            for unit in structure["units"]:
                unit["node"] = unit["id"]
        else:
            structure = {"units": [{"id": el["id"], "type": el["type"], "desc": el["desc"], "present": True, "valid": True, "bbox": el["bbox"], "count": 1, "colors": [], "pose": None} for el in state["elements"]], "canvas": list(state["canvas"])}
        result = {
            "prompt": prompt,
            "system": "baseline",
            "canvas": list(state["canvas"]),
            "elements": state["elements"],
            "scene_passed": bool(scene_reviews and scene_reviews[-1]["passed"]),
            "final_passed": bool(final_reviews and final_reviews[-1]["passed"]),
            "scene_rounds": scene_round,
            "final_rounds": final_round,
            "draft_svg": state.get("svg_path"),
            "draft_png": state.get("png"),
            "overlay_png": state.get("overlay"),
            "final_png": final_png,
            "final_meta": final_meta,
            "legend": legend,
            "scene_reviews": scene_reviews,
            "final_reviews": final_reviews,
            "repairs": repairs,
            "records": records,
            "reuse": {"demanded": len(state["elements"]) * (scene_round + 1), "store_hits": 0, "executed": len(state["elements"]) * (scene_round + 1), "skipped_clean": 0},
            "cost": self.cost.summary(),
            "structure": structure,
            "run_dir": str(run_path),
            "seconds": round(time.time() - started, 3),
            "status": "completed",
            "state_svg": state["svg"],
            "state_dag": state["dag"].to_dict() if state.get("dag") is not None else None,
        }
        (run_path / "result.json").write_text(json.dumps({k: v for k, v in result.items() if k not in ("state_dag",)}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return result

    def review_svg(self, prompt: str, result: Dict[str, Any], svg: str, run_dir: Path, tag: str) -> Dict[str, Any]:
        state = {"svg": svg, "elements": result["elements"], "canvas": result["canvas"], "dag": None}
        if result.get("state_dag") is not None and self.b.monolithic is not None:
            state["dag"] = DAG.from_dict(result["state_dag"])
        self._render(state, run_dir, tag, "repair")
        return self._review(prompt, state, state["png"], state["overlay"], "repair")
