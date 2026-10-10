from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .dag import ACTION, BACKGROUND, DAG, ENTITY, FINAL, OBJECT, SCENE

LAYOUT_TYPES = ("position_mismatch", "size_mismatch")
STOP = {"a", "an", "the", "of", "and", "is", "are", "with", "in", "on", "to", "at", "its", "it", "not", "no", "too", "very", "one", "two", "three", "four", "five", "six", "left", "right", "above", "below", "top", "bottom", "object", "objects", "draft", "image", "scene", "canvas", "missing", "wrong", "count", "color", "colour", "should", "be", "has", "have", "instead", "only", "but", "than", "there", "about"}


def _tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z][a-z\-]*", str(text or "").lower())
    out = []
    for t in toks:
        if t in STOP or len(t) < 3:
            continue
        if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            t = t[:-1]
        out.append(t)
    return out


def _node_tokens(dag: DAG, node_id: str) -> set:
    n = dag[node_id]
    toks = set(_tokens(n.desc()))
    toks |= set(_tokens(node_id.replace("_", " ")))
    toks |= set(_tokens(" ".join(n.colors())))
    noun = n.spec.get("noun")
    if noun:
        toks |= set(_tokens(str(noun)))
    for attr in n.spec.get("attributes") or []:
        toks |= set(_tokens(str(attr)))
    return toks


def match_by_text(description: str, dag: DAG, candidates: Optional[List[str]] = None) -> List[Tuple[str, float]]:
    desc_tokens = set(_tokens(description))
    if not desc_tokens:
        return []
    ids = candidates if candidates is not None else [n.id for n in dag.draft_nodes()]
    scored = []
    for nid in ids:
        if nid not in dag:
            continue
        ntoks = _node_tokens(dag, nid)
        if not ntoks:
            continue
        inter = desc_tokens & ntoks
        if not inter:
            continue
        score = len(inter) / (len(ntoks) ** 0.5)
        scored.append((nid, score))
    scored.sort(key=lambda t: -t[1])
    return scored


def _resolve_action(dag: DAG, node_id: str, issue: Dict[str, Any]) -> str:
    consumed = dag.consumed()
    itype = issue.get("type")
    if node_id in consumed and itype in ("pose_mismatch", "position_mismatch", "size_mismatch", "other", "identity_mismatch"):
        return consumed[node_id]
    if node_id in dag and itype in LAYOUT_TYPES and dag[node_id].type != ACTION:
        return dag.scene_id() or node_id
    return node_id


def attribute(issues: List[Dict[str, Any]], dag: DAG, stage: str = "scene") -> Dict[str, List[Dict[str, Any]]]:
    scene_id = dag.scene_id() or "scene"
    final_id = dag.final_id() or "final"
    blamed: Dict[str, List[Dict[str, Any]]] = {}
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        itype = str(issue.get("type") or "other")
        targets: List[str] = []
        if itype == "draft_drift" or stage == "final" and itype == "other" and not issue.get("nodes"):
            targets = [final_id]
        else:
            named = [str(n) for n in (issue.get("nodes") or [])]
            for n in named:
                key = n.strip()
                if key in dag:
                    targets.append(_resolve_action(dag, key, issue))
                elif key.lower() in ("scene", "layout", "composition"):
                    targets.append(scene_id)
                elif key.lower() in ("final", "generation", "generate", "diffusion"):
                    targets.append(final_id)
                else:
                    lowered = {nid.lower(): nid for nid in dag.nodes}
                    if key.lower() in lowered:
                        targets.append(_resolve_action(dag, lowered[key.lower()], issue))
            if not targets:
                matches = match_by_text(str(issue.get("description") or "") + " " + str(issue.get("fix_instruction") or ""), dag)
                if itype in LAYOUT_TYPES:
                    top = _resolve_action(dag, matches[0][0], issue) if matches else scene_id
                    targets.append(top if top in dag and dag[top].type == ACTION else scene_id)
                elif matches and (len(matches) == 1 or matches[0][1] >= 1.5 * matches[1][1] or matches[0][1] >= 1.0):
                    targets.append(_resolve_action(dag, matches[0][0], issue))
                elif matches and itype in ("count_mismatch", "color_mismatch", "missing_object", "pose_mismatch", "identity_mismatch"):
                    targets.append(_resolve_action(dag, matches[0][0], issue))
            if not targets:
                targets.append(scene_id)
        for t in dict.fromkeys(targets):
            item = dict(issue)
            item["attributed_to"] = t
            blamed.setdefault(t, []).append(item)
    return blamed


def summarize_attribution(blamed: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for nid, items in blamed.items():
        out.append({"node": nid, "issues": len(items), "types": sorted({i.get("type", "other") for i in items})})
    return out
