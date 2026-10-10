"""Duplicate nodes in a finished plan, removed without any model.

Two kinds, both found on 10-10 in the v84 plans (the coverage passes v5/v6 added nodes the plan already had, and the
one-fact-per-node pass kept both twins when they came back in different wording):
  * twin child nodes: an attribute, state, relation, action or absent node that asks, about the same participants, the
    same question as an earlier node (same words in the same order once articles, auxiliaries and inflection are
    folded: "Is the boy sitting on the floor?" = "Does the boy sit on the floor?"; "on" and "under", or the order of two
    participants, keep them apart). The later one goes, nothing else changes.
  * twin entity nodes: two entity nodes for one object of the description ("kiosk" and "kiosk_entity"). Which pairs
    are one object is decided by reading the description (configs/dedupe_merges_*.json, every group with its reason):
    this module only carries out a listed merge, and only when nothing is lost: the dropped node's question words are
    all asked by the kept node or by the nodes about the kept entity (sound/motion cues, hedges and frame positions
    excepted, as in planner._deletable). Which twin is kept: the one that covers the other; when both do, the earlier
    one in the plan (its box is the first planner's). The dropped entity's children and edges move to the kept one.
Why it matters: schema.add_generated_nodes turns two count-1 entities of one class into a count node that wants two
objects and a pair node that wants two different individuals, so a twin entity makes the loop paint a second copy."""
import json
import re

from redag.planner import LOST_FREE, MOD_FREE, NONVISUAL_WORD, SPLIT_FREE, _content, _cstem, _frame_sub, _of_ids, _placed_on_other, _qnames, _words
from redag.schema import MULTI_TYPES, Plan

PARA_DROP = {"a", "an", "the", "is", "are", "does", "do", "there", "any"}
CHILD = ("attribute", "state", "relation", "action", "absent")


def para_key(q):
    """The question as a tuple of stems, articles and question auxiliaries left out, word order kept."""
    t = re.sub(r"[’']s\b", "", str(q or "").lower())
    return tuple(_cstem(w) for w in _words(t) if w not in PARA_DROP)


def _repoint(d, drop, keep):
    """Every reference to entity drop now names keep: of lists (a relation or action left with one participant becomes a
    state of it, or a relation with the entity its question places it against, as planner.lift_states_to_relations),
    edges (self loops and twins removed). Returns the ids of the nodes whose of changed."""
    changed = []
    d["nodes"] = [n for n in d["nodes"] if n["id"] != drop]
    for n in d["nodes"]:
        of = _of_ids(n)
        if drop not in of:
            continue
        new = list(dict.fromkeys(keep if u == drop else u for u in of))
        if n.get("type") in MULTI_TYPES and len(new) == 1:  # both participants were the twins: a state of the kept one,
            n["type"], n["of"] = "state", new[0]  # or a relation with the entity its question places it against
            hit = _placed_on_other(d, n)
            if hit and hit[0] != new[0]:
                n["type"], n["of"] = "relation", [new[0], hit[0]]
        elif isinstance(n.get("of"), list):
            n["of"] = new
        else:
            n["of"] = new[0]
        changed.append(n["id"])
    edges, seen = [], set()
    for e in d.get("edges", []):
        a = keep if e.get("from") == drop else e.get("from")
        b = keep if e.get("to") == drop else e.get("to")
        if a == b:
            continue
        k = e.get("kind", "support")
        key = (k,) + (tuple(sorted((a, b))) if k == "interfere" else (a, b))
        if key in seen:
            continue
        seen.add(key)
        edges.append(dict(e, **{"from": a, "to": b}))
    d["edges"] = edges
    return changed


def lost_by_merge(d, drop, keep):
    """Content words of drop's question and class that keep's own question, class and plural do not carry. Empty means
    the merge loses nothing. Children of the twins do not count: they only say "the cat ..." and would cover the kind
    word of a wrongly listed pair (a cat merged into a boy)."""
    text, _ = _frame_sub(drop.get("question", ""))
    words = (_content(text, SPLIT_FREE) | _content(f"{drop.get('class', '')} {drop.get('plural', '')}")) - LOST_FREE - MOD_FREE - _qnames(d["nodes"], [keep["id"]])
    return sorted(w for w in words if not NONVISUAL_WORD.fullmatch(w))


def merge_entities(d, groups):
    """groups: lists of entity ids that name one object of the description. Returns (merges done, merges refused)."""
    done, refused = [], []
    for g in groups:
        order = [n["id"] for n in d["nodes"]]
        byid = {n["id"]: n for n in d["nodes"]}
        ids = sorted([i for i in g if i in byid], key=order.index)
        if len(ids) < 2 or any(byid[i].get("type") != "entity" for i in ids):
            refused.append(dict(group=list(g), why="fewer than two of them are entity nodes of the plan"))
            continue
        anchor = ids[0]
        for other in ids[1:]:
            if not lost_by_merge(d, byid[other], byid[anchor]):
                keep, drop = anchor, other
            elif not lost_by_merge(d, byid[anchor], byid[other]):
                keep, drop = other, anchor
            else:
                refused.append(dict(group=list(g), ids=[anchor, other], why="would lose " + json.dumps(lost_by_merge(d, byid[other], byid[anchor]))))
                continue
            if keep == other:  # the later twin's words are kept, the earlier twin's box: it is the first planner's layout
                byid[keep]["box"] = byid[anchor].get("box")
            moved = _repoint(d, drop, keep)
            byid = {n["id"]: n for n in d["nodes"]}
            done.append(dict(drop=drop, keep=keep, question=byid[keep].get("question"), moved=moved))
            anchor = keep
    return done, refused


def dedupe_children(d):
    """Drop every child node that repeats an earlier one about the same participants. Returns the dropped records."""
    seen, dropped, out = {}, [], []
    for n in d["nodes"]:
        if n.get("type") in CHILD:
            key = (tuple(sorted(_of_ids(n))), para_key(n.get("question", "")))
            if key[1] and key in seen:
                dropped.append(dict(id=n["id"], twin=seen[key], question=n.get("question")))
                continue
            seen.setdefault(key, n["id"])
        out.append(n)
    gone = {x["id"] for x in dropped}
    d["nodes"] = out
    d["edges"] = [e for e in d.get("edges", []) if e.get("from") not in gone and e.get("to") not in gone]
    return dropped


def dedupe_plan(d, groups=(), max_nodes=128, min_side=48):
    """A copy of plan dict d with its twins removed, and a record of what was done. d itself is never changed; it is
    returned (and the record says why) when nothing was found or the result does not validate."""
    out = json.loads(json.dumps(d))
    qs = {n["id"]: n.get("question") for n in d["nodes"]}
    merged, refused = merge_entities(out, groups or [])
    for m in merged:
        m["dropped_question"] = qs.get(m["drop"])
    dropped = dedupe_children(out)
    info = dict(merged=merged, refused=refused, dropped=dropped, n_before=len(d["nodes"]), n_after=len(out["nodes"]), error=None)
    if not merged and not dropped:
        return d, info
    errs, _w = Plan(out).validate(min_side=min_side, max_nodes=max_nodes)
    if errs:
        info["error"] = errs[:5]
        info["n_after"] = len(d["nodes"])
        return d, info
    gone = {x["id"] for x in dropped} | {x["drop"] for x in merged}
    left = {n["id"] for n in out["nodes"]}
    assert not (gone & left) and len(left) == len(d["nodes"]) - len(gone)
    return out, info
