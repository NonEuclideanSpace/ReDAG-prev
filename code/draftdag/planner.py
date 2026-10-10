import json
import os
import re
import time

import requests

from .paths import P0
from .schema import Plan, add_generated_nodes

LEDGER = os.path.join(P0, "runs", "api_ledger.jsonl")
SECRETS = os.environ.get("P0_SECRETS", os.path.join(P0, "secrets.env"))
API_URL = os.environ.get("PLANNER_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
DEFAULT_MODEL = os.environ.get("PLANNER_MODEL", "qwen3.7-plus")

SYSTEM = """You decompose a scene description into a graph of checkable requirements for an image of {W}x{H} pixels (x to the right, y downwards, origin at the top-left corner). The image is painted in one pass by a text-to-image model; a vision-language checker then answers one yes/no question per requirement; requirements that fail are corrected by an image editing model one at a time. Your graph decides what is checked, where, and in which order, so every node must be a single checkable fact.

Return one JSON object with this schema:
{{
 "style": "<short style phrase>",
 "nodes": [
  {{"id": "bg", "type": "background", "desc": "...", "question": "...", "box": [0, 0, {W}, {H}]}},
  {{"id": "<snake_case>", "type": "entity", "class": "<kind>", "plural": "<kinds>", "count": 1, "count_exact": true, "desc": "...", "question": "...", "box": [x0, y0, x1, y1]}},
  {{"id": "<snake_case>", "type": "attribute", "of": "<entity id>", "desc": "...", "question": "..."}},
  {{"id": "<snake_case>", "type": "state", "of": "<entity id>", "desc": "...", "question": "..."}},
  {{"id": "<snake_case>", "type": "action", "of": ["<entity id>", "<entity id>"], "desc": "...", "question": "..."}},
  {{"id": "<snake_case>", "type": "relation", "of": ["<entity id>", "<entity id>"], "desc": "...", "question": "..."}},
  {{"id": "<snake_case>", "type": "absent", "desc": "...", "question": "..."}}
 ],
 "edges": [
  {{"from": "<entity id>", "to": "<entity id>", "kind": "support"}}
 ]
}}

Rules for nodes
1. One node says one thing, and every stated fact appears in exactly one node. An entity node states only that an object is present, with the minimum intrinsic look needed to tell it apart from the other entities: kind, color, clothing, material, size ("a woman in a floral dress", "a black cat", "a second judge in a formal suit"). The entity's desc and question never contain a verb of action, pose or condition (pointing, kneeling, lying, standing, erupting, bubbling, piled high, peeking out, being handed), never a place (no "on the table", "under the car", "near the window", "behind the counter", "next to", "held by", and no "on the left", "in the center", "in the foreground", "in the background"), and never another entity. What it holds, what it does, where it is and how it relates to others go into separate attribute, state, action or relation nodes; its stated position goes into the box. A look used in the entity node is not repeated as an attribute node. A verb with a single participant is a state ("the participant wipes his forehead"); action and relation always have two or more participants.
2. Every person, animal or object that stands in the scene on its own (on a surface, on the floor, in the air) and that the description singles out gets its own entity node. An object that is only held, worn, carried or handed over by someone (a teacup in her hand, a wrench he holds, a bottle being handed over) is not an entity node: it belongs to that person's attribute node, or to the action node that moves it. Items that the description treats as one group ("three red tomatoes", "a stack of tires", "several beakers", "dirty plates") are one entity node: "count" is the stated number and "count_exact" is true; when no number is stated, set "count" to 2 and "count_exact" to false. "class" is the kind a viewer would use to count such things, in one to three lowercase words without colors ("researcher", "cat", "tomato", "sedan", not "silver sedan"); family and social roles that cannot be seen (grandmother, uncle, cousin, colleague, friend) are replaced by what is seen (elderly woman, man, child, researcher); entities of the same kind share exactly the same class. Classes never overlap: if one entity is class "child", then every boy, girl, toddler and baby in the plan is also class "child" (their desc tells them apart), and if one is class "man", an elderly man or a businessman is also "man" unless the description identifies him by an occupation visible in his clothing (pilot, nurse, firefighter); "plural" is the plural of the class.
3. "attribute" is a lasting property of one entity (a clothing detail, a color, an object it holds or wears, a material). "state" is a pose, orientation, expression or condition of one entity that can be judged by looking at that entity alone (facing left, bubbling, lit, open, sitting, crawling, smiling, sleeping, looking up). Neither mentions another entity node: "the cat sits on the stack of tires" is a state "the cat is sitting" plus a relation cat-tires; "the mechanic lies on the creeper under the sedan" is a state "lying down" plus relations mechanic-creeper and mechanic-sedan. Both have "of" set to the one entity id they describe, never "bg". Scene-wide lighting, rendering technique, brushwork, color palette, composition devices, mood and atmosphere are not nodes: put them in "style".
4. "action" and "relation" describe something between two or more entities: an interaction (handing over, pointing at, looking at, watching, pushing, talking to) or a placement relative to another entity (on, under, next to, beside, behind, in front of, near, toward, leaning on). Every placement the description states between two entities is a relation node, including ones stated in passing at the start of a sentence: "Beside her, a boy ..." (boy next to girl), "To his left, a workbench ..." (workbench to the left of the carpenter), "Behind them, a volcano ..." (volcano behind the judges), "Next to the blacksmith, a boy ...", "Nearby, a winch ..." (winch near the entity just mentioned), "gathered at her feet" (chickens at the girl's feet), "a cooler in front of the umbrella". Resolve every such pronoun (her, his, them, it) to the entity it refers to and write one relation node per placement; only "on the left", "in the center", "in the foreground" and similar phrases that place something in the image frame are not relations. Anything that mentions a second entity belongs here, not in a state. "of" lists all participating entity ids, two or more, and every id in "of" must be an entity node of this plan: an object that is pushed, ridden, sat on, perched on or stood next to (a cart, a bicycle, a sofa, a railing, a tool chest) is therefore an entity node, never part of the background and never folded into an attribute. Positions relative to the image frame (center, left, foreground) are not relations; they go into the box only.
5. "absent" is something the description says must NOT appear. Its question is answered "yes" when the thing IS visible (for example "Are there any bananas in the image?").
6. "desc" is a noun phrase saying what must be visible for this node, exactly as the description states it. Fixed scenery that nothing refers to (walls, windows, curtains, floor, banners, shelving, lighting fixtures) is described in the background node; scenery becomes an entity when another entity is placed on, in or next to it, or when the description gives it a detail of its own.
7. "question" is one yes/no question that is answered "yes" exactly when this node is satisfied, so it asks about one thing only. Ask only what the description states. Never add details the description does not give: no invented colors, tools, sides, positions, poses or counts. Refer to entities by their stated look, never by ids and never by words like "this" or "it"; two entities that look alike are "one <kind>" and "a second <kind>". Positions relative to the image frame (left, right, center, foreground, background) never appear in any question; they go into the box. No pixel coordinates.
8. "box" is given for the background and for entities only: the pixel rectangle [x0, y0, x1, y1] where the entity should be in a sensible layout; respect stated positions, perspective and relative sizes (background things higher and smaller, foreground things lower and larger); every side at least 48 px; boxes of different entities should overlap little. The box is only a layout suggestion and is never used to judge correctness.
9. Cover every requirement of the description and do not add any requirement of your own. Requirements include every stated pose, expression and condition of an entity, even an ordinary one (a man who "stands" gets a state node "standing", a woman who "sits" gets "sitting"; a squawking parrot: beak open; a whistling kettle: steam from the spout; a crackling fire: flames; an idle winch; crouching, swaying, fluttering, running), every stated property of an object that is not already its identifying look (golden lemonade, a tool belt filled with tools, a tangled rope), every stated placement between two entities, every stated count ("two sailors", "three robots in the background"), and every stated absence ("no clutter", "without a hat", "there is no ...") as an absent node. Only art style, lighting, shadows, reflections, textures, mood and frame positions are left out.

Rules for edges
10. Add a "support" edge from an entity to another entity that it physically carries (a beaker standing on the bench: bench -> beaker; a cat lying on the sofa: sofa -> cat), only between two entity nodes. No edges from "bg" and no other kind of edge.

Write everything in English. Output only the JSON object."""

EXAMPLE_USER = "Scene description:\nIn a chemistry lab with reagent cabinets along the back wall, a smiling researcher wearing goggles stands behind the bench holding a test tube rack. Beside her, a colleague hands her a bottle of solution. Several beakers stand on the bench; one of them is heated over an alcohol lamp. There are no windows."

EXAMPLE_ASSISTANT = json.dumps({
    "style": "photorealistic photo",
    "nodes": [
        {"id": "bg", "type": "background", "desc": "a chemistry laboratory with reagent cabinets along the back wall", "question": "Does the image show a chemistry laboratory with reagent cabinets along the back wall?", "box": [0, 0, 1024, 1024]},
        {"id": "bench", "type": "entity", "class": "workbench", "plural": "workbenches", "count": 1, "count_exact": True, "desc": "a laboratory workbench", "question": "Is there a laboratory workbench in the image?", "box": [0, 560, 1024, 900]},
        {"id": "researcher", "type": "entity", "class": "researcher", "plural": "researchers", "count": 1, "count_exact": True, "desc": "a researcher wearing goggles", "question": "Is there a researcher wearing goggles?", "box": [430, 160, 700, 760]},
        {"id": "colleague", "type": "entity", "class": "researcher", "plural": "researchers", "count": 1, "count_exact": True, "desc": "a second researcher", "question": "Is there a second researcher?", "box": [60, 180, 330, 780]},
        {"id": "beakers", "type": "entity", "class": "beaker", "plural": "beakers", "count": 2, "count_exact": False, "desc": "several glass beakers", "question": "Are there several glass beakers?", "box": [720, 600, 960, 780]},
        {"id": "lamp", "type": "entity", "class": "alcohol lamp", "plural": "alcohol lamps", "count": 1, "count_exact": True, "desc": "an alcohol lamp", "question": "Is there an alcohol lamp?", "box": [780, 760, 900, 880]},
        {"id": "researcher_smiling", "type": "state", "of": "researcher", "desc": "the researcher wearing goggles is smiling", "question": "Is the researcher wearing goggles smiling?"},
        {"id": "researcher_rack", "type": "attribute", "of": "researcher", "desc": "the researcher wearing goggles holds a test tube rack", "question": "Is the researcher wearing goggles holding a test tube rack?"},
        {"id": "colleague_bottle", "type": "attribute", "of": "colleague", "desc": "the second researcher holds a bottle of solution", "question": "Is the second researcher holding a bottle?"},
        {"id": "researcher_behind_bench", "type": "relation", "of": ["researcher", "bench"], "desc": "the researcher wearing goggles stands behind the bench", "question": "Is the researcher wearing goggles standing behind the workbench?"},
        {"id": "colleague_beside_researcher", "type": "relation", "of": ["colleague", "researcher"], "desc": "the second researcher is beside the researcher wearing goggles", "question": "Is the second researcher beside the researcher wearing goggles?"},
        {"id": "handover", "type": "action", "of": ["colleague", "researcher"], "desc": "the second researcher hands the bottle to the researcher wearing goggles", "question": "Is the second researcher handing a bottle to the researcher wearing goggles?"},
        {"id": "beakers_on_bench", "type": "relation", "of": ["beakers", "bench"], "desc": "the beakers stand on the bench", "question": "Are the beakers standing on the workbench?"},
        {"id": "beaker_heated", "type": "relation", "of": ["beakers", "lamp"], "desc": "one beaker is heated over the alcohol lamp", "question": "Is one of the beakers positioned over the alcohol lamp, being heated?"},
        {"id": "no_windows", "type": "absent", "desc": "windows", "question": "Are there any windows in the image?"}
    ],
    "edges": [
        {"from": "bench", "to": "beakers", "kind": "support"},
        {"from": "bench", "to": "lamp", "kind": "support"}
    ]
}, ensure_ascii=False)

AUDIT_BEYOND = """Scene description:
{prompt}

Below are yes/no questions that were derived from the description, each with its node type in brackets. For each question decide two things.
beyond: does it ask for anything the description does not state (an invented color, object, tool, side, position, count, pose or action)? Being more specific than the description counts as beyond it.
multi: does the question bundle two separately checkable facts, so that an image could satisfy one of them and not the other? Judge by node type. An entity question names the object with its identifying look (kind, color, clothing, material, size, what it contains or displays); that look is not a separate fact, but the question is multi if it also asks about a pose, an action, a condition, a placement or a second object. An attribute or state question necessarily names its entity by its identifying look ("the woman in a silver dress"), which is not multi; it is multi only if it asks about two different properties, poses or expressions of that entity. An action or relation question necessarily names its participants, which is not multi; it is multi only if it asks about two different interactions or placements (for example pointing at the screen while talking to someone). A manner stated in the description (firmly, intently, slightly) is not a separate fact.
Reply with one JSON object and nothing else: {{"<id>": {{"beyond": true or false, "multi": true or false, "reason": "<short reason or empty>"}}, ...}}

Questions:
{questions}"""

AUDIT_FACTS = """Scene description:
{prompt}

Split this description into its concrete visible facts, one fact per entry, as short phrases. Keep only facts of these kinds: the setting; each person, animal or object; a property, pose or expression of one of them (smiling, beak open, sitting, wearing boots); an interaction between two of them; a placement of one object relative to another object (beside, behind, in front of, on, under, near); how many of a kind; a thing that must not appear. Skip everything else, in particular: art style and rendering technique; time of day and weather light; sunlight, light beams, shadows, reflections, glow, lens flares, dust motes, sparkles, speed or motion lines; focus, blur, sharpness, depth of field; textures, materials and finishes of surfaces (rough bark, fabric folds, metallic sheen, polished floor, wood grain, veneer); what an expression or scene conveys or suggests; mood, atmosphere, composition and narrative commentary; sounds, smells, temperature and taste; anything that cannot be judged from one still image, such as change over time (ticking in reverse, pulsing, flickering, vanishing, slowly accumulating), what something was or will be, similes and "as if" descriptions; and where something sits relative to the image frame (left, right, center, foreground, background). Never turn a frame position into a placement between two objects: "On the right, a cat" is a frame position and is skipped, while "to his left", "beside her", "behind them", "in front of the umbrella" relate two objects and are kept. Reply with one JSON object and nothing else: {{"facts": ["<fact>", ...]}}"""

AUDIT_MATCH = """Facts stated by a scene description (numbered F1, F2, ...):
{facts}

Requirements that a planner extracted from the same description (numbered R1, R2, ...):
{descs}

For each fact, give the number of one requirement that covers it, or null if none does. A fact is covered when a requirement states the same thing, even in different words, or when it is clearly implied by an entity requirement together with that entity's state, attribute or relation requirement. A placement relative to another object (beside, behind, on, under) is covered only by a requirement that names that placement. Reply with one JSON object and nothing else: {{"F1": <R number or null>, "F2": <R number or null>, ...}}"""


def _key():
    k = os.environ.get("DASHSCOPE_API_KEY")
    if k:
        return k
    if os.path.exists(SECRETS):
        with open(SECRETS) as f:
            for line in f:
                if line.startswith("DASHSCOPE_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no planner api key (DASHSCOPE_API_KEY or secrets.env)")


def ledger_total():
    tot = 0
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            for line in f:
                try:
                    tot += json.loads(line).get("total_tokens", 0)
                except Exception:
                    pass
    return tot


def api_chat(messages, model=DEFAULT_MODEL, max_tokens=6000, temperature=0.2, tag="", json_mode=True, cap=None, retries=3):
    cap = cap or int(os.environ.get("P0_API_TOKEN_CAP", "3000000"))
    used = ledger_total()
    if used >= cap:
        raise RuntimeError(f"api token cap reached: {used} >= {cap}")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "enable_thinking": False}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    s = requests.Session()
    s.trust_env = os.environ.get("P0_TRUST_ENV", "1") != "0"
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = s.post(API_URL, headers={"Authorization": "Bearer " + _key(), "Content-Type": "application/json"}, json=body, timeout=300)
            d = r.json()
        except Exception as ex:
            last = ex
            time.sleep(2 + 3 * attempt)
            continue
        if "choices" not in d:
            last = RuntimeError("api error: " + json.dumps(d, ensure_ascii=False)[:500])
            time.sleep(2 + 3 * attempt)
            continue
        u = d.get("usage", {})
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as f:
            f.write(json.dumps(dict(ts=time.strftime("%F %T"), model=model, tag=tag, prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0), total_tokens=u.get("total_tokens", 0), sec=round(time.time() - t0, 1))) + "\n")
        return d["choices"][0]["message"]["content"]
    raise RuntimeError(f"planner api failed after {retries} attempts: {last}")


def parse_json(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json object in planner output")
    return json.loads(m.group(0))


ENTITY_BAD = re.compile(r"\b(held by|holding|carrying|being|talking|looking|standing|sitting|lying|leaning|kneeling|to the left|to the right|on the left|on the right|in the center|in the middle|foreground|background|center|centre|left side|right side)\b", re.I)
ENTITY_PLACE = re.compile(r"\b(on (the|a|an|top of)|under|beneath|behind|beside|near|next to|in front of|above|below|over (the|a|an)|atop|by the)\b", re.I)
CHILD_BAD = re.compile(r"\b(on the|under the|underneath the|behind the|beside the|next to the|near the|in front of the|over the|above the|below the|toward the|towards the|at the|onto the|into the|against the|from the|with the|to the)\b", re.I)


FRAME_POS = re.compile(r",?\s*(positioned |located |standing |sitting )?(in the (foreground|background|center|centre|middle|distance)( (left|right))?( of the (image|scene|frame|picture))?|(on|to|at) the (far )?(left|right)( side| edge)?( of the (image|scene|frame|picture))?|in the (upper|lower|top|bottom) (left|right)( corner)?|at the (top|bottom)( of the (image|scene|frame|picture))?)", re.I)


def strip_frame_positions(text):
    """Rule 7: positions relative to the image frame never appear in a question; they live in the box."""
    out = FRAME_POS.sub("", str(text))
    out = re.sub(r"\s+", " ", out).strip()
    out = re.sub(r"\s+\?$", "?", out)
    return out


def _other_entity_words(d, own):
    """Words (class, plural, last word of desc) of every entity node other than `own`, mapped to that entity's id."""
    words = {}
    for n in d.get("nodes", []):
        if n.get("type") != "entity" or n.get("id") == own:
            continue
        for w in (n.get("class"), n.get("plural")):
            if w:
                words.setdefault(str(w).lower().strip(), n["id"])
        desc = re.sub(r"[^a-z ]", " ", str(n.get("desc", "")).lower()).split()
        if desc:
            words.setdefault(desc[-1], n["id"])
    return words


def _placed_on_other(d, n):
    """For an attribute or state node: the id of another entity node that its question places it relative to, else None."""
    others = _other_entity_words(d, n.get("of"))
    q = str(n.get("question", ""))
    for m in CHILD_BAD.finditer(q):
        tail = re.sub(r"[^a-z ]", " ", q[m.end():].lower()).split()[:4]
        for w in tail:
            if w in others:
                return others[w], m.group(0), w
            if w.rstrip("s") in others:
                return others[w.rstrip("s")], m.group(0), w
    return None


def lift_states_to_relations(d):
    """A state or attribute whose question places its entity relative to another entity node is, by rule 3 and 4, a
    relation between the two: retype it (same question) instead of asking the planner to try again."""
    for n in d.get("nodes", []):
        if n.get("type") in ("attribute", "state") and isinstance(n.get("of"), str):
            hit = _placed_on_other(d, n)
            if hit and hit[0] != n.get("of"):
                n["type"], n["of"] = "relation", [n["of"], hit[0]]
    return d


def planner_errors(d):
    """Checks on the planner's wording that Plan.validate does not make (the mock fixtures violate them on purpose):
    entity questions ask presence and intrinsic look only; attribute and state questions place their entity relative to
    no other entity node of the plan (a place relative to the background, "on the floor", is allowed)."""
    errs = []
    for n in d.get("nodes", []):
        q = str(n.get("question", ""))
        if n.get("type") == "entity":
            m = ENTITY_BAD.search(q)
            if m:
                errs.append(f"node {n.get('id')}: entity question contains '{m.group(0)}'; ask only presence and intrinsic look (kind, color, clothing, material), move the place, pose or interaction to a state or relation node")
                continue
            others = _other_entity_words(d, n.get("id"))
            for m in ENTITY_PLACE.finditer(q):
                tail = re.sub(r"[^a-z ]", " ", q[m.end():].lower()).split()[:3]
                hit = [w for w in tail if w in others or w.rstrip("s") in others]
                if hit:
                    errs.append(f"node {n.get('id')}: entity question says '{m.group(0)} ... {hit[0]}', placing it relative to another entity node; ask only presence and intrinsic look here and make the placement a relation node with both entity ids")
                    break
            else:
                nw = len(q.split())
                if nw > 22:
                    errs.append(f"node {n.get('id')}: entity question is {nw} words long; keep only the kind and one or two identifying looks, and write each other stated property as its own attribute node")
        elif n.get("type") in ("attribute", "state"):
            hit = _placed_on_other(d, n)
            if hit:
                errs.append(f"node {n.get('id')}: {n.get('type')} question says '{hit[1]} ... {hit[2]}', which places it relative to another entity node; keep only the pose or property here and make the placement a relation node with both entity ids")
    return errs


def _scale_example(W, H):
    d = json.loads(EXAMPLE_ASSISTANT)
    for n in d["nodes"]:
        if "box" in n:
            n["box"] = [int(round(n["box"][0] * W / 1024)), int(round(n["box"][1] * H / 1024)), int(round(n["box"][2] * W / 1024)), int(round(n["box"][3] * H / 1024))]
    return json.dumps(d, ensure_ascii=False)


def plan_prompt(prompt, pid, W=1024, H=1024, model=DEFAULT_MODEL, retries=2, max_nodes=64, min_side=48, chat=api_chat):
    """Prompt -> validated Plan dict (planner nodes only; generated nodes are added by add_generated_nodes)."""
    msgs = [
        {"role": "system", "content": SYSTEM.format(W=W, H=H)},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": _scale_example(W, H)},
        {"role": "user", "content": "Scene description:\n" + prompt.strip()},
    ]
    last = (None, ["no attempt"], [])
    seen_boxes = {}
    for attempt in range(retries + 1):
        out = chat(msgs, model=model, tag=f"plan:{pid}:{attempt}", max_tokens=12000)
        try:
            d = parse_json(out)
            d["id"] = pid
            d["prompt"] = prompt
            d["canvas"] = [W, H]
            d["meta"] = dict(planner=f"{model}:v4", attempts=attempt + 1)
            d["nodes"] = [n for n in d.get("nodes", []) if not (n.get("type") in ("attribute", "state") and (n.get("of") in (None, "", "bg")))]
            for n in d.get("nodes", []):
                if n.get("type") == "entity":
                    for k in ("question", "desc"):
                        if n.get(k):
                            n[k] = strip_frame_positions(n[k])
                if n.get("type") in ("entity", "background") and not isinstance(n.get("box"), list) and n.get("id") in seen_boxes:
                    n["box"] = list(seen_boxes[n["id"]])
                if isinstance(n.get("box"), list) and len(n["box"]) == 4 and n.get("type") == "entity":
                    try:
                        n["box"] = _grow_box([float(x) for x in n["box"]], W, H, min_side)
                    except (TypeError, ValueError):
                        pass
                if isinstance(n.get("box"), list):
                    n["box"] = [int(round(float(x))) for x in n["box"]]
                    if n.get("type") in ("entity", "background"):
                        seen_boxes[n["id"]] = list(n["box"])
                if n.get("type") == "entity":
                    n["class"] = str(n.get("class", "")).strip().lower()
                    n["plural"] = str(n.get("plural", "")).strip().lower() or None
                    try:
                        n["count"] = int(n.get("count", 1) or 1)
                    except (TypeError, ValueError):
                        n["count"] = 1
                    if "count_exact" in n and not isinstance(n["count_exact"], bool):
                        n["count_exact"] = str(n["count_exact"]).strip().lower() not in ("false", "0", "no")
                if n.get("type") in ("action", "relation") and isinstance(n.get("of"), list):
                    ent_ids = {m.get("id") for m in d["nodes"] if m.get("type") == "entity"}
                    n["of"] = list(dict.fromkeys(u for u in n["of"] if u != "bg" and (u in ent_ids or attempt < retries)))
                    if len(n["of"]) == 1:
                        n["type"], n["of"] = "state", n["of"][0]
            for e in d.get("edges", []):
                e.setdefault("kind", "support")
            lift_states_to_relations(d)
            errs, warns = Plan(d).validate(min_side=min_side, max_nodes=max_nodes)
            if not errs:
                errs = planner_errors(d)
        except Exception as ex:
            d, errs, warns = None, [f"invalid json: {ex}"], []
        if not errs:
            d["meta"]["warnings"] = warns
            return d, [], warns
        last = (d, errs, warns)
        print(f"  attempt {attempt} of {pid} invalid: {errs[:4]}", flush=True)
        try:
            os.makedirs(os.path.join(P0, "logs"), exist_ok=True)
            with open(os.path.join(P0, "logs", "invalid_attempts.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(pid=pid, attempt=attempt, model=model, errors=errs, nodes=[dict(id=n.get("id"), type=n.get("type"), of=n.get("of")) for n in (d or {}).get("nodes", []) if n.get("id") in " ".join(errs)]), ensure_ascii=False) + "\n")
        except Exception:
            pass
        msgs = msgs + [{"role": "assistant", "content": out}, {"role": "user", "content": "The plan is invalid:\n- " + "\n- ".join(errs[:12]) + "\nReturn the complete corrected JSON object with every node, keeping the boxes of all entities."}]
    return last


OUT_OF_SCOPE = re.compile(r"\b(conveys?|suggests?|evok\w+|clearly visible|visible|shadows?|reflections?|reflect\w*|dew|raindrops?|rain|lens|sunlight|daylight|moonlight|lighting|lights?|lit|glow\w*|flares?|dust|motes?|sparkles?|speed lines|motion lines|blur\w*|focus|sharp|crisp|textures?|textured|sheen|hues?|tones?|palette|mood|atmosphere|composition|corner|edge of the (image|frame|scene)|(bottom|top) edge|upper (left|right)|lower (left|right)|far (left|right)|in the (background|foreground|center|middle|distance)|on the (left|right)( side)?|to the (left|right)$|overhead)\b", re.I)
NOT_STILL = re.compile(r"\b(trembl\w*|sway\w*|vibrat\w*|flicker\w*|puls\w*|tick\w*|vanish\w*|dissolv\w*|forming|unravel\w*|rise and fall|accumulat\w*|slowly|gradually|gently turn\w*|turns gently|begins? to|starts? to|stops? \w+ing|years? old|no older than|no younger than|as if|as though|seems? to|appears? to|once was|used to|will|humm\w*|ringing|groan\w*|reverberat\w*|whisper\w*|murmur\w*|chirp\w*|buzz\w*|echo\w*|smell\w*|scent\w*)\b", re.I)
RELATIVE_OK = re.compile(r"\b(to (his|her|its|their) (left|right)|to the (left|right) of (the|a|an|his|her|its|their) \w+)\b", re.I)


def in_scope(fact):
    """Facts the checker never verifies by design (lighting, texture, mood, frame positions, commentary) are dropped
    before matching; a placement relative to another object ("to his left") is kept."""
    f = str(fact)
    if NOT_STILL.search(f):
        return False
    if RELATIVE_OK.search(f):
        return True
    return not OUT_OF_SCOPE.search(f)


def _missing_facts(facts, matches):
    out = []
    for i, f in enumerate(facts):
        m = matches.get(f"F{i + 1}", matches.get(str(i + 1)))
        if isinstance(m, str):
            m = m.strip().upper().lstrip("R") or None
            m = None if m in ("NULL", "NONE", "") else m
        if m in (None, 0):
            out.append(f)
    return out


COVER_USER = """Scene description:
{prompt}

A first pass produced these nodes (id, type, of, desc):
{nodes}

List every concrete visible fact stated in the description that none of these nodes covers yet: a person, animal or object; a pose, expression or condition of an entity that the description states in words (including ordinary ones such as "stands", "sits", "walks" when those words are there); a stated property of an object; an interaction or a placement between two entities (beside her, between the woman and the man, in front of the tent, resting on the floor); a stated count; a stated absence. Never add a fact the description does not state in words: no inferred poses (a person who pours or waits is not "standing" unless the text says so), no inferred targets, no inferred placements. Skip art style, lighting, shadows, textures, mood, commentary and positions relative to the image frame. For each such fact write the node that covers it, following the same rules and schema as before (one fact per node; attribute and state have "of" set to one existing entity id; action and relation have "of" with two or more existing entity ids; a fact about an entity that has no node yet needs a new entity node with a box). Do not repeat or rewrite existing nodes, and do not add a node whose fact an existing node already states in other words. Do not add an entity node for an object that is only held, worn or handed over (it belongs in an attribute or action node of the person). Reply with one JSON object and nothing else: {{"nodes": [...], "edges": [...]}}; return {{"nodes": [], "edges": []}} when nothing is missing."""


def _node_listing(d):
    return "\n".join(f"{n['id']} [{n['type']}]{' of ' + json.dumps(n['of']) if n.get('of') else ''}: {n.get('desc', '')}" for n in d["nodes"])


def coverage_pass(d, model=DEFAULT_MODEL, chat=api_chat, min_side=48, max_nodes=64):
    """Second planner call: ask for nodes covering stated facts the first pass left out, merge the valid ones.
    Returns (plan dict, n_added, n_dropped)."""
    W, H = d.get("canvas", [1024, 1024])
    msgs = [
        {"role": "system", "content": SYSTEM.format(W=W, H=H)},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": _scale_example(W, H)},
        {"role": "user", "content": COVER_USER.format(prompt=d["prompt"].strip(), nodes=_node_listing(d))},
    ]
    out = chat(msgs, model=model, tag=f"cover:{d['id']}", max_tokens=8000)
    try:
        add = parse_json(out)
    except Exception:
        return d, 0, 0
    trial, n_added, n_dropped, _errs = _merge_added(d, add, min_side=min_side, max_nodes=max_nodes)
    if n_added:
        trial["meta"] = dict(trial.get("meta", {}), coverage_added=n_added, coverage_dropped=n_dropped)
    return trial, n_added, n_dropped


COVER_FACTS_USER = """Scene description:
{prompt}

Earlier passes produced these nodes (id, type, of, desc):
{nodes}

A separate reading of the description, made without seeing the nodes, listed the facts below; a check found no node that covers them:
{facts}

For each listed fact that the description states in words and that one still image can show, write the node that covers it, following the same rules and schema as before: one fact per node; attribute and state have "of" set to one existing entity id; action and relation have "of" with two or more existing entity ids; a placement between two objects is a relation node between their entity nodes; a fact about an object that has no entity node yet first needs a new entity node (with class, count and box), then the attribute, state or relation node. Every new node carries one more field, "evidence": the words of the description that state the fact, copied verbatim (one clause or sentence, at most 40 words); for an action or relation node the evidence names every participant (a pronoun counts). Skip a listed fact when it is motion or change over time, a sound, a smell, a mood, a simile, a frame position, something hidden or invisible, or when no sentence of the description states it (a listed fact can be wrong: "the dog is beside the tree" is not stated by "Beside the tree, a woman ..."); skip it too when an existing node already states it in other words. Never add anything the description does not state (no inferred poses, targets, sides or placements). Do not repeat or rewrite existing nodes. Do not add an entity node for an object that is only held, worn or handed over (it belongs in an attribute or action node of the person). Reply with one JSON object and nothing else: {{"nodes": [...], "edges": [...]}}; return {{"nodes": [], "edges": []}} when nothing is missing."""

PLAN_FACTS = AUDIT_FACTS.replace(
    "Split this description into its concrete visible facts, one fact per entry, as short phrases.",
    "Split this description into its concrete visible facts, one fact per entry, as short phrases. Be exhaustive: go through the description sentence by sentence and list every stated object (also a minor one mentioned only once), every stated placement of one object relative to another (beside it, on the bench, behind them, near the door), every stated pose, expression, condition or property of an object, every count and every absence; a long description yields 40 to 80 facts.")


def _fact_key(f):
    return frozenset(_stem(w) for w in _words(f) if w not in STOP and w not in PRONOUNS)


PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their", "hers", "theirs", "himself", "herself", "itself", "themselves", "who", "whose", "which", "that"}
STOP = {"a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "with", "for", "by", "from", "as", "is", "are", "was", "were", "be", "its", "his", "her", "their", "this", "that", "these", "those", "into", "onto", "over", "under", "near", "beside", "behind", "above", "below", "across", "through", "toward", "towards", "while", "where", "when", "one", "two", "three", "some", "several", "each", "every", "all", "both", "another", "other", "second"}


def _words(text):
    t = str(text or "").lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", " ").replace("–", " ")
    return re.findall(r"[a-z0-9']+", t)


def _stem(w):
    w = w.rstrip("'")
    for suf in ("ies", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _entity_terms(nodes, eid):
    """Content words a sentence could use to name entity eid: its class words plus the content words of its desc."""
    for n in nodes:
        if n.get("id") == eid:
            ws = _words(n.get("class", "")) + _words(n.get("plural", "")) + _words(n.get("desc", ""))
            return {_stem(w) for w in ws if w not in STOP and w not in PRONOUNS and len(w) > 2}
    return set()


def grounded(node, prompt, nodes):
    """True when the node's evidence quote is found in the description (3-gram overlap) and names the node's
    participants (by a class/desc word or a pronoun). Used by facts_pass to keep invented facts out of the plan."""
    ev = _words(node.get("evidence", ""))
    if len(ev) < 3:
        return False
    pw = _words(prompt)
    grams = {tuple(pw[i:i + 3]) for i in range(len(pw) - 2)}
    hit = sum(1 for i in range(len(ev) - 2) if tuple(ev[i:i + 3]) in grams)
    if hit / max(1, len(ev) - 2) < 0.6:
        return False
    evs = {_stem(w) for w in ev}
    has_pronoun = any(w in PRONOUNS for w in ev)
    if node.get("type") == "entity":
        return bool(_entity_terms([node], node.get("id")) & evs)
    of = node.get("of")
    ids = [e for e in (of if isinstance(of, list) else [of]) if e]
    if not ids:
        return True
    named = [bool(_entity_terms(nodes, e) & evs) for e in ids]
    unnamed = named.count(False)
    return unnamed == 0 or (unnamed == 1 and has_pronoun)


def facts_pass(d, model=DEFAULT_MODEL, chat=api_chat, min_side=48, max_nodes=96, samples=(0.0, 0.7)):
    """Checklist coverage pass for long prose prompts: (1) the planner model lists the description's checkable facts
    without seeing the plan, (2) it matches each fact to one requirement of the plan (generated count nodes included),
    (3) it writes nodes only for the facts that matched nothing, each with a verbatim evidence quote; nodes whose
    quote is not in the description or does not name their participants are dropped (grounded). The gate audit does
    the same extraction and matching with a different model. Returns (plan dict, info)."""
    info = dict(n_facts_raw=0, n_facts=0, n_uncovered=0, uncovered=[], proposed=0, ungrounded=[], added=0, dropped=0)
    prompt = d["prompt"].strip()
    raw, keys = [], set()
    for k, temp in enumerate(samples):  # two readings of the description, united (each one misses a few facts)
        out1 = chat([{"role": "user", "content": PLAN_FACTS.format(prompt=prompt)}], model=model, tag=f"plan_facts:{d['id']}", max_tokens=4000, temperature=temp)
        try:
            got = [str(f) for f in parse_json(out1).get("facts", []) if str(f).strip()]
        except Exception:
            if k == 0:
                return d, dict(info, error="facts_parse")
            continue
        for f in got:
            key = _fact_key(f)
            if key and key not in keys:
                keys.add(key)
                raw.append(f)
    facts = [f for f in raw if in_scope(f)]
    info.update(n_facts_raw=len(raw), n_facts=len(facts))
    if not facts:
        return d, info
    full = add_generated_nodes(Plan(json.loads(json.dumps(d))))
    req_nodes = [v for v in full.order if full.typ(v) not in ("pair", "quality")]
    descs = "\n".join(f"R{i + 1}. [{full.typ(v)}] {full.nodes[v]['desc']}" for i, v in enumerate(req_nodes))
    flist = "\n".join(f"F{i + 1}. {f}" for i, f in enumerate(facts))
    matches, uncovered = {}, list(facts)
    for _try in range(2):  # a matcher answer that leaves most facts unmatched is a format failure: ask once more
        out2 = chat([{"role": "user", "content": AUDIT_MATCH.format(facts=flist, descs=descs)}], model=model, tag=f"plan_match:{d['id']}", max_tokens=4000, temperature=0.0 if _try == 0 else 0.3)
        try:
            matches = parse_json(out2)
        except Exception:
            matches = {}
        uncovered = _missing_facts(facts, matches)
        if len(uncovered) <= max(2, len(facts) // 2):
            break
    if len(uncovered) > max(2, len(facts) // 2):
        return d, dict(info, n_uncovered=len(uncovered), error="match_suspect")
    info.update(n_uncovered=len(uncovered), uncovered=uncovered)
    if not uncovered:
        return d, info
    W, H = d.get("canvas", [1024, 1024])
    msgs = [
        {"role": "system", "content": SYSTEM.format(W=W, H=H)},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": _scale_example(W, H)},
        {"role": "user", "content": COVER_FACTS_USER.format(prompt=prompt, nodes=_node_listing(d), facts="\n".join(f"- {f}" for f in uncovered))},
    ]
    out3 = chat(msgs, model=model, tag=f"cover_facts:{d['id']}", max_tokens=8000)
    try:
        add = parse_json(out3)
    except Exception:
        return d, dict(info, error="cover_parse")
    proposed = [n for n in add.get("nodes", []) if isinstance(n, dict) and n.get("id")]
    pool = d["nodes"] + proposed
    keep, ungrounded = [], []
    for n in proposed:
        if grounded(n, prompt, pool):
            keep.append(n)
        else:
            ungrounded.append(dict(id=n.get("id"), type=n.get("type"), of=n.get("of"), desc=n.get("desc"), evidence=n.get("evidence")))
    info.update(proposed=len(proposed), ungrounded=ungrounded)
    for n in keep:
        n.pop("evidence", None)
    trial, n_added, n_dropped, merge_errs = _merge_added(d, dict(add, nodes=keep), min_side=min_side, max_nodes=max_nodes)
    info.update(added=n_added, dropped=n_dropped, kept=[dict(id=n.get("id"), type=n.get("type"), of=n.get("of"), desc=n.get("desc")) for n in keep], merge_errors=list(merge_errs)[:8])
    if n_added:
        trial["meta"] = dict(trial.get("meta", {}), facts_pass=dict(n_facts=len(facts), n_uncovered=len(uncovered), proposed=len(proposed), ungrounded=len(ungrounded), added=n_added, dropped=n_dropped))
    return trial, info


ATOMIZE_USER = """Scene description:
{prompt}

The current plan has these nodes (id [type] of: question):
{nodes}
Support edges: {edges}

Check every node against rules 1 and 7: one node asks about one checkable fact. Return only the changes needed.

1. "rewrite": a node whose question asks about two or more separately checkable facts, so that an image could satisfy one and not the other. Typical cases:
(a) an entity question that names, besides the kind and the one or two looks needed to tell it apart from other entities of the same kind, further properties ("a tall humanoid alien with translucent skin, an elongated head, four arms and webbed feet"): keep the entity, same id, with a short question (the kind plus the distinguishing look), and give each further property its own attribute or state node of that entity, unless another node already states it;
(b) an entity question that names a second object ("a bowl filled with lemons", "dirty plates and glasses", "herbs in clay pots"): keep the first object as the entity, same id, add an entity node for each other object (with class, plural, count, count_exact and box) and a relation node for how they are placed when the description states it;
(c) an attribute or state question about two or more properties or poses ("four articulated arms ending in suction-cup tips", "a smooth, glossy, pearlescent sheen"): keep the first property in the node, same id, and give each other property its own node;
(d) a background question that lists objects besides the setting ("a kitchen with granite countertops, a stove, an island and a sink"): keep the background, same id, with the setting only, and give each listed object an entity node with a box;
(e) an absent question about two or more things ("a coffee maker or espresso machine"): keep the first thing in the node, same id, and give each other thing its own absent node;
(f) a state, attribute or action question that joins a fact a still image shows with a sound, a smell, a temperature or motion over time ("narrow headlights pulsing with white light", "a fire pit crackling with burning logs"): keep only the visible fact in the node, same id ("narrow headlights glowing white", "a fire pit with burning logs"), with no new node for the part a still image cannot show.
Granularity: each of these is a separate fact: the number of parts (three hinges), a color (a white face), a material (made of dark brown wood), a pattern (carved floral patterns), a condition (rusty, worn, oxidized, cracked, open, lit), the position of a part (along the back edge), what something holds or contains (filled with lemons), and each item of clothing. So "three rusty metal hinges along its back edge" is three facts (it has three hinges; the hinges are rusty; the hinges are along the back edge), "a white face with black Roman numerals" is two, and "a woman in a straw hat and a floral sundress" is a woman plus two attributes. An entity question keeps only the kind, plus one look only when another entity of the same kind must be told apart from it ("a second researcher in a red coat"); every other look becomes an attribute node.
How to write a rewrite: "node" is the node after the rewrite, with the same id and type and all its fields, and its question now asks only one fact of the old question; "new" has one node for each other fact of the old question; "already" lists the ids of nodes the plan already has for some of those facts, which are not written again. The node, its new nodes and the already-listed nodes together ask every fact of the old question that a still image can show, each fact once. Never return the old question unchanged in "node", and never list a node that needs no change. Nodes most likely to need a rewrite: {flagged}.
2. "add": a requirement the description states and the plan has no node for, of two kinds only: a comparison between two entities (taller, older, thicker, wider, newer, bigger than) as a relation node of both; a count relation between two kinds ("twice as many carrots as apples", "one more guard than display cases"): when the numbers follow from the description give it in "set_count" (the entity id and its number) instead of a node, otherwise as a relation node of both entities.
3. "delete": a state, attribute or action node all of whose facts one still image cannot show: a sound, a smell, a temperature, or motion and change over time (rattling, trembling, flickering, pulsing, ticking); also a node that only asks where something is in the image frame (on the left side of the image). A node that also states something a still image shows is rewritten as in (f), not deleted. Nothing else is deleted: a placement between two objects (on the counter, near the stove, in the sink), a pose a still image shows (running, flying, fluttering, bending, steam rising, watching), a stated absence, and a node that seems to repeat another one all stay.

Every new node follows the schema and rules above, is about this description only (never about the example), and carries one more field, "evidence": the words of the description that state its fact, copied verbatim as one contiguous piece without "..." (one clause, at most 40 words); for a relation or action the evidence names every participant. Do not change nodes that are fine, do not add anything the description does not state, and do not add nodes of any other kind. Reply with one JSON object and nothing else:
{{"rewrite": [{{"node": {{...}}, "new": [...], "already": ["<id>", ...]}}], "add": [...], "set_count": [{{"id": "<entity id>", "count": <number>, "evidence": "..."}}], "delete": [{{"id": "<id>", "why": "<sound, smell, motion over time, ...>"}}], "edges": [...]}}
When every node is fine, reply {{"rewrite": [], "add": [], "set_count": [], "delete": [], "edges": []}}."""

ATOMIZE_REPAIR = """These rewrites could not be applied:
{problems}
Return corrected rewrites for these nodes only, as one JSON object {{"rewrite": [{{"node": {{...}}, "new": [...], "already": [...]}}], "edges": [...]}}, under the same rules: the node keeps its id and type and asks one fact of the old question; every other fact of the old question that a still image can show is asked by exactly one new or already-listed node; each new node carries an "evidence" quote copied verbatim from the description (one contiguous piece of at least three words that names every participant). Leave out a node you now think needs no rewrite."""

ATOMIZE_DELETABLE = ("state", "attribute", "action")

# words of a question that state what one still image cannot show: a sound, a smell, a temperature, motion over time
NONVISUAL_WORD = re.compile(r"(sounds?|sounding|nois(e|es|y|ily)|sizzl\w*|crackl\w*|creak\w*|ticks?|ticking|ticked|clicks?|clicking|clack\w*|chatter\w*|hums?|humm(ing|ed)|buzz\w*|ringing|chirp\w*|roar\w*|whistling|rattl\w*|clank\w*|clatter\w*|rustl\w*|squeak\w*|squeal\w*|purr\w*|howl\w*|growl\w*|echo\w*|murmur\w*|whisper\w*|groan\w*|reverberat\w*|hiss\w*|gurgl\w*|babbl\w*|tinkl\w*|jingl\w*|chiming|melod\w*|audibl\w*|barking|barks|barked|honking|honks|beeping|beeps|meowing|meows|clucking|mooing|neighing|loudly|quietly|smell\w*|scent\w*|fragran\w*|aroma\w*|odou?rs?|stink\w*|stench|perfumed|temperatures?|warm|warms|warming|warmth|warmed|hot|heat|heats|heated|heating|cold|cool|cools|cooling|cooled|chill|chills|chilled|chilling|chilly|freezing|frigid|lukewarm|scalding|sway\w*|flicker\w*|puls\w*|trembl\w*|vibrat\w*|twinkl\w*|slowly|gradually)")
NONVISUAL_PHRASE = re.compile(r"\b(back and forth|to and fro|rise and fall|rising and falling|begins? to|starts? to|over time"
                              r"|(smell|scent|aroma|odou?r|fragrance|stench|sound|noise|hum|buzz|roar|rumble|chime|clatter)(s|ed|ing)?( of| like)( \w+){1,3})\b", re.I)
# modifiers that come with such a word and state nothing a still image shows ("clicking rapidly", "swaying gently in the breeze")
MOD_FREE_WORDS = ("gently", "gentle", "softly", "soft", "rapidly", "quickly", "loudly", "quietly", "faintly", "faint", "steadily", "steady", "constantly", "constant",
                  "rhythmically", "rhythmic", "continuously", "occasionally", "back", "forth", "breeze", "motion", "movement", "moving", "visible", "visibly",
                  "produce", "produced", "producing", "emit", "emitting", "make", "makes", "making", "fill", "filled", "filling", "air", "slight", "slightly", "appear", "appears", "seem", "seems")
NEG = {"not", "no", "without", "never", "none", "nothing", "neither", "nor"}

COMPARE = re.compile(r"\b(\w+er than|more than|less than|fewer than|as many|as much|twice|double|triple|three times|half as|same number|equal number|same size|same height|largest|smallest|tallest|shortest|biggest|oldest|youngest|longest|widest)\b", re.I)


def _cstem(w):
    """Stem for content-word comparison: possessive, plural, -ing, -ed and a final e fold together, so that singular and
    plural, verb and participle meet (arms/arm, trees/tree, tables/table, glasses/glass, carved/carving/carve,
    sitting/sits, overlapping/overlap, alien's/alien)."""
    w = w[:-2] if w.endswith("'s") else w.rstrip("'")
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "shes", "ches", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            if w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def _content(text, extra_stop=()):
    return {_cstem(w) for w in _words(text) if w not in STOP and w not in PRONOUNS and len(w) > 2 and w not in extra_stop}


SPLIT_FREE = {"does", "do", "is", "are", "has", "have", "there", "any", "image", "show", "shows", "visible", "visibly", "wear", "wears", "wearing", "hold", "holds", "holding", "contain", "contains", "its", "his", "her", "their"}
MOD_FREE = {_cstem(_w) for _w in MOD_FREE_WORDS}
# words a rewrite may leave unasked: linking verbs and hedges, frame words (rule 7), motion words a still image cannot show
LOST_FREE = {_cstem(_w) for _w in ("show", "showing", "display", "displaying", "feature", "featuring", "depict", "depicting", "consist", "consisting", "made",
                                   "appear", "appears", "seem", "seems", "like", "such", "exactly", "slightly", "foreground", "background", "moving", "spinning",
                                   "area", "areas", "contain", "containing", "contains", "filled", "free", "calmly", "peacefully", "time", "various", "different",
                                   "entire", "entirely", "whole", "overall", "actual", "real", "kind", "type", "sort", "piece", "pieces")}


def _of_ids(n):
    of = n.get("of")
    return [e for e in (of if isinstance(of, list) else [of]) if e]


def _names(nodes, ids):
    """Content words that name the given entities (class, plural, desc)."""
    out = set()
    want = set(ids)
    for n in nodes:
        if n.get("id") in want:
            out |= _content(f"{n.get('class', '')} {n.get('plural', '')} {n.get('desc', '')}")
    return out


def _qnames(nodes, ids):
    """Words that name the given entities in what is checked: their class, plural and their own entity question. Not
    the desc, which often still carries looks no question asks (10-10: a countertop's desc "a cracked and worn countertop
    surface" made "Is the countertop surface cracked and worn?" look like it asked nothing, so any node "covered" it)."""
    out = set()
    want = set(ids)
    for n in nodes:
        if n.get("id") in want:
            out |= _content(f"{n.get('class', '')} {n.get('plural', '')}") | _content(n.get("question", ""), SPLIT_FREE)
    return out


def _neg(n):
    """The node asks for something to be missing: an absent node, or a question with a negation."""
    return n.get("type") == "absent" or bool(set(_words(n.get("question", ""))) & NEG)


def _class_names(nodes, ids):
    """Content words of the class and plural of the given entities (the nouns that name them: "heat lamp", "hot dog")."""
    out = set()
    want = set(ids)
    for n in nodes:
        if n.get("id") in want:
            out |= _content(f"{n.get('class', '')} {n.get('plural', '')}")
    return out


RELATIVE_TAIL = re.compile(r"\s+of\s+(the\s+|a\s+|an\s+)?(?!image\b|scene\b|frame\b|picture\b)[a-z]", re.I)


def _frame_sub(q):
    """q without the positions it gives in the image frame ("on the left side of the image", "in the background"); a
    position relative to another object ("on the left side of the waterhole", "in the center of the room") is kept
    (10-10: two such relations were deleted as frame positions). Returns (text, whether a frame position was there)."""
    out, last, hit = [], 0, False
    for m in FRAME_POS.finditer(q):
        if RELATIVE_TAIL.match(q, m.end()):
            continue
        out.append(q[last:m.start()])
        out.append(" ")
        last, hit = m.end(), True
    out.append(q[last:])
    return "".join(out), hit


def _nonvisual(q, names):
    """(stems of the question's words that state what a still image cannot show, the question without non-visual phrases
    and frame positions, whether such a phrase or position was there). Words that are part of a participant's class
    name ("heat lamps", "hot dogs") do not count."""
    q = str(q)
    t, framed = _frame_sub(NONVISUAL_PHRASE.sub(" ", q))
    phrase = bool(NONVISUAL_PHRASE.search(q) or framed)
    cues = {_cstem(w) for w in _words(t) if NONVISUAL_WORD.fullmatch(w)} - names
    return cues, t, phrase


def _deletable(node, nodes):
    """A child node may go only when its own question states what a still image cannot show (a sound, smell, temperature
    or motion over time by its words, or only a position in the image frame) and everything else it says is a
    participant's name or is asked by another node. The planner's stated reason is not used (10-09: "sound" was given for
    "Is the speaker mounted on the wall?"). Returns (ok, words that would be lost)."""
    names = _qnames(nodes, _of_ids(node))
    cues, rest_text, phrase = _nonvisual(node.get("question", ""), _class_names(nodes, _of_ids(node)))
    if not cues and not phrase:
        return False, ["(no sound, smell, temperature or motion word)"]
    rest = _content(rest_text, SPLIT_FREE) - cues - names - MOD_FREE
    pool = set()
    for m in nodes:
        if m.get("id") != node.get("id"):
            pool |= _content(m.get("question", ""), SPLIT_FREE)
    lost = sorted(rest - pool)
    return not lost, lost


def _from_old(new, old, nodes):
    """A split-out restates part of the node it came from: its content words, apart from the names of its participants,
    are the old node's words with at most one new word. A split-out entity (the second object of a bundled entity or an
    object listed in the background) is judged by its own name words against the old node."""
    have = _content(f"{old.get('question', '')} {old.get('desc', '')}", SPLIT_FREE)
    if new.get("type") == "entity":
        got = _content(f"{new.get('desc', '')} {new.get('class', '')}", SPLIT_FREE)
        return bool(got) and len(got - have) <= 1
    names = _names(nodes, _of_ids(new))
    got = _content(f"{new.get('question', '')}", SPLIT_FREE) - names
    return bool(got) and len(got - (have | names)) <= 1


def _ev_in_prompt(evidence, prompt):
    """The evidence quote is in the description: 3-gram overlap at least 0.6, or a two-word quote found as it is."""
    ev = _words(str(evidence).replace("...", " ").replace("…", " "))
    pw = _words(prompt)
    if len(ev) == 2:
        return any(pw[i:i + 2] == ev for i in range(len(pw) - 1))
    if len(ev) < 3:
        return False
    grams = {tuple(pw[i:i + 3]) for i in range(len(pw) - 2)}
    return sum(1 for i in range(len(ev) - 2) if tuple(ev[i:i + 3]) in grams) / max(1, len(ev) - 2) >= 0.6


BUNDLE_HINT = re.compile(r"(,| and | with | made of | filled with | containing | covered in | covered with | showing | ending in | wearing .* and | or )", re.I)


def likely_bundled(d):
    """Ids of nodes whose question probably asks more than one fact: an entity question longer than seven words or with a
    list, joined looks or a second object; a background, attribute, state or absent question with a list or a joined
    look. A hint for the atomize prompt only; the planner decides."""
    out = []
    for n in d.get("nodes", []):
        q = str(n.get("question", ""))
        t = n.get("type")
        core = re.sub(r"^(is|are|does|do) (there|the image show|this image show)? ?(an?|the|any)? ?", "", q.lower()).rstrip("?")
        if t == "entity" and (len(core.split()) > 6 or BUNDLE_HINT.search(core)):
            out.append(n["id"])
        elif t in ("background", "attribute", "state", "absent") and BUNDLE_HINT.search(core):
            out.append(n["id"])
    return out


def _dup_of_existing(n, nodes):
    """Same type, same participants and the same content words as an existing node: nothing new to add."""
    of = n.get("of")
    key = (n.get("type"), json.dumps(sorted(of) if isinstance(of, list) else of))
    words = _content(n.get("question", ""), SPLIT_FREE)
    for m in nodes:
        mof = m.get("of")
        if (m.get("type"), json.dumps(sorted(mof) if isinstance(mof, list) else mof)) == key and words and _content(m.get("question", ""), SPLIT_FREE) == words:
            return True
    return False


def _covered_by(n, pool, nodes):
    """Id of a node of `pool` that already asks n's fact, else None: same polarity (absent and negated questions on one
    side) and n's content words, apart from its participants' names, all in that node's question. An entity is covered
    only by an entity of the same class whose question has all its words."""
    if n.get("type") == "entity":
        words = _content(n.get("question", ""), SPLIT_FREE)
        cls = _content(n.get("class", ""))
        for m in pool:
            if m.get("type") == "entity" and m.get("id") != n.get("id") and cls and cls <= _content(f"{m.get('class', '')} {m.get('plural', '')}") and words <= _content(m.get("question", ""), SPLIT_FREE):
                return m["id"]
        return None
    full = _content(n.get("question", ""), SPLIT_FREE)
    pol = _neg(n)
    mine = sorted(_of_ids(n))

    def same(m):  # a node about exactly the same participants, or an absent node (no participants) of the same polarity
        return m.get("id") != n.get("id") and m.get("type") != "entity" and _neg(m) == pol and (m.get("type") == "absent" or sorted(_of_ids(m)) == mine)

    for m in pool:  # the same words
        if same(m) and full and _content(m.get("question", ""), SPLIT_FREE) == full:
            return m["id"]
    words = full - _qnames(nodes, _of_ids(n))
    if not words:
        return None
    for m in pool:
        if same(m) and words <= _content(m.get("question", ""), SPLIT_FREE):
            return m["id"]
    return None


def _redundant(node, nodes):
    """A child node that only bundles other child nodes about the same participants: each of those asks a part of its
    words (apart from participants' names), with the same polarity, and together they ask all of them. Removing it
    loses nothing. Words merely shared with another fact ("a red cup" against "a red hat") do not count."""
    def own(m):
        # a relation's words naming its participants are its content ("dropping a coin into the guitar case"): only an
        # attribute or state, about one entity, leaves that entity's names out
        w = _content(m.get("question", ""), SPLIT_FREE)
        return w if m.get("type") in ("relation", "action") else w - _qnames(nodes, _of_ids(m))
    words = own(node)
    if not words:
        return False
    subj, pol, cover = sorted(_of_ids(node)), _neg(node), set()
    for m in nodes:
        if m["id"] == node["id"] or m.get("type") not in ("attribute", "state", "relation", "action", "absent") or _neg(m) != pol:
            continue
        # only nodes about exactly the same participants: "the man is sitting" is not a part of "the man is sitting on
        # the bench" (the relation's second participant carries the placement; 10-10 cleanup removed pose states so)
        if m.get("type") != "absent" and sorted(_of_ids(m)) != subj:
            continue
        w = own(m)
        if w and w <= words:
            cover |= w
    return words <= cover


def _subject_pool(nodes, old, exclude):
    """Nodes about the same subject as `old`: for a background the entity and absent nodes (an object the background
    lists must stay in it or be an entity; a mention inside another node's placement does not ask that it is there,
    10-09 pilot: "a kitchen with granite countertops, a stove, an island, a sink" lost all four that way); for an entity
    its own child nodes; for a child node its participant entities and every node sharing a participant; absent nodes
    always."""
    t = old.get("type")
    if t == "background":
        return [n for n in nodes if n["id"] not in exclude and n.get("type") in ("entity", "absent")]
    subj = {old["id"]} if t == "entity" else set(_of_ids(old))
    return [n for n in nodes if n["id"] not in exclude and (n.get("type") == "absent" or n["id"] in subj or set(_of_ids(n)) & subj)]


LOST_OK_NEEDS_SPLIT = 0  # content words of the old question a rewrite may leave unasked, beyond LOST_FREE (10-09 pilot: one free word let "tall", "worn", "sleek" go)


def _surface(stems, text):
    """The words of `text` whose stems are in `stems`, in order, for a readable problem message."""
    out = []
    for w in _words(text):
        if _cstem(w) in stems and w not in out:
            out.append(w)
    return out


def _normalize_new(n, W, H, min_side):
    """A new node as the first pass would write it: entity class lower-cased, count an int, frame positions stripped,
    box grown to the minimum side; a relation or action with one participant becomes a state."""
    if n.get("type") == "entity":
        n["class"] = str(n.get("class", "")).strip().lower()
        n["plural"] = str(n.get("plural", "")).strip().lower() or None
        try:
            n["count"] = int(n.get("count", 1) or 1)
        except (TypeError, ValueError):
            n["count"] = 1
        if "count_exact" in n and not isinstance(n["count_exact"], bool):
            n["count_exact"] = str(n["count_exact"]).strip().lower() not in ("false", "0", "no")
        for k in ("question", "desc"):
            if n.get(k):
                n[k] = strip_frame_positions(n[k])
        if isinstance(n.get("box"), list) and len(n["box"]) == 4:
            try:
                n["box"] = _grow_box([float(x) for x in n["box"]], W, H, min_side)
            except (TypeError, ValueError):
                pass
    if n.get("type") in ("action", "relation") and isinstance(n.get("of"), list):
        n["of"] = list(dict.fromkeys(u for u in n["of"] if u != "bg"))
        if len(n["of"]) == 1:
            n["type"], n["of"] = "state", n["of"][0]
    return n


def _unnamed(n, nodes):
    """Participants of a relation or action whose question does not name them (by a word of their class, desc or
    entity question): such a question no longer asks about them (10-10: "dropping a coin into the guitar case" became
    "dropping a coin" while the case stayed a participant)."""
    if n.get("type") not in ("relation", "action"):
        return []
    q = {_cstem(w) for w in _words(n.get("question", ""))}
    out = []
    for e in _of_ids(n):
        terms = {_cstem(w) for w in _words(" ".join(str(m.get(k, "")) for m in nodes if m.get("id") == e for k in ("class", "plural", "desc", "question"))) if w not in STOP and len(w) > 2}
        if terms and not (terms & q):
            out.append(e)
    return out


def _apply_rewrites(plan, rewrites, prompt, info, only=None, round_no=1):
    """Apply the planner's rewrites to plan dict `plan` (changed in place). Each rewrite becomes a group: the old node
    shortened (or retyped) under its id plus the new nodes that take over its other facts. A group is accepted only when
    (1) the kept node asks part of the old question (at most one new word) and is not longer; (2) every new node restates
    part of the old node, or is an attribute or state of the same entity quoted from the description, or carries a
    verbatim evidence quote that names its participants; (3) nothing of the old question is lost: its content words are
    asked by the kept node, the new nodes or a node the plan already has about the same subject, with at most one word
    missing and never a negation. When the kept node is the old question unchanged, the first new node of the same type
    about the same participants takes over the old id. A new node that a plan node already asks is not added.
    Returns (groups {old id: record}, new nodes, problems {old id: message})."""
    groups, new_nodes, problems = {}, [], {}
    by_id = {n["id"]: n for n in plan["nodes"]}
    flagged = set(likely_bundled(plan))
    for rw in rewrites or []:
        if not isinstance(rw, dict) or not isinstance(rw.get("node"), dict):
            continue
        nn = dict(rw["node"])
        oid = nn.get("id")
        if oid not in by_id or oid in groups or (only is not None and oid not in only):
            continue
        info["rewrites"] += 1
        old = by_id[oid]
        rec = dict(id=oid, round=round_no, old=old.get("question"), outcome=None)
        info["groups"].append(rec)
        extra = [json.loads(json.dumps(n)) for n in (rw.get("new") or []) if isinstance(n, dict) and n.get("id")]
        for n in extra:
            info.setdefault("_proposed", {}).setdefault(n["id"], oid)
        for n in extra:  # a relation or action with the background as a participant is about the other participant
            if n.get("type") in ("action", "relation") and isinstance(n.get("of"), list):
                n["of"] = list(dict.fromkeys(u for u in n["of"] if u != "bg"))
                if len(n["of"]) == 1:
                    n["type"], n["of"] = "state", n["of"][0]
        refs = [str(x) for x in (rw.get("already") or []) if isinstance(x, (str, int)) and str(x) in by_id and str(x) != oid]
        same = next((n for n in extra if n["id"] == oid), None)
        rep = None
        if same is not None and same.get("type") == old.get("type"):
            if len(str(same.get("question", "")).split()) < len(str(nn.get("question", "")).split()):
                nn = dict(nn, **{k: same[k] for k in ("desc", "question") if same.get(k)})
            extra = [n for n in extra if n["id"] != oid]
        elif same is not None:
            rep = same  # the same id under another type (a state that is a relation)
            extra = [n for n in extra if n is not same]
        if rep is None and nn.get("type") != old.get("type") and nn.get("type") in ("relation", "action") and nn.get("of"):
            rep = dict(nn)
        if rep is not None and (rep.get("type") not in ("relation", "action", "state", "attribute") or old.get("type") not in ("state", "attribute", "relation", "action")):
            rec["outcome"] = "bad_type"
            problems[oid] = f"- {oid}: a {old.get('type')} node cannot become a {rep.get('type')} node; keep its type"
            continue
        if rep is None and nn.get("type") != old.get("type"):
            rec["outcome"] = "bad_type"
            problems[oid] = f"- {oid}: the rewritten node must keep its type ({old.get('type')})"
            continue
        kept = rep if rep is not None else nn
        old_words = _content(old.get("question", ""), SPLIT_FREE)
        old_all = _content(old.get("question", ""))  # every content word: "rust or wear" is not "rust" ("wear" is a split-free verb)
        if rep is None and _content(kept.get("question", "")) == old_all:
            # the old question returned unchanged: the first new node of the same type about the same participants,
            # restating part of the old one, takes over the old id
            def same_subject(n):
                if old.get("type") in ("entity", "background", "absent"):
                    return True
                return sorted(_of_ids(n)) == sorted(_of_ids(old))
            def same_kind(n):
                # an entity hands its id only to an entity of the same kind (10-10: "souvenir tables and craft tables
                # with supplies" handed its id to "craft supplies", and every node about the tables then meant supplies)
                if old.get("type") != "entity":
                    return True
                return bool(_content(f"{n.get('class', '')} {n.get('plural', '')}") & _content(f"{old.get('class', '')} {old.get('plural', '')}"))
            first = next((n for n in extra if n.get("type") == old.get("type") and same_subject(n) and same_kind(n) and _from_old(n, old, plan["nodes"] + extra)), None)
            if first is not None:
                alias = first["id"]
                kept = nn = dict(nn, **{k: first[k] for k in ("desc", "question", "class", "plural", "count", "count_exact") if first.get(k) is not None})
                extra = [n for n in extra if n is not first]
                for n in extra:
                    of = n.get("of")
                    if isinstance(of, list):
                        n["of"] = [oid if u == alias else u for u in of]
                    elif of == alias:
                        n["of"] = oid
                rec["alias"] = alias
                info.setdefault("_alias", {})[alias] = oid
        # new nodes: an id the plan has is a reference to that node when it asks the same, else the id is renamed; a node
        # the plan already asks about the same subject is a reference, not an addition
        pool_subject = _subject_pool(plan["nodes"], old, {oid})
        taken = set(by_id) | {n["id"] for n in new_nodes}
        fresh, ren = [], {}
        for n in extra:
            if n.get("type") not in ("entity", "attribute", "state", "action", "relation", "absent"):
                continue
            if n["id"] in by_id:
                hit = _covered_by(n, [by_id[n["id"]]], plan["nodes"] + extra)
                if hit:
                    refs.append(hit)
                    info["dups"] += 1
                    continue
                k = 2
                while f"{n['id']}_{k}" in taken or f"{n['id']}_{k}" in {m["id"] for m in fresh}:
                    k += 1
                ren[n["id"]] = f"{n['id']}_{k}"
                n["id"] = ren[n["id"]]
            elif n["id"] in taken or n["id"] in {m["id"] for m in fresh}:
                k = 2
                while f"{n['id']}_{k}" in taken or f"{n['id']}_{k}" in {m["id"] for m in fresh}:
                    k += 1
                ren[n["id"]] = f"{n['id']}_{k}"
                n["id"] = ren[n["id"]]
            hit = None
            if _dup_of_existing(n, plan["nodes"] + new_nodes + fresh):
                hit = "dup"
            else:
                hit = _covered_by(n, [m for m in pool_subject + new_nodes + fresh if m.get("id") != oid], plan["nodes"] + new_nodes + extra)
            if hit:
                if hit != "dup":
                    refs.append(hit)
                info["dups"] += 1
                continue
            fresh.append(n)
        for n in fresh:  # a renamed node keeps the references of the group's other new nodes
            of = n.get("of")
            if isinstance(of, list):
                n["of"] = [ren.get(u, u) for u in of]
            elif of in ren:
                n["of"] = ren[of]
        extra = fresh
        kept_words = _content(kept.get("question", ""), SPLIT_FREE)
        if rep is None and _content(kept.get("question", "")) == old_all:
            rec["outcome"] = "unchanged"
            if extra or oid in flagged:
                problems[oid] = f"- {oid}: the node still asks everything the old question asked (\"{old.get('question')}\"); shorten it to one fact and give each other fact a new node"
            continue
        if rep is None and len(str(kept.get("question", "")).split()) > len(str(old.get("question", "")).split()):
            rec["outcome"] = "longer"
            problems[oid] = f"- {oid}: the rewritten question \"{kept.get('question')}\" is longer than the old one; it must keep part of the old question"
            continue
        names_old = _qnames(plan["nodes"] + extra, _of_ids(old)) if old.get("type") not in ("entity", "background") else set()
        added_words = kept_words - old_words - names_old
        if len(added_words) > 1:
            rec["outcome"] = "adds"
            problems[oid] = f"- {oid}: the rewritten question \"{kept.get('question')}\" says what the old question \"{old.get('question')}\" does not ({', '.join(_surface(added_words, kept.get('question', '')))}); it must keep part of the old question"
            continue
        # participants: a kept child node the planner put on other entities (a correction: "reaching for the cookie jar",
        # not the coffee table) takes them when they are entities of the plan or of this rewrite; otherwise the rewrite
        # does not apply (10-10: 16 kept rewrites had new participants and the old ones stayed beside the new question)
        new_of = None
        if rep is None and old.get("type") in ("attribute", "state", "relation", "action") and kept.get("of") is not None:
            kof = list(dict.fromkeys(u for u in (kept["of"] if isinstance(kept["of"], list) else [kept["of"]]) if u and u != "bg"))
            if sorted(kof) != sorted(_of_ids(old)):
                ents = {n["id"] for n in plan["nodes"] + new_nodes + extra if n.get("type") == "entity"}
                single = old.get("type") in ("attribute", "state")
                if not (kof and all(u in ents for u in kof) and (len(kof) == 1 if single else len(kof) >= 2)):
                    rec["outcome"] = "bad_of"
                    problems[oid] = f"- {oid}: the rewritten node is about {', '.join(kof) or 'nothing'}, not entities of the plan; keep its participants ({', '.join(_of_ids(old))}) or add the entity it is about as a new node"
                    continue
                new_of = kof[0] if single else kof
        # the kept node must not still ask what one of its new nodes asks (the bundled node left beside its parts)
        view = [dict(old, **{k: kept[k] for k in ("desc", "question", "class", "plural") if kept.get(k)}) if m["id"] == oid else m for m in plan["nodes"]] + extra
        if old.get("type") != "background":
            subj = {oid} if old.get("type") == "entity" else set(_of_ids(kept)) | {oid}
            kn = kept_words - (_qnames(view, _of_ids(kept)) if old.get("type") != "entity" else set())
            twin = None
            for n in extra:
                if n.get("type") == "entity" or not (set(_of_ids(n)) & subj):
                    continue
                wp = _content(n.get("question", ""), SPLIT_FREE) - _qnames(view, _of_ids(n))
                if wp and wp <= kn:
                    twin = n
                    break
            if twin is not None:
                rec["outcome"] = "repeats"
                problems[oid] = f"- {oid}: the node \"{kept.get('question')}\" still asks what new node {twin['id']} (\"{twin.get('question')}\") asks; keep one fact in the node and every other fact in exactly one new node"
                continue
        # nothing of the old question may be lost (frame positions and what a still image cannot show excepted)
        cues, core_text, phrase = _nonvisual(old.get("question", ""), _class_names(plan["nodes"], _of_ids(old) + [oid]))
        core_words = _content(core_text, SPLIT_FREE) - cues
        cover = set(kept_words) | names_old
        for n in extra:
            cover |= _content(n.get("question", ""), SPLIT_FREE)
            if n.get("type") == "entity":
                cover |= _content(f"{n.get('class', '')} {n.get('plural', '')} {n.get('desc', '')}")
        known = {m["id"]: m for m in plan["nodes"] + new_nodes}

        def ref_ok(m):
            """What may stand in for part of the old node: for a background only entity and absent nodes (an object it
            lists must be asked as present); for an entity also its own child nodes; for other nodes anything."""
            if old.get("type") == "background":
                return m.get("type") in ("entity", "absent")
            if old.get("type") == "entity":
                return m.get("type") in ("entity", "absent") or oid in _of_ids(m)
            return True

        for r in refs:
            if r in known and ref_ok(known[r]):
                cover |= _content(known[r].get("question", ""), SPLIT_FREE)
                if known[r].get("type") == "entity":
                    cover |= _content(f"{known[r].get('class', '')} {known[r].get('plural', '')}")
        for m in pool_subject + _subject_pool(new_nodes, old, {oid}):
            cover |= _content(m.get("question", ""), SPLIT_FREE)
            if m.get("type") == "entity":
                cover |= _content(f"{m.get('class', '')} {m.get('plural', '')}")
        lost = core_words - cover - LOST_FREE - (MOD_FREE if (cues or phrase) else set())
        allow = LOST_OK_NEEDS_SPLIT if (extra or refs) else 0
        if lost & {_cstem(w) for w in NEG} or len(lost) > allow:
            rec["outcome"] = "lost"
            rec["lost"] = sorted(lost)
            problems[oid] = f"- {oid}: no node asks these words of the old question \"{old.get('question')}\": {', '.join(_surface(lost, old.get('question', '')))}"
            continue
        # every new node restates part of the old one or is quoted from the description
        narrowed = dict(old, **{k: kept[k] for k in ("desc", "question") if kept.get(k)}, **({"of": new_of} if new_of is not None else {})) if rep is None else rep
        pool = [narrowed if n["id"] == oid else n for n in plan["nodes"]] + new_nodes + extra + ([rep] if rep is not None else [])
        cand = extra + ([rep] if rep is not None else [])
        for n in cand:
            n["evidence"] = str(n.get("evidence", "")).replace("...", " ").replace("…", " ")
        own = {oid} | set(_of_ids(old))

        def inherited(n):
            """A split-out about the same entity as the node it came from, with its words quoted from the description:
            the old node already names the participant, so the quote need not."""
            ids = set(_of_ids(n))
            return n.get("type") in ("attribute", "state") and bool(ids) and ids <= own and _ev_in_prompt(n.get("evidence", ""), prompt)

        bad = [n for n in cand if not (_from_old(n, old, pool) or inherited(n) or grounded(n, prompt, pool))]
        if not bad:
            mute = [(n, _unnamed(n, pool)) for n in ([narrowed] if rep is None else []) + cand]
            mute = [(n, u) for n, u in mute if u]
            if mute:
                n, u = mute[0]
                rec["outcome"] = "unnamed"
                problems[oid] = f"- {oid}: the {n.get('type')} node {n.get('id')} (\"{n.get('question')}\") does not name {', '.join(u)}; a relation or action question names every entity it is about"
                continue
        if bad:
            rec["outcome"] = "ungrounded"
            info["ungrounded"] += [dict(id=n.get("id"), type=n.get("type"), of=n.get("of"), question=n.get("question"), evidence=n.get("evidence"), group=oid) for n in bad]
            b = bad[0]
            problems[oid] = f"- {oid}: new node {b.get('id')} (\"{b.get('question')}\") restates nothing of the old question and its evidence \"{b.get('evidence')}\" is not a verbatim quote of the description that names {', '.join(_of_ids(b)) or 'it'}"
            continue
        groups[oid] = dict(old=json.loads(json.dumps(old)), retyped=rep is not None, members=[n["id"] for n in extra] + ([oid] if rep is not None else []), round=round_no)
        rec.update(outcome="kept", new=[n["id"] for n in extra], refs=refs, lost=sorted(lost))
        if rep is not None:
            plan["nodes"] = [n for n in plan["nodes"] if n["id"] != oid]
            by_id.pop(oid, None)
        else:
            for k in ("desc", "question"):
                if kept.get(k):
                    old[k] = strip_frame_positions(kept[k]) if old.get("type") == "entity" else kept[k]
            if new_of is not None:
                old["of"] = new_of
            if old.get("type") == "entity":
                for k in ("class", "plural"):
                    if kept.get(k):
                        old[k] = str(kept[k]).strip().lower()
                for k in ("count", "count_exact"):
                    if kept.get(k) is not None:
                        old[k] = kept[k]
        for n in cand:
            n["_grp"] = oid
        new_nodes += cand
    return groups, new_nodes, problems


def _settle(trial, base_for_pre, groups, grp_of, added_ids, model_edges, count_undo, min_side, max_nodes, problems, related=None, proposed=None):
    """Validate trial (plan dict, changed in place); a group one of whose nodes breaks a rule is reverted whole (its old
    node back, its new nodes out) and its error kept as a problem; a broken edge the planner proposed is dropped; an
    addition that breaks a rule is dropped; a count setting that breaks a rule is undone. `related` collects, per
    reverted group, untouched nodes the planner should rewrite with it in the repair round. Returns (reverted ids, error)."""
    related = {} if related is None else related
    pre = set(planner_errors(base_for_pre))
    order0 = [n["id"] for n in base_for_pre["nodes"]]
    reverted = []
    # an attribute or state the passes did not touch that places its entity on an entity a group added becomes a
    # relation of the two, as the first pass does (lift_states_to_relations); it goes back with that group
    lifted = {}
    for n in trial["nodes"]:
        if n["id"] in grp_of or n["id"] in groups or n.get("type") not in ("attribute", "state") or not isinstance(n.get("of"), str):
            continue
        hit = _placed_on_other(trial, n)
        if hit and hit[0] != n["of"] and hit[0] in grp_of:
            lifted.setdefault(grp_of[hit[0]], []).append(json.loads(json.dumps(n)))
            n["type"], n["of"] = "relation", [n["of"], hit[0]]

    def revert(oid, why=None):
        g = groups[oid]
        members = set(g["members"])
        nodes = [n for n in trial["nodes"] if n["id"] not in members]
        if g["retyped"]:
            pos = min(order0.index(oid) if oid in order0 else len(nodes), len(nodes))
            nodes.insert(pos, g["old"])
        else:
            nodes = [g["old"] if n["id"] == oid else n for n in nodes]
        back = {n["id"]: n for n in lifted.pop(oid, [])}
        trial["nodes"] = [back.get(n["id"], n) for n in nodes]
        reverted.append(oid)
        if why:
            problems[oid] = f"- {oid}: {why}"

    for _ in range(80):
        have = {n["id"] for n in trial["nodes"]}
        for oid, g in groups.items():
            if oid not in reverted and not all(m in have for m in g["members"]):
                gone = [m for m in g["members"] if m not in have]
                revert(oid, f"new node {gone[0]} is invalid: an attribute or state needs one entity id in \"of\" (the background has no attributes)")
        have = {n["id"] for n in trial["nodes"]}
        trial["edges"] = [e for e in trial["edges"] if e.get("from") in have and e.get("to") in have]
        errs, _w = Plan(trial).validate(min_side=min_side, max_nodes=max_nodes)
        if not errs:
            errs = [e for e in planner_errors(trial) if e not in pre]
        if not errs:
            return reverted, None
        fixed = False
        if any("too many nodes" in e for e in errs):
            live = [o for o in groups if o not in reverted]
            if live:
                revert(live[-1], "the plan has too many nodes")
                continue
            if added_ids & have:
                last = [n["id"] for n in trial["nodes"] if n["id"] in added_ids][-1]
                trial["nodes"] = [n for n in trial["nodes"] if n["id"] != last]
                continue
        for e in errs:
            m = re.match(r"(node|edge) ([^:]+?)(->([^:]+))?:", e)
            if not m:
                continue
            if m.group(1) == "edge":
                a, b = m.group(2).strip(), (m.group(4) or "").strip()
                if (a, b) in model_edges:
                    n0 = len(trial["edges"])
                    trial["edges"] = [x for x in trial["edges"] if not (x.get("from") == a and x.get("to") == b)]
                    fixed |= len(trial["edges"]) < n0
                    continue
                names = [a, b]
            else:
                names = [m.group(2).strip()]
            for b in names:
                g = grp_of.get(b) if b in grp_of else (b if b in groups else None)
                if g and g not in reverted:
                    why = e.split(":", 1)[1].strip() if ":" in e else e
                    node = next((n for n in trial["nodes"] if n["id"] == b), {})
                    ents = {n["id"] for n in trial["nodes"] if n.get("type") == "entity"}
                    gone = [x for x in _of_ids(node) if x not in ents]
                    src = [(x, (proposed or {}).get(x)) for x in gone if (proposed or {}).get(x) not in (None, g)]
                    if src:
                        why = f"node {b} needs entity {src[0][0]}, a new node of the rewrite of {src[0][1]}, which was not applied; fix the rewrite of {src[0][1]} and give this one again"
                    revert(g, why)
                    fixed = True
                elif b in added_ids and b in {n["id"] for n in trial["nodes"]}:
                    trial["nodes"] = [n for n in trial["nodes"] if n["id"] != b]
                    fixed = True
                elif b in count_undo:
                    for n in trial["nodes"]:
                        if n["id"] == b:
                            n["count"], n["count_exact"] = count_undo[b]
                            if n["count_exact"] is None:
                                n.pop("count_exact", None)
                    count_undo.pop(b)
                    fixed = True
                else:
                    # a node the passes did not touch now places itself on an entity a group added or changed: that
                    # group goes back
                    w = re.search(r"'[^']*\.\.\. ([a-z]+)'", e)
                    node = next((n for n in trial["nodes"] if n["id"] == b), {})
                    own = node.get("of") if node.get("type") in ("attribute", "state") else b
                    hits = []
                    if w:
                        for n in trial["nodes"]:
                            if n.get("type") != "entity" or n["id"] == own:
                                continue
                            desc = re.sub(r"[^a-z ]", " ", str(n.get("desc", "")).lower()).split()
                            words = {str(n.get("class", "")).lower().strip(), str(n.get("plural", "")).lower().strip()} | ({desc[-1]} if desc else set())
                            if w.group(1) in words or w.group(1).rstrip("s") in words:
                                hits.append(n["id"])
                    for hit in hits:
                        g = grp_of.get(hit) or (hit if hit in groups else None)
                        if g and g not in reverted:
                            revert(g, f"after it node {b} places itself on entity {hit} (\"{e.split(':', 1)[1].strip()[:120]}\"); rewrite {b} too: keep its presence and look and make the placement a relation node")
                            related.setdefault(g, set()).add(b)
                            fixed = True
                            break
            if m.group(1) == "edge" and not fixed:
                a, b = m.group(2).strip(), (m.group(4) or "").strip()
                n0 = len(trial["edges"])
                trial["edges"] = [x for x in trial["edges"] if not (x.get("from") == a and x.get("to") == b)]
                fixed |= len(trial["edges"]) < n0
        if not fixed:
            # no change can be blamed by name (an edge cycle names a node, not the edge): first drop the edges the
            # planner proposed, then take back the latest change and try again, so one bad group does not cost the
            # prompt everything else
            live = [o for o in groups if o not in reverted]
            have = {n["id"] for n in trial["nodes"]}
            mine = [e for e in trial["edges"] if (e.get("from"), e.get("to")) in model_edges]
            if mine:
                trial["edges"] = [e for e in trial["edges"] if (e.get("from"), e.get("to")) not in model_edges]
                model_edges.clear()
            elif live:
                revert(live[-1], "the plan became invalid with this rewrite: " + errs[0][:160])
            elif added_ids & have:
                last = [n["id"] for n in trial["nodes"] if n["id"] in added_ids][-1]
                trial["nodes"] = [n for n in trial["nodes"] if n["id"] != last]
            elif count_undo:
                b = next(iter(count_undo))
                for n in trial["nodes"]:
                    if n["id"] == b:
                        n["count"], n["count_exact"] = count_undo[b]
                        if n["count_exact"] is None:
                            n.pop("count_exact", None)
                count_undo.pop(b)
            else:
                return reverted, "validate: " + "; ".join(errs[:3])
    return reverted, "validate: no convergence"


def _add_new(trial, new_nodes, edges_in, W, H, min_side):
    """Normalize the new nodes and put them and the planner's edges into trial (changed in place); a new attribute or
    state that places its entity on another entity becomes a relation, as in the first pass."""
    for n in new_nodes:
        _normalize_new(n, W, H, min_side)
    trial["nodes"] = trial["nodes"] + new_nodes
    trial["nodes"] = [n for n in trial["nodes"] if not (n.get("type") in ("attribute", "state") and n.get("of") in (None, "", "bg"))]
    for n in new_nodes:
        if n.get("type") in ("attribute", "state") and isinstance(n.get("of"), str):
            hit = _placed_on_other(trial, n)
            if hit and hit[0] != n.get("of"):
                n["type"], n["of"] = "relation", [n["of"], hit[0]]
    have_e = {(e.get("from"), e.get("to")) for e in trial["edges"]}
    for e in edges_in:
        key = (e.get("from"), e.get("to"))
        if key not in have_e:
            have_e.add(key)
            trial["edges"].append(dict(e, kind=e.get("kind", "support")))


def _model_edges(ans, known, alias):
    out = []
    for e in (ans.get("edges", []) or []):
        if not isinstance(e, dict):
            continue
        a, b = alias.get(e.get("from"), e.get("from")), alias.get(e.get("to"), e.get("to"))
        if a in known and b in known and a != b and e.get("kind", "support") == "support":
            out.append(dict(e, **{"from": a, "to": b}))
    return out


def atomize_pass(d, model=DEFAULT_MODEL, chat=api_chat, min_side=48, max_nodes=128, repair=True):
    """One-fact-per-node pass on an existing plan (measured against the CoReBench official checklists, 10-09: a third of
    the official questions were covered only by a requirement that also asked something else). The planner returns
    rewrites (a node shortened to one fact, or retyped under the same id, plus the nodes taking over the rest), additions
    (comparisons and count relations), count settings and deletions (what a still image cannot show). Rewrites are
    accepted by _apply_rewrites (a group is all or nothing; nothing of the old question may be lost); a deletion only when
    _deletable holds; a group that later breaks a plan rule is reverted whole. Rewrites that were not applied go back to
    the planner once with the reason (repair round) and are checked the same way. Returns (plan dict, info)."""
    info = dict(rewrites=0, rewrites_kept=0, new_nodes=0, added=0, set_count=0, deleted=[], kept_delete=[], ungrounded=[], reverted=[], dropped=0, dups=0,
                groups=[], repair=None, error=None)
    prompt = d["prompt"].strip()
    W, H = d.get("canvas", [1024, 1024])
    listing = "\n".join(f"{n['id']} [{n['type']}]{' of ' + json.dumps(n['of']) if n.get('of') else ''}: {n.get('question', n.get('desc', ''))}" for n in d["nodes"])
    edges = ", ".join(f"{e.get('from')} -> {e.get('to')}" for e in d.get("edges", []) if e.get("kind", "support") == "support") or "none"
    flagged = ", ".join(likely_bundled(d)) or "none found by the simple check, still check every node"
    msgs = [
        {"role": "system", "content": SYSTEM.format(W=W, H=H)},
        {"role": "user", "content": EXAMPLE_USER},
        {"role": "assistant", "content": _scale_example(W, H)},
        {"role": "user", "content": ATOMIZE_USER.format(prompt=prompt, nodes=listing, edges=edges, flagged=flagged)},
    ]
    out = chat(msgs, model=model, tag=f"atomize:{d['id']}", max_tokens=8000, temperature=0.0)
    try:
        ans = parse_json(out)
    except Exception:
        # 10-10 full run: 3 of 462 replies drifted into reasoning inside the JSON (two ran to 8000 tokens); ask once more
        msgs = msgs[:-1] + [{"role": "user", "content": msgs[-1]["content"] + "\nReply with the JSON object only: no reasoning, no explanation, nothing before or after it."}]
        out = chat(msgs, model=model, tag=f"atomize_retry:{d['id']}", max_tokens=8000, temperature=0.0)
        try:
            ans = parse_json(out)
            info["parse_retry"] = True
        except Exception:
            return d, dict(info, error="parse", raw=str(out)[:4000])
    info["raw"] = json.loads(json.dumps(ans))  # as the planner returned it; the processing below changes its own copy
    base = json.loads(json.dumps(d))
    by_id = {n["id"]: n for n in base["nodes"]}
    ent_ids = {n["id"] for n in base["nodes"] if n.get("type") == "entity"}
    # deletions: only what a still image cannot show, only child nodes, judged on the node's own words
    for x in ans.get("delete", []) or []:
        if isinstance(x, dict) and x.get("id") in by_id and by_id[x["id"]].get("type") in ATOMIZE_DELETABLE:
            node = by_id[x["id"]]
            ok, lost = _deletable(node, base["nodes"])
            rec = dict(id=x["id"], why=str(x.get("why", ""))[:60], question=node.get("question"))
            if ok:
                base["nodes"] = [n for n in base["nodes"] if n["id"] != x["id"]]
                info["deleted"].append(rec)
            else:
                info["kept_delete"].append(dict(rec, lost=lost[:6]))
    by_id = {n["id"]: n for n in base["nodes"]}
    # count settings: a stated number for an entity group
    count_undo = {}
    for x in ans.get("set_count", []) or []:
        try:
            eid, k = x.get("id"), int(x.get("count"))
        except (TypeError, ValueError, AttributeError):
            continue
        if eid in ent_ids and eid in by_id and 1 <= k <= 50 and _ev_in_prompt(x.get("evidence", ""), prompt) and (_entity_terms([by_id[eid]], eid) & {_stem(w) for w in _words(x.get("evidence", ""))}):
            count_undo.setdefault(eid, (by_id[eid].get("count"), by_id[eid].get("count_exact")))
            by_id[eid]["count"], by_id[eid]["count_exact"] = k, True
            info["set_count"] += 1
    pre_base = json.loads(json.dumps(base))
    # rewrites
    groups, new_nodes, problems = _apply_rewrites(base, json.loads(json.dumps(ans.get("rewrite", []) or [])), prompt, info)
    # additions: comparisons and count relations between two entities, quoted from the description
    added_ids = set()
    for n in json.loads(json.dumps(ans.get("add", []) or [])):
        if not (isinstance(n, dict) and n.get("id") and n.get("type") == "relation" and len(set(_of_ids(n))) >= 2 and COMPARE.search(str(n.get("question", "")))):
            if isinstance(n, dict):
                info.setdefault("add_refused", []).append(dict(id=n.get("id"), type=n.get("type"), question=n.get("question")))
            continue
        if n["id"] in by_id or n["id"] in {m["id"] for m in new_nodes} or _dup_of_existing(n, base["nodes"] + new_nodes):
            info["dups"] += 1
            continue
        n["evidence"] = str(n.get("evidence", "")).replace("...", " ").replace("…", " ")
        if grounded(n, prompt, base["nodes"] + new_nodes + [n]) and in_scope(n.get("question", "")):
            new_nodes.append(n)
            added_ids.add(n["id"])
        else:
            info["ungrounded"].append(dict(id=n.get("id"), type=n.get("type"), of=n.get("of"), question=n.get("question"), evidence=n.get("evidence")))
    grp_of = {}
    for n in new_nodes:
        n.pop("evidence", None)
        g = n.pop("_grp", None)
        if g:
            grp_of[n["id"]] = g
    known = {n["id"] for n in base["nodes"]} | {n["id"] for n in new_nodes}
    edges_in = _model_edges(ans, known, info.pop("_alias", {}))
    trial = json.loads(json.dumps(base))
    _add_new(trial, new_nodes, edges_in, W, H, min_side)
    related = {}
    proposed = info.pop("_proposed", {})
    reverted, err = _settle(trial, pre_base, groups, grp_of, added_ids, {(e["from"], e["to"]) for e in edges_in}, count_undo, min_side, max_nodes, problems, related, proposed)
    if err:
        return d, dict(info, error=err, reverted=reverted)
    for g in reverted:
        for r in info["groups"]:
            if r["id"] == g and r.get("outcome") == "kept":
                r["outcome"] = "reverted"
    # repair round: the rewrites that were not applied go back once with the reason
    live_old = {n["id"]: n for n in trial["nodes"]}
    todo = {oid: msg for oid, msg in problems.items() if oid in live_old and (oid not in groups or oid in reverted)}
    info["problems"] = dict(todo)
    if repair and todo:
        rep_msgs = msgs + [{"role": "assistant", "content": str(out)}, {"role": "user", "content": ATOMIZE_REPAIR.format(problems="\n".join(todo[k] for k in todo))}]
        rinfo = dict(asked=sorted(todo), kept=0, error=None)
        info["repair"] = rinfo
        try:
            out2 = chat(rep_msgs, model=model, tag=f"atomize_repair:{d['id']}", max_tokens=8000, temperature=0.0)
            ans2 = parse_json(out2)
        except Exception as ex:
            rinfo["error"] = "repair: " + str(ex)[:120]
            ans2 = None
        if isinstance(ans2, dict):
            rinfo["raw"] = json.loads(json.dumps(ans2))
            base2 = json.loads(json.dumps(trial))
            pre2 = json.loads(json.dumps(trial))
            problems2 = {}
            also = {b for g in todo for b in related.get(g, ())}
            groups2, new2, p2 = _apply_rewrites(base2, json.loads(json.dumps(ans2.get("rewrite", []) or [])), prompt, info, only=set(todo) | also, round_no=2)
            problems2.update(p2)
            grp2 = {}
            for n in new2:
                n.pop("evidence", None)
                g = n.pop("_grp", None)
                if g:
                    grp2[n["id"]] = g
            known2 = {n["id"] for n in base2["nodes"]} | {n["id"] for n in new2}
            e2 = _model_edges(ans2, known2, info.pop("_alias", {}))
            trial2 = json.loads(json.dumps(base2))
            _add_new(trial2, new2, e2, W, H, min_side)
            rev2, err2 = _settle(trial2, pre2, groups2, grp2, set(), {(e["from"], e["to"]) for e in e2}, {}, min_side, max_nodes, problems2, None, info.pop("_proposed", {}))
            if err2:
                rinfo["error"] = err2
            else:
                trial = trial2
                for g in rev2:
                    for r in info["groups"]:
                        if r["id"] == g and r.get("round") == 2 and r.get("outcome") == "kept":
                            r["outcome"] = "reverted"
                rinfo["kept"] = sum(1 for o in groups2 if o not in rev2)
                rinfo["problems_left"] = [problems2[k] for k in problems2]
                for oid in groups2:
                    if oid not in rev2:
                        groups[oid] = groups2[oid]
                        grp_of.update({k: v for k, v in grp2.items() if v == oid})
                        if oid in reverted:
                            reverted.remove(oid)
                reverted += [o for o in rev2]
    reverted = list(dict.fromkeys(reverted))
    # cleanup on the final plan: a deletion refused at first is taken when the rewrites now ask everything else the node
    # said; a node shortened without new nodes that only repeats other nodes goes
    info["cleanup"] = []
    for x in list(info["kept_delete"]):
        node = next((n for n in trial["nodes"] if n["id"] == x["id"]), None)
        if node is not None and node.get("type") in ATOMIZE_DELETABLE and _deletable(node, trial["nodes"])[0]:
            trial["nodes"] = [n for n in trial["nodes"] if n["id"] != x["id"]]
            info["kept_delete"].remove(x)
            info["deleted"].append(dict(x, late=True))
    # a child node that only bundles nodes about the same participants goes, one at a time (of two that ask the same,
    # one stays), when this pass made it or one of its parts (10-10 full run: "Does the cleaning cart hold a mop and a
    # bucket of soapy water?", untouched, stayed beside "cart_mop" and "cart_bucket" that the entity's rewrite made)
    touched = {oid for oid, g in groups.items() if oid not in reverted} | {i for i, g in grp_of.items() if g not in reverted}
    for node in [n for n in trial["nodes"] if n.get("type") in ("attribute", "state", "relation", "action")]:
        if not _redundant(node, trial["nodes"]):
            continue
        def own(m):
            w = _content(m.get("question", ""), SPLIT_FREE)
            return w if m.get("type") in ("relation", "action") else w - _qnames(trial["nodes"], _of_ids(m))
        words = own(node)
        parts = [m["id"] for m in trial["nodes"] if m["id"] != node["id"] and sorted(_of_ids(m)) == sorted(_of_ids(node)) and own(m) <= words]
        if node["id"] in touched or any(p in touched for p in parts):
            trial["nodes"] = [n for n in trial["nodes"] if n["id"] != node["id"]]
            info["cleanup"].append(dict(id=node["id"], question=node.get("question"), parts=parts[:6]))
    have = {n["id"] for n in trial["nodes"]}
    trial["edges"] = [e for e in trial["edges"] if e.get("from") in have and e.get("to") in have]
    info["reverted"] = reverted
    info["rewrites_kept"] = sum(1 for oid in groups if oid not in reverted)
    info["new_nodes"] = sum(1 for i in grp_of if i in have and grp_of[i] not in reverted and i not in groups)
    info["added"] = sum(1 for i in added_ids if i in have)
    changed = info["rewrites_kept"] or info["added"] or info["set_count"] or info["deleted"] or info["cleanup"]
    if changed:
        trial["meta"] = dict(trial.get("meta", {}), atomize=dict(rewrites=info["rewrites_kept"], new_nodes=info["new_nodes"], added=info["added"], set_count=info["set_count"], deleted=len(info["deleted"]),
                                                               reverted=len(reverted), ungrounded=len(info["ungrounded"]), repair=(info["repair"] or {}).get("kept", 0), cleanup=len(info["cleanup"])))
    return (trial if changed else d), info


def _grow_box(box, W, H, min_side):
    """Expand a too-thin entity box around its center to the minimum side, inside the canvas; rounded ints."""
    def grow(a, b, limit):
        if not 0 < b - a < min_side:
            return a, b
        c = (a + b) / 2
        a, b = max(0.0, c - min_side / 2), min(float(limit), c + min_side / 2)
        if b - a < min_side:  # clamped at one edge: push the other side
            a, b = (0.0, min(float(limit), float(min_side))) if a <= 0 else (max(0.0, b - min_side), b)
        return a, b

    x0, y0, x1, y1 = box
    x0, x1 = grow(x0, x1, W)
    y0, y1 = grow(y0, y1, H)
    return [int(round(v)) for v in (x0, y0, x1, y1)]


def _merge_added(d, add, min_side=48, max_nodes=64):
    """Merge a planner's additional nodes/edges into plan dict d: normalize them like the first pass, then drop
    invalid new nodes one by one until the plan validates. Returns (plan dict, n_added, n_dropped, errors left)."""
    W, H = d.get("canvas", [1024, 1024])
    ids = {n["id"] for n in d["nodes"]}
    new_nodes = [n for n in add.get("nodes", []) if isinstance(n, dict) and n.get("id") and n["id"] not in ids and n.get("type") in PLANNER_TYPES_FOR_COVER]
    new_edges = [e for e in add.get("edges", []) if isinstance(e, dict)]
    if not new_nodes:
        return d, 0, 0, []
    base = json.loads(json.dumps(d))
    for n in new_nodes:
        if n.get("type") == "entity":
            n["class"] = str(n.get("class", "")).strip().lower()
            n["plural"] = str(n.get("plural", "")).strip().lower() or None
            try:
                n["count"] = int(n.get("count", 1) or 1)
            except (TypeError, ValueError):
                n["count"] = 1
            if "count_exact" in n and not isinstance(n["count_exact"], bool):
                n["count_exact"] = str(n["count_exact"]).strip().lower() not in ("false", "0", "no")
            for k in ("question", "desc"):
                if n.get(k):
                    n[k] = strip_frame_positions(n[k])
            if isinstance(n.get("box"), list) and len(n["box"]) == 4:
                try:
                    n["box"] = _grow_box([float(x) for x in n["box"]], W, H, min_side)
                except (TypeError, ValueError):
                    pass
        if n.get("type") in ("action", "relation") and isinstance(n.get("of"), list):
            n["of"] = list(dict.fromkeys(u for u in n["of"] if u != "bg"))
            if len(n["of"]) == 1:
                n["type"], n["of"] = "state", n["of"][0]
    seen_ids = set()
    new_nodes = [n for n in new_nodes if not (n["id"] in seen_ids or seen_ids.add(n["id"]))]
    have = {(e.get("from"), e.get("to")) for e in base["edges"]}
    kept_edges = []
    for e in new_edges:  # an edge the plan already has, or one listed twice, must not get a new node dropped
        key = (e.get("from"), e.get("to"))
        if key not in have:
            have.add(key)
            kept_edges.append(dict(e, kind=e.get("kind", "support")))
    trial = dict(base, nodes=base["nodes"] + new_nodes, edges=base["edges"] + kept_edges)
    trial["nodes"] = [n for n in trial["nodes"] if not (n.get("type") in ("attribute", "state") and n.get("of") in (None, "", "bg"))]
    lift_states_to_relations(trial)
    pre = set(planner_errors(base))  # complaints the base plan already had are not the additions' fault
    dropped = 0
    seen_errs = []
    for _ in range(len(new_nodes) + 3):
        errs, _w = Plan(trial).validate(min_side=min_side, max_nodes=max_nodes)
        if not errs:
            errs = [e for e in planner_errors(trial) if e not in pre]
        if not errs:
            break
        seen_errs += errs
        bad = set()
        for e in errs:
            m = re.match(r"(node|edge) ([^:]+?)(->([^:]+))?:", e)
            if m:
                bad.add(m.group(2).strip())
                if m.group(4):
                    bad.add(m.group(4).strip())
        added_ids = {n["id"] for n in new_nodes}
        if "too many nodes" in " ".join(errs):
            bad |= set(list(added_ids)[-3:])
        kill = bad & added_ids
        if not kill:  # an error that names no added node: drop the newest addition and try again
            kill = {new_nodes[-1]["id"]}
        trial["nodes"] = [n for n in trial["nodes"] if n["id"] not in kill]
        trial["edges"] = [e for e in trial["edges"] if e.get("from") not in kill and e.get("to") not in kill]
        new_nodes = [n for n in new_nodes if n["id"] not in kill]
        dropped += len(kill)
        if not new_nodes:
            return d, 0, dropped, seen_errs
    else:
        return d, 0, len(new_nodes), seen_errs
    return trial, len(new_nodes), dropped, seen_errs


PLANNER_TYPES_FOR_COVER = ("entity", "attribute", "state", "action", "relation", "absent")


def audit_plan(plan, model=DEFAULT_MODEL, chat=api_chat):
    """Three calls for the planner gate: questions that go beyond the description or bundle two facts; the description's
    facts extracted without seeing the plan; and a match of those facts against the plan's requirements."""
    qs = "\n".join(f"{v} [{plan.typ(v)}]: {plan.nodes[v]['question']}" for v in plan.order if plan.typ(v) not in ("count", "pair", "quality", "background"))
    out1 = chat([{"role": "user", "content": AUDIT_BEYOND.format(prompt=plan.prompt, questions=qs)}], model=model, tag=f"audit_beyond:{plan.id}", max_tokens=4000, temperature=0.0)
    out2 = chat([{"role": "user", "content": AUDIT_FACTS.format(prompt=plan.prompt)}], model=model, tag=f"audit_facts:{plan.id}", max_tokens=4000, temperature=0.0)
    beyond, facts, matches = {}, [], {}
    try:
        beyond = {k: v for k, v in parse_json(out1).items() if isinstance(v, dict)}
    except Exception:
        pass
    n_raw = 0
    try:
        raw = [str(f) for f in parse_json(out2).get("facts", []) if str(f).strip()]
        n_raw = len(raw)
        facts = [f for f in raw if in_scope(f)]
    except Exception:
        pass
    if facts:
        full = add_generated_nodes(Plan(plan.to_dict()))
        req_nodes = [v for v in full.order if full.typ(v) not in ("pair", "quality")]
        descs = "\n".join(f"R{i + 1}. [{full.typ(v)}] {full.nodes[v]['desc']}" for i, v in enumerate(req_nodes))
        flist = "\n".join(f"F{i + 1}. {f}" for i, f in enumerate(facts))
        for _try in range(2):
            out3 = chat([{"role": "user", "content": AUDIT_MATCH.format(facts=flist, descs=descs)}], model=model, tag=f"audit_match:{plan.id}", max_tokens=4000, temperature=0.0 if _try == 0 else 0.3)
            try:
                matches = parse_json(out3)
            except Exception:
                matches = {}
            if len(_missing_facts(facts, matches)) <= max(2, len(facts) // 2):
                break
    missing = _missing_facts(facts, matches)
    match_suspect = bool(facts) and len(missing) > max(2, len(facts) // 2)
    n_q = len([v for v in plan.order if plan.typ(v) not in ("count", "pair", "quality", "background")])
    n_beyond = sum(1 for v in beyond.values() if v.get("beyond"))
    n_multi = sum(1 for v in beyond.values() if v.get("multi"))
    return dict(n_questions=n_q, n_beyond=n_beyond, n_multi=n_multi, n_facts=len(facts), n_facts_raw=n_raw, match_suspect=match_suspect, beyond={k: v for k, v in beyond.items() if v.get("beyond")}, multi={k: v for k, v in beyond.items() if v.get("multi")}, missing=missing)


def load_or_plan(prompt, pid, plan_dir, **kw):
    os.makedirs(plan_dir, exist_ok=True)
    path = os.path.join(plan_dir, pid + ".json")
    if os.path.exists(path):
        return Plan.load(path), []
    d, errs, warns = plan_prompt(prompt, pid, **kw)
    if errs:
        with open(os.path.join(plan_dir, pid + ".failed.json"), "w", encoding="utf-8") as f:
            json.dump(dict(errors=errs, plan=d), f, ensure_ascii=False, indent=1)
        return None, errs
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    return Plan(d), []


def full_plan(plan):
    """Planner output -> plan with generated count, pair and quality nodes."""
    return add_generated_nodes(Plan(plan.to_dict()))
