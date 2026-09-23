from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from test0.dag import ACTION, BACKGROUND, DAG, DRAFT_TYPES, ENTITY, OBJECT, SCENE

from .actions import ACTIONS, Action
from .reward import weight

POLICY_DEFAULTS = {
    "POLICY_BACKEND": "mock",
    "POLICY_MODEL": "Qwen/Qwen3-8B",
    "POLICY_BASE_URL": "",
    "POLICY_API_KEY": "",
    "POLICY_LORA_R": 16,
    "POLICY_LORA_ALPHA": 32,
    "POLICY_MAX_NEW_TOKENS": 2048,
    "POLICY_TEMPERATURE": 0.9,
    "POLICY_TOP_P": 0.95,
    "POLICY_MAX_LEN": 6144,
    "TRAIN_LR": 1e-5,
    "TRAIN_CLIP_EPS": 0.2,
    "TRAIN_KL_BETA": 0.04,
    "TRAIN_MICRO_BATCH": 2,
    "MOCK_POLICY_LR": 0.6,
    "MOCK_POLICY_TEMPERATURE": 1.0,
}


def setting(cfg: Any, key: str) -> Any:
    value = cfg.get(key) if hasattr(cfg, "get") else None
    return POLICY_DEFAULTS[key] if value is None else value


@dataclass
class Sample:
    text: str
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainSample:
    prompt: str
    completion: str
    advantage: float
    reward: float
    kind: str
    group: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "completion": self.completion, "advantage": self.advantage, "reward": self.reward, "kind": self.kind, "group": self.group, "meta": self.meta}


class Policy:
    name = "base"
    trainable = False

    def sample(self, system: str, user: str, n: int, kind: str, context: Dict[str, Any]) -> List[Sample]:
        raise NotImplementedError

    def update(self, batch: Sequence[TrainSample]) -> Dict[str, Any]:
        return {"updated": False}

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "policy.json").write_text(json.dumps({"name": self.name}), encoding="utf-8")

    def load(self, path: str) -> None:
        return None


def _softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    if not values:
        return []
    scaled = [v / max(temperature, 1e-6) for v in values]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


DRAFT_CHOICES = {
    BACKGROUND: ["correct", "wrong_color", "malformed"],
    ENTITY: ["correct", "wrong_color", "wrong_count", "wrong_pose", "malformed", "off_bbox"],
    OBJECT: ["correct", "wrong_color", "wrong_count", "malformed", "off_bbox"],
    ACTION: ["correct", "missing_participant", "wrong_pose", "wrong_color"],
}
EDIT_CHOICES = ["stop", "edit_blamed", "edit_scene", "adjust_final", "reuse", "edit_random", "delete_random", "split_random"]


class MockPolicy(Policy):
    name = "mock"
    trainable = True

    def __init__(self, cfg: Any, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(int(seed))
        self.lr = float(setting(cfg, "MOCK_POLICY_LR"))
        self.temperature = float(setting(cfg, "MOCK_POLICY_TEMPERATURE"))
        self.beta = float(setting(cfg, "TRAIN_KL_BETA"))
        self.logits: Dict[str, Dict[str, float]] = {}
        self.updates = 0

    def _table(self, key: str, choices: Sequence[str]) -> Dict[str, float]:
        table = self.logits.setdefault(key, {})
        for c in choices:
            table.setdefault(c, 0.0)
        return table

    def probabilities(self, key: str, choices: Sequence[str]) -> Dict[str, float]:
        table = self._table(key, choices)
        probs = _softmax([table[c] for c in choices], self.temperature)
        return dict(zip(choices, probs))

    def _draw(self, key: str, choices: Sequence[str]) -> str:
        probs = self.probabilities(key, choices)
        r = self.rng.random()
        acc = 0.0
        for c in choices:
            acc += probs[c]
            if r <= acc:
                return c
        return choices[-1]

    def sample(self, system: str, user: str, n: int, kind: str, context: Dict[str, Any]) -> List[Sample]:
        if kind == "draft":
            return [self._sample_draft(system, user, context) for _ in range(n)]
        return [self._sample_edit(system, user, context) for _ in range(n)]

    def _sample_draft(self, system: str, user: str, context: Dict[str, Any]) -> Sample:
        node = context["node"]
        dag = context["dag"]
        choices = DRAFT_CHOICES.get(node.type, ["correct", "malformed"])
        key = f"draft:{node.type}"
        variant = self._draw(key, choices)
        drafter = context["backends"].drafter
        text = drafter.draft(node, dag, context.get("deps") or {}, context.get("feedback") or (), variant=variant)
        return Sample(text=text, prompt=user, meta={"key": key, "choice": variant, "choices": list(choices)})

    def _edit_candidates(self, context: Dict[str, Any]) -> Dict[str, Action]:
        dag: DAG = context["dag"]
        review = context.get("review") or {}
        blamed: Dict[str, List[Dict[str, Any]]] = context.get("blamed") or {}
        store = context.get("store")
        backends = context.get("backends")
        scene_id = dag.scene_id()
        final_id = dag.final_id()
        out: Dict[str, Action] = {"stop": Action("stop", {}, "mock stop")}
        draft_blamed = [nid for nid in blamed if nid in dag and dag[nid].type in DRAFT_TYPES]
        if draft_blamed:
            target = max(draft_blamed, key=lambda nid: len(blamed[nid]))
            issues = blamed[target]
            spec = backends.planner.revise_spec(dag[target], issues, dag) if backends is not None else dict(dag[target].spec)
            feedback = "; ".join(str(i.get("fix_instruction") or i.get("description") or "") for i in issues)[:300]
            out["edit_blamed"] = Action("edit_spec", {"node": target, "spec": spec, "feedback": feedback}, "fix attributed issues")
            if store is not None:
                cands = store.candidates(dag[target], limit=1)
                if cands:
                    out["reuse"] = Action("reuse", {"node": target, "key": cands[0]["key"]}, "reuse cached fragment")
        if scene_id in blamed:
            spec = backends.planner.revise_spec(dag[scene_id], blamed[scene_id], dag) if backends is not None else dict(dag[scene_id].spec)
            out["edit_scene"] = Action("edit_spec", {"node": scene_id, "spec": spec, "feedback": "; ".join(str(i.get("fix_instruction") or "") for i in blamed[scene_id])[:300]}, "fix layout")
        if final_id in blamed:
            out["adjust_final"] = Action("adjust_final", {}, "regenerate with stronger control")
        others = [n.id for n in dag.draft_nodes() if n.id not in blamed]
        if others:
            victim = self.rng.choice(others)
            spec = dict(dag[victim].spec)
            spec["notes"] = list(spec.get("notes") or []) + ["redraw"]
            out["edit_random"] = Action("edit_spec", {"node": victim, "spec": spec}, "random edit")
        deletable = [n.id for n in dag.draft_nodes() if n.type in (ENTITY, OBJECT) and n.id not in dag.consumed()]
        if deletable:
            out["delete_random"] = Action("delete", {"node": self.rng.choice(deletable)}, "random delete")
        splittable = [n for n in dag.draft_nodes() if n.type in (ENTITY, OBJECT) and n.count() > 1 and n.id not in dag.consumed()]
        if splittable:
            node = self.rng.choice(splittable)
            parts = []
            for k in range(node.count()):
                spec = dict(node.spec)
                spec["count"] = 1
                spec["desc"] = f"{node.desc()} (part {k + 1})"
                parts.append({"id": f"{node.id}_p{k + 1}", "type": node.type, "spec": spec})
            out["split_random"] = Action("split", {"node": node.id, "parts": parts}, "random split")
        return out

    def _sample_edit(self, system: str, user: str, context: Dict[str, Any]) -> Sample:
        candidates = self._edit_candidates(context)
        review = context.get("review") or {}
        blamed = context.get("blamed") or {}
        dag: DAG = context["dag"]
        if review.get("passed"):
            state = "pass"
        elif any(nid in dag and dag[nid].type in DRAFT_TYPES for nid in blamed):
            state = "fail_node"
        else:
            state = "fail_scene"
        key = f"edit:{state}"
        choices = [c for c in EDIT_CHOICES if c in candidates]
        choice = self._draw(key, choices)
        action = candidates[choice]
        return Sample(text=action.to_json(), prompt=user, meta={"key": key, "choice": choice, "choices": choices, "action": action.name})

    def update(self, batch: Sequence[TrainSample]) -> Dict[str, Any]:
        moved = 0
        for sample in batch:
            key = sample.meta.get("key")
            choice = sample.meta.get("choice")
            choices = sample.meta.get("choices") or []
            if not key or choice not in choices:
                continue
            table = self._table(key, choices)
            probs = self.probabilities(key, choices)
            adv = float(sample.advantage)
            for c in choices:
                grad = adv * ((1.0 if c == choice else 0.0) - probs[c])
                table[c] += self.lr * grad - self.lr * self.beta * table[c]
                table[c] = max(-8.0, min(8.0, table[c]))
            moved += 1
        self.updates += 1
        return {"updated": True, "samples": moved, "tables": {k: self.probabilities(k, list(v.keys())) for k, v in self.logits.items()}}

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "mock_policy.json").write_text(json.dumps({"logits": self.logits, "updates": self.updates}, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        file = Path(path) / "mock_policy.json"
        if file.is_file():
            data = json.loads(file.read_text(encoding="utf-8"))
            self.logits = data.get("logits", {})
            self.updates = int(data.get("updates", 0))


class APIPolicy(Policy):
    name = "api"
    trainable = False

    def __init__(self, cfg: Any, llm: Any):
        self.cfg = cfg
        self.llm = llm
        self.temperature = float(setting(cfg, "POLICY_TEMPERATURE"))

    def sample(self, system: str, user: str, n: int, kind: str, context: Dict[str, Any]) -> List[Sample]:
        out = []
        node = context.get("node")
        role = "executor" if kind == "draft" else "planner"
        for _ in range(n):
            text = self.llm.chat(role, system, user, json_mode=(kind == "edit"), node=(node.id if node is not None else "policy"), temperature=self.temperature)
            out.append(Sample(text=text, prompt=user, meta={"kind": kind}))
        return out


class HFPolicy(Policy):
    name = "hf"
    trainable = True

    def __init__(self, cfg: Any, model: Any = None, tokenizer: Any = None, device: Optional[str] = None):
        self.cfg = cfg
        self.max_new_tokens = int(setting(cfg, "POLICY_MAX_NEW_TOKENS"))
        self.temperature = float(setting(cfg, "POLICY_TEMPERATURE"))
        self.top_p = float(setting(cfg, "POLICY_TOP_P"))
        self.max_len = int(setting(cfg, "POLICY_MAX_LEN"))
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.updater = None
        if self.model is None:
            self._load()
        else:
            self._finish_setup()

    def _load(self) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = str(setting(self.cfg, "POLICY_MODEL"))
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        lora = LoraConfig(r=int(setting(self.cfg, "POLICY_LORA_R")), lora_alpha=int(setting(self.cfg, "POLICY_LORA_ALPHA")), lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora)
        self._finish_setup()

    def _finish_setup(self) -> None:
        import torch

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.updater = None

    def _ensure_updater(self):
        if self.updater is None:
            from .grpo import GRPOUpdater

            self.updater = GRPOUpdater(self.model, self.tokenizer, lr=float(setting(self.cfg, "TRAIN_LR")), clip_eps=float(setting(self.cfg, "TRAIN_CLIP_EPS")), beta=float(setting(self.cfg, "TRAIN_KL_BETA")), micro_batch=int(setting(self.cfg, "TRAIN_MICRO_BATCH")), max_len=self.max_len)
        return self.updater

    def render_prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def sample(self, system: str, user: str, n: int, kind: str, context: Dict[str, Any]) -> List[Sample]:
        import torch

        prompt = self.render_prompt(system, user)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention = encoded["attention_mask"].to(self.device)
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=int(n),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        out = []
        for row in generated:
            completion_ids = row[input_ids.shape[1]:]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            out.append(Sample(text=text, prompt=prompt, meta={"kind": kind, "tokens": int(completion_ids.shape[0])}))
        return out

    def update(self, batch: Sequence[TrainSample]) -> Dict[str, Any]:
        updater = self._ensure_updater()
        stats = updater.step([s.to_dict() for s in batch])
        stats["updated"] = True
        return stats

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        try:
            self.tokenizer.save_pretrained(path)
        except Exception:
            pass

    def load(self, path: str) -> None:
        from peft import PeftModel

        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.model = PeftModel.from_pretrained(base, path, is_trainable=True)
        self.model.to(self.device)
        self.updater = None


def build_policy(cfg: Any, backends: Any, backend: Optional[str] = None, seed: int = 0) -> Policy:
    name = str(backend or setting(cfg, "POLICY_BACKEND")).lower()
    if name == "mock":
        return MockPolicy(cfg, seed=seed)
    if name == "api":
        llm = backends.llm
        if llm is None:
            from test0.llm import OpenAICompatible

            policy_cfg = cfg.with_overrides(OPENAI_BASE_URL=str(setting(cfg, "POLICY_BASE_URL")) or None, OPENAI_API_KEY=str(setting(cfg, "POLICY_API_KEY")) or None, OPENAI_MODEL_NAME=str(setting(cfg, "POLICY_MODEL")) or None)
            llm = OpenAICompatible(policy_cfg, backends.cost)
        return APIPolicy(cfg, llm)
    if name == "hf":
        return HFPolicy(cfg)
    raise ValueError(f"unknown policy backend {name!r}")
