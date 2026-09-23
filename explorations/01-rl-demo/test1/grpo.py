from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def encode_pair(tokenizer: Any, prompt: str, completion: str, max_len: int) -> Tuple[List[int], List[int]]:
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    completion_ids = list(tokenizer(completion, add_special_tokens=False)["input_ids"])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (not completion_ids or completion_ids[-1] != eos):
        completion_ids.append(int(eos))
    if len(completion_ids) > max_len - 8:
        completion_ids = completion_ids[: max_len - 8]
    room = max_len - len(completion_ids)
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]
    ids = prompt_ids + completion_ids
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    return ids, mask


def collate(pairs: Sequence[Tuple[List[int], List[int]]], pad_id: int):
    import torch

    length = max(len(ids) for ids, _m in pairs)
    input_ids = torch.full((len(pairs), length), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(pairs), length), dtype=torch.long)
    completion = torch.zeros((len(pairs), length), dtype=torch.float32)
    for i, (ids, mask) in enumerate(pairs):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, : len(ids)] = 1
        completion[i, : len(ids)] = torch.tensor(mask, dtype=torch.float32)
    return input_ids, attention, completion


def token_logprobs(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    import torch

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, 2, targets.unsqueeze(-1)).squeeze(-1)


def grpo_loss(logp: Any, ref_logp: Any, advantages: Any, mask: Any, clip_eps: float, beta: float, old_logp: Optional[Any] = None) -> Tuple[Any, Dict[str, float]]:
    import torch

    if old_logp is None:
        old_logp = logp.detach()
    ratio = torch.exp(logp - old_logp)
    adv = advantages.unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_term = -torch.min(unclipped, clipped)
    diff = ref_logp - logp
    kl = torch.exp(diff) - diff - 1.0
    per_token = policy_term + beta * kl
    denom = mask.sum(dim=1).clamp(min=1.0)
    loss = ((per_token * mask).sum(dim=1) / denom).mean()
    stats = {
        "loss": float(loss.detach().item()),
        "kl": float(((kl * mask).sum(dim=1) / denom).mean().detach().item()),
        "logp": float(((logp * mask).sum(dim=1) / denom).mean().detach().item()),
    }
    return loss, stats


def completion_masked(mask: Any) -> Any:
    return mask[:, 1:]


class GRPOUpdater:
    def __init__(self, model: Any, tokenizer: Any, lr: float, clip_eps: float, beta: float, micro_batch: int, max_len: int, max_grad_norm: float = 1.0):
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.clip_eps = float(clip_eps)
        self.beta = float(beta)
        self.micro_batch = max(1, int(micro_batch))
        self.max_len = int(max_len)
        self.max_grad_norm = float(max_grad_norm)
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
        self.device = next(model.parameters()).device
        self.steps = 0

    def _ref_logprobs(self, input_ids: Any, attention: Any) -> Any:
        import torch

        with torch.no_grad():
            if hasattr(self.model, "disable_adapter"):
                with self.model.disable_adapter():
                    return token_logprobs(self.model, input_ids, attention)
            return token_logprobs(self.model, input_ids, attention)

    def step(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        import torch

        items = [s for s in samples if abs(float(s.get("advantage", 0.0))) > 1e-8]
        if not items:
            return {"loss": 0.0, "kl": 0.0, "samples": 0, "skipped": len(samples)}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else (self.tokenizer.eos_token_id or 0)
        total_loss = 0.0
        total_kl = 0.0
        batches = 0
        for start in range(0, len(items), self.micro_batch):
            chunk = items[start:start + self.micro_batch]
            pairs = [encode_pair(self.tokenizer, s["prompt"], s["completion"], self.max_len) for s in chunk]
            input_ids, attention, completion = collate(pairs, pad_id)
            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)
            mask = completion_masked(completion).to(self.device)
            adv = torch.tensor([float(s["advantage"]) for s in chunk], dtype=torch.float32, device=self.device)
            ref_logp = self._ref_logprobs(input_ids, attention)
            logp = token_logprobs(self.model, input_ids, attention)
            loss, stats = grpo_loss(logp, ref_logp, adv, mask, self.clip_eps, self.beta)
            (loss * len(chunk) / len(items)).backward()
            total_loss += stats["loss"] * len(chunk)
            total_kl += stats["kl"] * len(chunk)
            batches += 1
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm)
        self.optimizer.step()
        self.steps += 1
        return {"loss": total_loss / len(items), "kl": total_kl / len(items), "samples": len(items), "skipped": len(samples) - len(items), "micro_batches": batches}
