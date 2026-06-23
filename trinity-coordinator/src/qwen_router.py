"""Qwen3-0.6B feature extractor + linear head for the TRINITY coordinator.

Faithful to the paper's design (Section 3.1, Appendix A.4):
    - Extract the last-token hidden state from the SECOND-TO-LAST layer
      (paper: "h_{n-1}" from the penultimate transformer layer)
    - Pass through a linear head (1024 -> n_a) + softmax/argmax
    - Frozen backbone, only head is trainable

The full backbone has ~600M params in bf16 (~1.2GB); the head is 1024 * n_a
~10K params (or 1024 if block-diagonal-10). Backbone fits comfortably on MPS.

Note: this module ships WITHOUT the backbone weights to keep the repo light.
Pass `model_id` to load from HuggingFace cache or download on demand.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# Cache for the loaded tokenizer / model so we don't re-load every step.
_MODEL_CACHE: dict[str, tuple] = {}


def load_qwen3(model_id: str = "Qwen/Qwen3-0.6B-Base",
               device: str = "cpu",  # mps is fine; cpu is safe default
               dtype: torch.dtype = torch.float32,
               attn_impl: str = "sdpa"):
    """Load Qwen3 model + tokenizer. Cached by (model_id, device, dtype).

    Returns (model, tokenizer, hidden_size, n_layers).
    """
    cache_key = (model_id, device, str(dtype))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # load in fp32 so the extracted hidden states are stable on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        output_hidden_states=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)

    hidden_size = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    _MODEL_CACHE[cache_key] = (model, tokenizer, hidden_size, n_layers)
    return model, tokenizer, hidden_size, n_layers


@torch.no_grad()
def extract_hidden_state(model, tokenizer, text: str,
                         layer_idx: int = -2,
                         device: str = "cpu",
                         max_length: int = 2048) -> np.ndarray:
    """Return the last-token hidden state from layer `layer_idx` (default: 2nd-to-last).

    Shape: (hidden_size,) numpy array.
    """
    # Qwen3 base has no chat template; pass plain text.
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length)
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    # out.hidden_states is a tuple of (num_layers + 1) tensors, each (1, T, hidden)
    # layer 0 = embedding output; layer i = output of layer i
    # layer_idx = -2 -> second-to-last
    h = out.hidden_states[layer_idx][0, -1, :]  # (hidden_size,)
    return h.cpu().float().numpy()


@dataclass
class QwenCoordinatorConfig:
    """Settings for QwenRouter (linear head over Qwen3 hidden state)."""
    model_id: str = "Qwen/Qwen3-0.6B-Base"
    head: str = "linear"          # paper's best is "linear"
    n_outputs: int = 10           # L + 3 roles; toy: 5
    hidden_size: int = 1024       # Qwen3-0.6B's hidden_size
    use_argmax: bool = False      # paper uses softmax; argmax is faster but lossy
    layer_idx: int = -2           # paper: second-to-last layer
    deterministic: bool = True    # if True, sample with temperature=1 + greedy
    temperature: float = 1.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


class QwenRouter(nn.Module):
    """Qwen3-0.6B backbone (frozen) + small linear/block-diag head.

    Backbone is loaded lazily on first .to(device) or .get_params() call.
    For training we only touch the head parameters.
    """

    def __init__(self, cfg: QwenCoordinatorConfig | None = None):
        super().__init__()
        from .heads import HeadConfig, make_head
        self.cfg = cfg or QwenCoordinatorConfig()
        # head will be created lazily after we know hidden_size from the backbone
        self._head: nn.Module | None = None
        self._model = None
        self._tokenizer = None
        # we register a tiny buffer so torch can introspect before lazy init
        self.register_buffer("_init_dummy", torch.zeros(1), persistent=False)

    # ---- lazy init ----
    def _ensure_loaded(self):
        if self._model is not None:
            return
        model, tok, h_size, n_layers = load_qwen3(
            self.cfg.model_id,
            device=self.cfg.device,
            dtype=self.cfg.dtype,
        )
        if h_size != self.cfg.hidden_size:
            print(f"[QwenRouter] overriding hidden_size: {self.cfg.hidden_size} -> {h_size}")
            self.cfg.hidden_size = h_size
        if abs(self.cfg.layer_idx) > n_layers:
            raise ValueError(f"layer_idx {self.cfg.layer_idx} out of range "
                             f"(model has {n_layers} layers)")
        self._model = model
        self._tokenizer = tok
        from .heads import HeadConfig, make_head
        head_cfg = HeadConfig(
            in_dim=self.cfg.hidden_size,
            n_outputs=self.cfg.n_outputs,
            kind=self.cfg.head,
            n_blocks=self.cfg.n_outputs,
        )
        self._head = make_head(head_cfg)

    @property
    def head(self) -> nn.Module:
        self._ensure_loaded()
        return self._head  # type: ignore

    @property
    def n_params(self) -> int:
        return self.head.num_parameters()

    # ---- feature extraction ----
    def features(self, context: str) -> np.ndarray:
        """Extract paper-style h_{n-1} for a transcript string."""
        self._ensure_loaded()
        return extract_hidden_state(
            self._model, self._tokenizer, context,
            layer_idx=self.cfg.layer_idx,
            device=self.cfg.device,
        )

    def features_batch(self, contexts: list[str]) -> np.ndarray:
        """Extract features for a batch of transcripts."""
        return np.stack([self.features(c) for c in contexts], axis=0)

    # ---- forward ----
    def forward(self, contexts: list[str]) -> torch.Tensor:
        """Return logits (1, n_outputs) for a single context."""
        h = torch.from_numpy(self.features(contexts[0])).unsqueeze(0)
        return self.head(h)

    # ---- action selection ----
    def select(self, context: str) -> int:
        """Pick an action index from the context."""
        self.eval()
        with torch.no_grad():
            logits = self.forward([context])[0]
            if self.cfg.use_argmax:
                return int(torch.argmax(logits).item())
            if self.cfg.deterministic:
                # softmax + argmax of probabilities (= argmax of logits)
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits / max(1e-6, self.cfg.temperature), dim=-1)
            return int(torch.multinomial(probs, 1).item())

    # ---- param i/o (numpy flat vector for CMA-ES) ----
    def get_params(self) -> np.ndarray:
        self._ensure_loaded()
        return np.concatenate([
            p.detach().cpu().numpy().flatten()
            for p in self.head.parameters()
        ])

    def set_params(self, flat: np.ndarray):
        self._ensure_loaded()
        idx = 0
        for p in self.head.parameters():
            n = p.numel()
            new = flat[idx:idx + n].reshape(p.shape).astype(np.float32)
            p.data = torch.from_numpy(new).to(p.device)
            idx += n
        assert idx == flat.size, f"param size mismatch: {idx} vs {flat.size}"

    def head_parameters(self):
        return list(self.head.parameters())

    def num_parameters(self) -> int:
        return self.n_params


# ---- adapter so the existing TrinitySystem + make_fitness_fn can use it ----

class QwenCoordinator:
    """Coordinator wrapping a QwenRouter.

    Implements the same `.act(context) -> (model_idx, role)` protocol
    as MLPCoordinator / HeuristicCoordinator so TrinitySystem works
    unchanged.
    """

    def __init__(self, params: Optional[np.ndarray] = None,
                 cfg: Optional[QwenCoordinatorConfig] = None,
                 deterministic: bool = True):
        self.cfg = cfg or QwenCoordinatorConfig()
        if deterministic is not None:
            self.cfg.deterministic = deterministic
        self.router = QwenRouter(self.cfg)
        if params is not None:
            self.router.set_params(params)
        self._action_to_pair = None  # built lazily once n_outputs known
        self._models = None
        self._roles = None

    def configure_outputs(self, models: list, roles: list):
        """Call once TrinitySystem has a pool + role list, so we can map
        action index -> (model_idx, role)."""
        self._models = models
        self._roles = roles
        n_a = len(models) * len(roles)
        if n_a != self.cfg.n_outputs:
            print(f"[QwenCoordinator] n_outputs {self.cfg.n_outputs} -> {n_a}")
            self.cfg.n_outputs = n_a

    def act(self, context: str) -> tuple[int, str]:
        if self._models is None or self._roles is None:
            raise RuntimeError("call configure_outputs(pool.models, roles) first")
        action = self.router.select(context)
        n_roles = len(self._roles)
        model_idx = action // n_roles
        role = self._roles[action % n_roles]
        return model_idx, role

    def route(self, turn: int, transcript: list[dict], task=None,
              max_turns: int = 6) -> tuple[str, str]:
        """TrinitySystem-compatible interface. Formats transcript as a
        context string and delegates to act()."""
        if self._models is None or self._roles is None:
            raise RuntimeError("call configure_outputs(pool.models, roles) first")
        # Build a paper-style context: include the original question + prior
        # assistant turns (tagged). Same convention as TrinitySystem.solve.
        lines = []
        if task is not None:
            lines.append(f"Original Question: {task.prompt}")
            lines.append("")
        for m in transcript:
            lines.append(f"{m['role']}: {m['content']}")
        # Hint the current turn so the router knows it's its turn to act
        lines.append(f"\n[turn {turn}/{max_turns}] Choose next (model, role):")
        context = "\n".join(lines)
        model_idx, role = self.act(context)
        return self._models[model_idx], role

    # ---- convenience for sep-CMA-ES fitness ----
    def get_params(self) -> np.ndarray:
        return self.router.get_params()

    def set_params(self, p: np.ndarray):
        self.router.set_params(p)

    def num_parameters(self) -> int:
        return self.router.num_parameters()

    def reset(self):
        # QwenCoordinator is stateless across episodes — the backbone is
        # frozen and shared across all CMA-ES candidates, and per-task
        # history is owned by TrinitySystem. Match MLPCoordinator's no-op.
        pass
