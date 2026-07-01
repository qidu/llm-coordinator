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


# ------------------------------------------------------------------
# SVF (Singular Value Fine-tuning) — paper Section 3.1, Appendix A.2
# ------------------------------------------------------------------
#
# Idea: take a target weight matrix W (e.g. a Qwen3 attention q_proj), compute
# a truncated SVD W ≈ U · diag(s) · V^T (top-r components), and freeze U, V.
# Only the r singular values `s` are trainable. Forward pass replaces
# W x with U · diag(σ·(1+s)) · V^T x  where σ = s_init are the original
# singular values and (1+s) starts at 1.0 (i.e. identity transform) and
# gets perturbed by training.
#
# Trainable parameters: r per target matrix. Paper uses 6 target matrices
# × 1536 dim → 9,216 params. We pick 9 Qwen3-0.6B q_proj matrices (last 9
# transformer blocks) × 1024 top singular values → 9,216 params, matching
# the paper exactly.

class SVFLinear(nn.Module):
    """A drop-in replacement for nn.Linear whose weight is parameterized
    via SVF (U, s, V frozen → only s trainable).

    Forward: y = x W^T + b where W = U · diag(σ·(1+s)) · V^T.
    `s` is initialized to zeros so the initial output equals the original
    linear layer's output (since σ·(1+0) = σ recovers the original).

    Args:
        base_linear: the original nn.Linear to mimic (frozen).
        rank:        number of top singular values to keep.
    """

    def __init__(self, base_linear: nn.Linear, rank: int | None = None):
        super().__init__()
        W = base_linear.weight.detach().float()              # (out, in)
        out_features, in_features = W.shape
        if rank is None:
            rank = min(out_features, in_features)
        rank = min(rank, out_features, in_features)
        # full SVD on float32 — small enough (1024x1024) to be cheap
        U_full, s_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
        # keep top-r
        U = U_full[:, :rank].contiguous()                    # (out, r)
        s = s_full[:rank].contiguous()                       # (r,)
        Vh = Vh_full[:rank, :].contiguous()                  # (r, in)
        self.register_buffer("U", U)
        self.register_buffer("sigma", s)                     # original singular values
        self.register_buffer("Vh", Vh)
        # the only trainable parameter: a per-singular-value scalar
        # initialized to 0 so initial output = base_linear(input)
        self.delta = nn.Parameter(torch.zeros(rank))
        if base_linear.bias is not None:
            self.register_buffer("bias", base_linear.bias.detach().clone())
        else:
            self.bias = None
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

    def effective_weight(self) -> torch.Tensor:
        """Compute the current effective weight matrix W = U·diag(σ·(1+δ))·V^T."""
        scale = self.sigma * (1.0 + self.delta)              # (r,)
        # (out, r) * (r, r diag) * (r, in) = (out, in)
        return (self.U * scale) @ self.Vh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.effective_weight()                          # (out, in)
        out = x @ W.T
        if self.bias is not None:
            out = out + self.bias
        return out

    def num_trainable(self) -> int:
        return self.delta.numel()

    def reset_to_identity(self) -> None:
        """Reset δ=0, recovering the original base_linear behavior exactly."""
        with torch.no_grad():
            self.delta.zero_()


def attach_svf(model, target: str = "last_k_q_proj",
               rank: int = 1024, n_blocks: int = 9) -> list[SVFLinear]:
    """Replace selected nn.Linear modules in a model with SVFLinear copies.

    Args:
        model:    a HuggingFace causal LM (e.g. Qwen3).
        target:   which modules to SVF-wrap. Currently only:
                    - "last_k_q_proj" — q_proj of the last `n_blocks` layers
        rank:     number of top singular values per matrix.
        n_blocks: how many trailing transformer blocks to wrap.

    Returns:
        list of installed SVFLinear modules, in install order. These are
        *additional* parameters that the caller can pass to QwenRouter.
    """
    installed: list[SVFLinear] = []
    if target != "last_k_q_proj":
        raise ValueError(f"unknown SVF target: {target!r}")
    # Qwen3 / Qwen2 model structure: model.model.layers[i].self_attn.q_proj
    if hasattr(model, "model"):
        layers = model.model.layers
    else:
        layers = model.layers
    n = len(layers)
    start = max(0, n - n_blocks)
    for i in range(start, n):
        attn = layers[i].self_attn
        if not hasattr(attn, "q_proj"):
            raise AttributeError(
                f"layer {i} self_attn has no q_proj — model not supported"
            )
        base = attn.q_proj
        if not isinstance(base, nn.Linear):
            raise TypeError(f"q_proj at layer {i} is {type(base)}, expected nn.Linear")
        svf = SVFLinear(base, rank=rank)
        # Move SVF to the same device/dtype as the original linear
        svf = svf.to(device=base.weight.device, dtype=base.weight.dtype)
        attn.q_proj = svf
        installed.append(svf)
    return installed


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
        dtype=dtype,
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
    out = model(**inputs, use_cache=False)
    # out.hidden_states is a tuple of (num_layers + 1) tensors, each (1, T, hidden)
    # layer 0 = embedding output; layer i = output of layer i
    # layer_idx = -2 -> second-to-last
    h = out.hidden_states[layer_idx][0, -1, :]  # (hidden_size,)
    return h.cpu().float().numpy()


@torch.no_grad()
def extract_hidden_state_multi(model, tokenizer, text: str,
                               layer_idxs: list[int] | tuple[int, ...] = (-2,),
                               device: str = "cpu",
                               max_length: int = 2048) -> np.ndarray:
    """Return the last-token hidden states from multiple layers, concatenated.

    Args:
        layer_idxs: layer indices (negative = from end). E.g. (-2,) gives
            the paper's default second-to-last layer; (-2, -4) gives
            second-to-last and fourth-to-last concatenated along the
            feature dim.

    Shape: (len(layer_idxs) * hidden_size,) numpy array. If multiple layers
    are requested the head's in_dim must be set accordingly.
    """
    if len(layer_idxs) == 1:
        return extract_hidden_state(model, tokenizer, text,
                                    layer_idx=layer_idxs[0], device=device,
                                    max_length=max_length)
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length)
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    out = model(**inputs, use_cache=False)
    # out.hidden_states is a tuple of (num_layers + 1) tensors, each (1, T, hidden)
    parts = [out.hidden_states[li][0, -1, :] for li in layer_idxs]
    h = torch.cat(parts, dim=-1)  # (len(layer_idxs) * hidden_size,)
    return h.cpu().float().numpy()


@torch.no_grad()
def extract_hidden_state_batch(model, tokenizer, texts: list[str],
                               layer_idx: int = -2,
                               device: str = "cpu",
                               max_length: int = 2048,
                               batch_size: int = 0,
                               layer_idxs: list[int] | tuple[int, ...] | None = None) -> np.ndarray:
    """Batched version: extract last-token hidden state for many texts at once.

    Shape: (len(texts), hidden_size) numpy array — or
    (len(texts), len(layer_idxs)*hidden_size) if `layer_idxs` is given.

    This is the critical speedup for sep-CMA-ES: instead of N separate
    forward passes, do one batched forward. For transcripts of similar
    length, this is ~N times faster on CPU and ~N times less memory
    pressure (single KV cache, single embedding lookup, etc).

    When batch_size > 0, splits the input into chunks to avoid OOM on
    GPUs with limited VRAM. Each chunk is padded independently so the
    padding overhead doesn't cascade across the full set.

    `layer_idxs` (preferred over `layer_idx`): if provided, concatenates
    the last-token hidden states from each requested layer. `layer_idx`
    is kept for backwards compatibility and is used only when
    `layer_idxs` is None.
    """
    if layer_idxs is None:
        layer_idxs = (layer_idx,)
    if not texts:
        out_dim = len(layer_idxs) * model.config.hidden_size
        return np.zeros((0, out_dim), dtype=np.float32)
    if batch_size > 0 and len(texts) > batch_size:
        chunks = []
        for i in range(0, len(texts), batch_size):
            chunk = extract_hidden_state_batch(
                model, tokenizer, texts[i:i + batch_size],
                layer_idx=layer_idx, device=device, max_length=max_length,
                batch_size=0,  # inner calls don't recurse
                layer_idxs=layer_idxs,
            )
            chunks.append(chunk)
        return np.concatenate(chunks, axis=0)
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,        # pad to longest in batch
        truncation=True,
        max_length=max_length,
    )
    if hasattr(enc, "to"):
        enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc, use_cache=False)
    # last *real* token (right-padded), found via attention_mask sum - 1
    last_idx = enc["attention_mask"].sum(dim=1) - 1  # (B,)
    last_idx = last_idx.clamp(min=0)
    B = out.hidden_states[0].size(0)
    parts = [out.hidden_states[li][torch.arange(B), last_idx, :]
             for li in layer_idxs]                  # list of (B, H)
    h = torch.cat(parts, dim=-1)                    # (B, len(layer_idxs)*H)
    return h.cpu().float().numpy()


@dataclass
class QwenCoordinatorConfig:
    """Settings for QwenRouter (linear head over Qwen3 hidden state)."""
    model_id: str = "Qwen/Qwen3-0.6B-Base"
    head: str = "linear"          # paper's best is "linear"
    n_outputs: int = 10           # L + 3 roles; toy: 5
    hidden_size: int = 1024       # Qwen3-0.6B's hidden_size (set by backbone)
    use_argmax: bool = False      # paper uses softmax; argmax is faster but lossy
    layer_idx: int = -2           # paper: second-to-last layer (legacy single-layer)
    layer_idxs: tuple[int, ...] | None = None  # multi-layer concat (preferred)
    deterministic: bool = True    # if True, sample with temperature=1 + greedy
    temperature: float = 1.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    # --- SVF (paper Section 3.1) — 9,216 trainable params in the backbone ---
    use_svf: bool = False              # if True, attach SVF to last N blocks' q_proj
    svf_rank: int = 1024               # top-r singular values per matrix
    svf_n_blocks: int = 9              # 9 × 1024 = 9,216 (paper's parameter budget)


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
        # SVF-wrapped modules (list of SVFLinear). Empty when use_svf=False.
        self._svf_modules: list[SVFLinear] = []
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
        # Resolve layer_idxs (multi-layer concat) vs layer_idx (single, legacy)
        if self.cfg.layer_idxs is None:
            self.cfg.layer_idxs = (self.cfg.layer_idx,)
        else:
            self.cfg.layer_idxs = tuple(self.cfg.layer_idxs)
        # in_dim = hidden_size * len(layer_idxs)
        effective_in_dim = h_size * len(self.cfg.layer_idxs)
        if effective_in_dim != self.cfg.hidden_size:
            print(f"[QwenRouter] overriding hidden_size: {self.cfg.hidden_size} -> {effective_in_dim} "
                  f"(hidden_size={h_size} * {len(self.cfg.layer_idxs)} layers)")
            self.cfg.hidden_size = effective_in_dim
        for li in self.cfg.layer_idxs:
            if abs(li) > n_layers:
                raise ValueError(f"layer_idx {li} out of range "
                                 f"(model has {n_layers} layers)")
        # Optionally attach SVF wrappers to the backbone's last few q_proj.
        # We do this BEFORE freezing, so we control which params stay trainable.
        if self.cfg.use_svf:
            self._svf_modules = attach_svf(
                model, target="last_k_q_proj",
                rank=self.cfg.svf_rank, n_blocks=self.cfg.svf_n_blocks,
            )
            n_svf_params = sum(m.num_trainable() for m in self._svf_modules)
            print(f"[QwenRouter] SVF attached: {len(self._svf_modules)} modules, "
                  f"{n_svf_params} trainable params "
                  f"(rank={self.cfg.svf_rank}, n_blocks={self.cfg.svf_n_blocks})")
        else:
            self._svf_modules = []
        # Now freeze the rest of the backbone. SVF deltas are nn.Parameters
        # so requires_grad is already True for them; other params get frozen.
        model.eval()
        for name, p in model.named_parameters():
            p.requires_grad_(False)
        # Re-enable gradients for the SVF deltas (they survived the freeze
        # above by virtue of being nn.Parameter, but be explicit).
        for svf in self._svf_modules:
            svf.delta.requires_grad_(True)
        model.to(self.cfg.device)

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
        # Move head to the same device/dtype as the backbone so the
        # forward doesn't bounce tensors between CPU and GPU. For
        # heads <100K params this is free.
        if self.cfg.device != "cpu":
            self._head = self._head.to(self.cfg.device)
        if self.cfg.dtype != torch.float32:
            self._head = self._head.to(self.cfg.dtype)

    @property
    def head(self) -> nn.Module:
        self._ensure_loaded()
        return self._head  # type: ignore

    @property
    def n_params(self) -> int:
        return self.head.num_parameters()

    # ---- feature extraction ----
    def features(self, context: str) -> np.ndarray:
        """Extract paper-style h_{n-1} for a transcript string.

        If cfg.layer_idxs has multiple entries, returns their concatenation
        (length = len(layer_idxs) * hidden_size).
        """
        self._ensure_loaded()
        return extract_hidden_state_multi(
            self._model, self._tokenizer, context,
            layer_idxs=self.cfg.layer_idxs,
            device=self.cfg.device,
        )

    def features_batch(self, contexts: list[str], batch_size: int = 0) -> np.ndarray:
        """Batched feature extraction: Qwen3 forward over all contexts.

        This is the key perf win — a batched forward is ~N times faster
        than N separate forwards for similar-length contexts.

        Args:
            contexts: list of text strings to extract features from.
            batch_size: if >0, split into chunks of this size to avoid OOM.
        """
        self._ensure_loaded()
        if not contexts:
            return np.zeros((0, self.cfg.hidden_size), dtype=np.float32)
        # If we only have one context, the regular path is just as fast and
        # avoids tokenizer overhead. But for >1 contexts, always batch.
        if len(contexts) == 1:
            return self.features(contexts[0])[None, :]
        return extract_hidden_state_batch(
            self._model, self._tokenizer, contexts,
            layer_idx=self.cfg.layer_idx,
            device=self.cfg.device,
            batch_size=batch_size,
            layer_idxs=self.cfg.layer_idxs,
        )

    # ---- forward ----
    def forward(self, contexts: list[str]) -> torch.Tensor:
        """Return logits (1, n_outputs) for a single context."""
        h = torch.from_numpy(self.features(contexts[0])).unsqueeze(0)
        # Match the head's device + dtype so the matmul agrees.
        head_params = next(self.head.parameters())
        h = h.to(device=head_params.device, dtype=head_params.dtype)
        return self.head(h)

    def forward_batched_features(self, features: np.ndarray) -> torch.Tensor:
        """Batched head forward over pre-computed features.

        Args:
            features: (B, hidden_size) numpy array of pre-extracted Qwen
                      features (use .features_batch() to get them).

        Returns:
            (B, n_outputs) logits tensor.
        """
        self._ensure_loaded()
        h = torch.from_numpy(np.ascontiguousarray(features))
        # Match the head's device + dtype so the matmul agrees.
        h = h.to(device=next(self.head.parameters()).device,
                 dtype=next(self.head.parameters()).dtype)
        return self.head(h)

    def head_select(self, logits: torch.Tensor, deterministic: bool = True) -> list[int]:
        """Convert a batched (B, n_outputs) logit tensor to per-sample actions.

        deterministic=True -> argmax (paper's default; we want reproducibility)
        """
        if deterministic:
            return torch.argmax(logits, dim=-1).tolist()
        probs = torch.softmax(logits / max(1e-6, self.cfg.temperature), dim=-1)
        return [int(torch.multinomial(p, 1).item()) for p in probs]

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
        """Return head + SVF deltas as a single flat numpy array.

        Order: head parameters first, then each SVF module's delta vector
        in installation order. The vector matches num_parameters().
        """
        self._ensure_loaded()
        parts = [
            p.detach().cpu().float().numpy().flatten()
            for p in self.head.parameters()
        ]
        for svf in self._svf_modules:
            parts.append(svf.delta.detach().cpu().float().numpy().flatten())
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def set_params(self, flat: np.ndarray):
        """Inverse of get_params: write into head + SVF deltas.

        Expects the same layout (head first, then SVF deltas in order).
        """
        self._ensure_loaded()
        idx = 0
        for p in self.head.parameters():
            n = p.numel()
            new = flat[idx:idx + n].reshape(p.shape)
            t = torch.from_numpy(np.ascontiguousarray(new)).to(p.device).to(p.dtype)
            p.data = t
            idx += n
        for svf in self._svf_modules:
            n = svf.delta.numel()
            new = flat[idx:idx + n].reshape(svf.delta.shape)
            t = torch.from_numpy(np.ascontiguousarray(new)).to(svf.delta.device).to(svf.delta.dtype)
            svf.delta.data = t
            idx += n
        assert idx == flat.size, f"param size mismatch: {idx} vs {flat.size}"

    @property
    def n_params(self) -> int:
        """Total trainable parameter count: head + (SVF deltas if attached)."""
        return self.num_parameters()

    def head_parameters(self):
        return list(self.head.parameters())

    def svf_parameters(self) -> list[nn.Parameter]:
        """All SVF delta parameters (one Parameter of length `rank` per module)."""
        self._ensure_loaded()
        return [m.delta for m in self._svf_modules]

    def num_parameters(self) -> int:
        """Total trainable params: head + SVF deltas."""
        n = sum(p.numel() for p in self.head.parameters())
        for m in self._svf_modules:
            n += m.num_trainable()
        return n

    def num_head_parameters(self) -> int:
        return sum(p.numel() for p in self.head.parameters())

    def num_svf_parameters(self) -> int:
        return sum(m.num_trainable() for m in self._svf_modules)


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
