"""Coordinator implementations: heuristic, learned (any head), Qwen (stub).

The paper (Appendix A.4) studies 4 head architectures:
    linear, low_rank, sparse, block_diag
The block-diagonal-10 head + argmax output is the most parameter-efficient.

This module exposes a unified `MLPCoordinator` that picks any of those heads
behind the same `route(turn, transcript, task)` interface.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .features import (
    FEATURE_DIM,
    NUM_ROLES,
    IDX_TO_ROLE,
    action_to_label,
    extract_features,
    label_to_action,
)
from .heads import HeadConfig, make_head


# ------------------------------------------------------------------
# Heuristic baseline (mirrors the OpenAI prototype in the user's post)
# ------------------------------------------------------------------

class HeuristicCoordinator:
    """Same state machine as the hand-coded OpenAI prototype.

    Pattern: start with Thinker -> Worker -> Verifier -> (loop on REVISE).
    """

    def __init__(self):
        self.history: list[str] = []

    def route(self, turn: int, transcript: list[dict], task=None,
              max_turns: int = 6) -> tuple[str, str]:
        if turn == 1 or not self.history:
            choice = ("Model_B", "Thinker")
        elif self.history[-1] == "Thinker":
            choice = ("Model_A", "Worker")
        elif self.history[-1] == "Worker":
            choice = ("Model_B", "Verifier")
        else:  # Verifier -> REVISE -> back to Thinker
            choice = ("Model_B", "Thinker")
        self.history.append(choice[1])
        return choice

    def reset(self):
        self.history = []


# ------------------------------------------------------------------
# Learned coordinator with paper-faithful head architectures
# ------------------------------------------------------------------

@dataclass
class CoordinatorConfig:
    in_dim: int = FEATURE_DIM
    n_models: int = 2       # L
    n_roles: int = NUM_ROLES  # 3
    n_outputs: int = 5      # n_a = L + 3
    hidden: int = 32        # used by low_rank and mlp
    head: str = "block_diag"  # linear / low_rank / sparse / block_diag / mlp
    n_blocks: int = 5       # for block_diag; set to n_outputs for full block-diag
    use_argmax: bool = True  # paper's best for block_diag-10 uses argmax
    temperature: float = 1.0
    deterministic: bool = True


class MLPCoordinator:
    """Coordinator with any of the 4 paper head architectures.

    By default, uses `block_diag` (5 blocks for n_a=5) + `argmax` output,
    mirroring the paper's block-diagonal-10 + argmax configuration (scaled
    down for n_a=5 here).
    """

    def __init__(self, params: np.ndarray | None = None,
                 cfg: CoordinatorConfig | None = None):
        self.cfg = cfg or CoordinatorConfig()
        head_cfg = HeadConfig(
            in_dim=self.cfg.in_dim,
            n_outputs=self.cfg.n_outputs,
            hidden=self.cfg.hidden,
            kind=self.cfg.head,
            n_blocks=self.cfg.n_blocks,
        )
        self.head = make_head(head_cfg)
        self.history: list[str] = []
        if params is not None:
            self.set_params(params)

    # --- parameter access (CMA-ES friendly) ---
    def get_params(self) -> np.ndarray:
        out = []
        for p in self.head.parameters():
            out.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(out).astype(np.float32)

    def set_params(self, flat: np.ndarray) -> None:
        i = 0
        with torch.no_grad():
            for p in self.head.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(flat[i:i + n].reshape(p.shape)))
                i += n

    def num_parameters(self) -> int:
        return self.head.num_parameters()

    def reset(self):
        self.history = []

    @torch.no_grad()
    def route(self, turn: int, transcript: list[dict], task=None,
              max_turns: int = 6) -> tuple[str, str]:
        x = torch.from_numpy(extract_features(transcript, task, max_turns)).unsqueeze(0)
        z = self.head(x).squeeze(0)  # (n_a,)
        z = z / max(self.cfg.temperature, 1e-3)
        n_models = self.cfg.n_models
        model_logits = z[:n_models]
        role_logits = z[n_models:]
        if self.cfg.use_argmax:
            m_idx = int(model_logits.argmax().item())
            r_idx = int(role_logits.argmax().item())
        else:
            m_probs = torch.softmax(model_logits, dim=-1)
            r_probs = torch.softmax(role_logits, dim=-1)
            m_idx = int(torch.multinomial(m_probs, 1).item())
            r_idx = int(torch.multinomial(r_probs, 1).item())
        choice = label_to_action(m_idx, r_idx)
        self.history.append(choice[1])
        return choice


# ------------------------------------------------------------------
# Qwen-based coordinator (optional, for the faithful paper reproduction)
# ------------------------------------------------------------------

class QwenCoordinator:
    """Coordinator that uses a small Qwen model's hidden state + linear head.

    Faithful to the paper's Section 3 design. Use Qwen2-0.5B as a stand-in
    for the unavailable Qwen3-0.6B. Head uses Linear by default (paper's
    most stable architecture).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct",
                 head: str = "block_diag", n_blocks: int = 10,
                 use_argmax: bool = True, n_models: int = 7, n_roles: int = 3,
                 device: str | None = None):
        self.model_name = model_name
        self.n_models = n_models
        self.n_roles = n_roles
        self.n_outputs = n_models + n_roles
        self._loaded = False
        self.head_module: nn.Module | None = None
        self.head_kind = head
        self.n_blocks = n_blocks
        self.use_argmax = use_argmax
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if self._loaded:
            return
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        head_cfg = HeadConfig(
            in_dim=self.backbone.config.hidden_size,
            n_outputs=self.n_outputs,
            kind=self.head_kind,
            n_blocks=self.n_blocks,
        )
        self.head_module = make_head(head_cfg).to(self.device).eval()
        self._loaded = True

    @torch.no_grad()
    def route(self, turn: int, transcript: list[dict], task=None,
              max_turns: int = 6) -> tuple[str, str]:
        self.load()
        text = "\n".join(f"{m['role']}: {m['content']}" for m in transcript)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=2048).to(self.device)
        out = self.backbone(**inputs)
        h = out.last_hidden_state[:, -1, :]  # last token (paper allows earlier)
        z = self.head_module(h).squeeze(0)
        n_models = self.n_models
        model_logits = z[:n_models]
        role_logits = z[n_models:]
        if self.use_argmax:
            m_idx = int(model_logits.argmax().item())
            r_idx = int(role_logits.argmax().item())
        else:
            m_idx = int(torch.softmax(model_logits, -1).argmax().item())
            r_idx = int(torch.softmax(role_logits, -1).argmax().item())
        return label_to_action(m_idx, r_idx)

    def num_parameters(self) -> int:
        if not self._loaded:
            return 0
        return sum(p.numel() for p in self.head_module.parameters())

    def get_params(self) -> np.ndarray:
        if not self._loaded:
            return np.zeros(0, dtype=np.float32)
        out = []
        for p in self.head_module.parameters():
            out.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(out).astype(np.float32)

    def set_params(self, flat: np.ndarray) -> None:
        if not self._loaded:
            return
        i = 0
        with torch.no_grad():
            for p in self.head_module.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(flat[i:i + n].reshape(p.shape)).to(self.device))
                i += n

    def reset(self):
        pass
