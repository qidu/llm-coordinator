"""Coordinator implementations: heuristic, learned (any head).

The paper (Appendix A.4) studies 4 head architectures:
    linear, low_rank, sparse, block_diag
The block-diagonal-10 head + argmax output is the most parameter-efficient.

This module exposes a unified `MLPCoordinator` that picks any of those heads
behind the same `route(turn, transcript, task)` interface.

For the paper-faithful Qwen3-0.6B-backed coordinator, see `src/qwen_router.py`.
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
    model_keys,
)
from .heads import HeadConfig, make_head


# ------------------------------------------------------------------
# Heuristic baseline (mirrors the OpenAI prototype in the user's post)
# ------------------------------------------------------------------

class HeuristicCoordinator:
    """Same state machine as the hand-coded OpenAI prototype.

    Pattern: start with Thinker -> Worker -> Verifier -> (loop on REVISE).
    Uses model_keys() from features.py so it works with 2 mocks or N real LLMs.
    """

    def __init__(self):
        self.history: list[str] = []
        keys = model_keys()
        self._worker = keys[0] if len(keys) > 0 else "Model_A"
        self._strong = keys[-1] if len(keys) > 0 else "Model_B"

    def route(self, turn: int, transcript: list[dict], task=None,
              max_turns: int = 6) -> tuple[str, str]:
        if turn == 1 or not self.history:
            choice = (self._strong, "Thinker")
        elif self.history[-1] == "Thinker":
            choice = (self._worker, "Worker")
        elif self.history[-1] == "Worker":
            choice = (self._strong, "Verifier")
        else:  # Verifier -> REVISE -> back to Thinker
            choice = (self._strong, "Thinker")
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
    n_models: int = 2       # L  (overridden at runtime by len(model_keys()))
    n_roles: int = NUM_ROLES  # 3
    n_outputs: int = 5      # n_a = L + 3  (overridden at runtime by len(model_keys()) + n_roles)
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
        # n_outputs adapts to the number of model keys set at runtime.
        n_outputs = len(model_keys()) + NUM_ROLES
        head_cfg = HeadConfig(
            in_dim=self.cfg.in_dim,
            n_outputs=n_outputs,
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
        # Use len(model_keys()) at runtime so n_models adapts if keys change.
        n_models = len(model_keys())
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
