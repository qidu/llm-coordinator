"""Coordinator implementations: heuristic, MLP, Qwen (stub)."""

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
# MLP router — the "small linear head" of the paper
# ------------------------------------------------------------------

class MLPRouter(nn.Module):
    """Tiny MLP mapping transcript features -> (model_logits, role_logits).

    Default config: 16 -> 32 -> 32 -> (2+3) = ~1.4K params, well under the
    paper's ~10K linear head. Easy to scale up via width.
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 32,
                 n_models: int = 2, n_roles: int = NUM_ROLES):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.model_head = nn.Linear(hidden, n_models)
        self.role_head = nn.Linear(hidden, n_roles)
        self.in_dim = in_dim
        self.hidden = hidden
        self.n_models = n_models
        self.n_roles = n_roles

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.model_head(h), self.role_head(h)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class MLPCoordinatorConfig:
    in_dim: int = FEATURE_DIM
    hidden: int = 32
    n_models: int = 2
    n_roles: int = NUM_ROLES


class MLPCoordinator:
    """Wraps an MLPRouter with the same interface as HeuristicCoordinator."""

    def __init__(self, params: np.ndarray | None = None, cfg: MLPCoordinatorConfig | None = None,
                 temperature: float = 1.0, deterministic: bool = False):
        self.cfg = cfg or MLPCoordinatorConfig()
        self.net = MLPRouter(self.cfg.in_dim, self.cfg.hidden,
                             self.cfg.n_models, self.cfg.n_roles)
        self.temperature = temperature
        self.deterministic = deterministic
        self.history: list[str] = []
        if params is not None:
            self.set_params(params)

    # --- parameter access (CMA-ES friendly) ---
    def get_params(self) -> np.ndarray:
        out = []
        for p in self.net.parameters():
            out.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(out).astype(np.float32)

    def set_params(self, flat: np.ndarray) -> None:
        i = 0
        with torch.no_grad():
            for p in self.net.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(flat[i:i + n].reshape(p.shape)))
                i += n

    def num_parameters(self) -> int:
        return self.net.num_parameters()

    def reset(self):
        self.history = []

    @torch.no_grad()
    def route(self, turn: int, transcript: list[dict], task=None,
              max_turns: int = 6) -> tuple[str, str]:
        x = torch.from_numpy(extract_features(transcript, task, max_turns)).unsqueeze(0)
        m_logits, r_logits = self.net(x)
        m_logits = m_logits / max(self.temperature, 1e-3)
        r_logits = r_logits / max(self.temperature, 1e-3)
        if self.deterministic:
            m_idx = int(m_logits.argmax(dim=-1).item())
            r_idx = int(r_logits.argmax(dim=-1).item())
        else:
            m_probs = torch.softmax(m_logits, dim=-1)
            r_probs = torch.softmax(r_logits, dim=-1)
            m_idx = int(torch.multinomial(m_probs, 1).item())
            r_idx = int(torch.multinomial(r_probs, 1).item())
        choice = label_to_action(m_idx, r_idx)
        self.history.append(choice[1])
        return choice


# ------------------------------------------------------------------
# Qwen-based coordinator (optional, for the faithful paper reproduction)
# ------------------------------------------------------------------

class QwenCoordinator:
    """Coordinator that uses a small Qwen model's last hidden state + linear head.

    Faithful to the paper's Section 4 design, but kept pluggable so the rest of
    the pipeline doesn't depend on a 0.6B checkpoint being present.

    Use case: extract hidden state from a local Qwen2-0.5B-Instruct (close
    substitute for the unavailable Qwen-0.6B) and train a linear head.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct",
                 head_hidden: int = 64, n_models: int = 2, n_roles: int = NUM_ROLES,
                 device: str | None = None):
        self.model_name = model_name
        self.n_models = n_models
        self.n_roles = n_roles
        self._loaded = False
        self.head: nn.Module | None = None
        self.head_hidden = head_hidden
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        if self._loaded:
            return
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.backbone = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, self.head_hidden),
            nn.GELU(),
            nn.Linear(self.head_hidden, self.n_models + self.n_roles),
        ).to(self.device).eval()
        self._loaded = True

    @torch.no_grad()
    def route(self, turn: int, transcript: list[dict], task=None,
              max_turns: int = 6) -> tuple[str, str]:
        self.load()
        text = "\n".join(f"{m['role']}: {m['content']}" for m in transcript)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        out = self.backbone(**inputs)
        h = out.last_hidden_state[:, -1, :]  # last token
        logits = self.head(h)
        m_logits, r_logits = logits[..., :self.n_models], logits[..., self.n_models:]
        m_idx = int(m_logits.argmax(dim=-1).item())
        r_idx = int(r_logits.argmax(dim=-1).item())
        return label_to_action(m_idx, r_idx)

    def num_parameters(self) -> int:
        if not self._loaded:
            return 0
        return sum(p.numel() for p in self.head.parameters())

    def get_params(self) -> np.ndarray:
        if not self._loaded:
            return np.zeros(0, dtype=np.float32)
        out = []
        for p in self.head.parameters():
            out.append(p.detach().cpu().numpy().ravel())
        return np.concatenate(out).astype(np.float32)

    def set_params(self, flat: np.ndarray) -> None:
        if not self._loaded:
            return
        i = 0
        with torch.no_grad():
            for p in self.head.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(flat[i:i + n].reshape(p.shape)).to(self.device))
                i += n

    def reset(self):
        pass
