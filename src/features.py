"""Feature extraction from the running transcript.

The paper feeds the last hidden state of a 0.6B LLM into a small linear head.
For the MLP prototype we don't need an LLM — we just need a vector that
captures the state of the conversation well enough that an MLP can learn a
useful routing policy.

The vector is deliberately interpretable and small (16 dims).

    [0]  turn index (normalized 0..1)
    [1]  last role one-hot[0] (Thinker)
    [2]  last role one-hot[1] (Worker)
    [3]  last role one-hot[2] (Verifier)
    [4]  # times Thinker used
    [5]  # times Worker used
    [6]  # times Verifier used
    [7]  # times rejected so far
    [8]  # consecutive rejects (1 if last verifier said REVISE else 0)
    [9]  context length (chars, log-normalized)
    [10] context length in turns
    [11] last output had a "FINAL:" line
    [12] last output had "JUDGMENT: ACCEPT"
    [13] last output had "JUDGMENT: REVISE"
    [14] task difficulty (1-5, normalized 0..1)
    [15] bias = 1
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

from .tasks import Task

ROLE_TO_IDX = {"Thinker": 0, "Worker": 1, "Verifier": 2}
IDX_TO_ROLE = {v: k for k, v in ROLE_TO_IDX.items()}
NUM_ROLES = 3

# Default model keys for the 2-mock setup.  Override with set_model_keys().
_DEFAULT_MODEL_KEYS = ["Model_A", "Model_B"]
_model_keys: list[str] = list(_DEFAULT_MODEL_KEYS)
_IDX_TO_MODEL: dict[int, str] = {i: k for i, k in enumerate(_model_keys)}


def set_model_keys(keys: list[str]) -> None:
    """Override model key mapping (call before training with real LLMs).

    Must be called before MLPCoordinator / HeuristicCoordinator are instantiated
    so the action<->label tables are consistent throughout a run.
    """
    global _model_keys, _IDX_TO_MODEL
    _model_keys = list(keys)
    _IDX_TO_MODEL = {i: k for i, k in enumerate(_model_keys)}


def model_keys() -> list[str]:
    """Current model keys list (length = n_models)."""
    return _model_keys


def label_to_action(model_idx: int, role_idx: int) -> tuple[str, str]:
    """Decode router logits to (model_key, role)."""
    return _IDX_TO_MODEL.get(model_idx, f"Model_{model_idx}"), IDX_TO_ROLE[role_idx]


def action_to_label(model_key: str, role: str) -> tuple[int, int]:
    """Encode (model_key, role) -> (model_idx, role_idx)."""
    try:
        model_idx = _model_keys.index(model_key)
    except ValueError:
        model_idx = 0
    role_idx = ROLE_TO_IDX[role]
    return model_idx, role_idx

FEATURE_DIM = 16


def _last_role(transcript: Sequence[dict]) -> str:
    for msg in reversed(transcript):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            for r in ("Thinker", "Worker", "Verifier"):
                if content.startswith(f"[{r}"):
                    return r
    return "Thinker"  # initial assumption


def _count_roles(transcript: Sequence[dict]) -> Counter:
    c = Counter()
    for msg in transcript:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        for r in ("Thinker", "Worker", "Verifier"):
            if content.startswith(f"[{r}"):
                c[r] += 1
                break
    return c


def _last_assistant_text(transcript: Sequence[dict]) -> str:
    for msg in reversed(transcript):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def extract_features(transcript: Sequence[dict], task: Task | None, max_turns: int) -> np.ndarray:
    """Return shape (FEATURE_DIM,) float32 vector."""
    turn_idx = sum(1 for m in transcript if m.get("role") == "assistant")
    last_role = _last_role(transcript)
    last_idx = ROLE_TO_IDX[last_role]
    one_hot = [0.0] * NUM_ROLES
    one_hot[last_idx] = 1.0

    counts = _count_roles(transcript)
    n_thinker = counts.get("Thinker", 0)
    n_worker = counts.get("Worker", 0)
    n_verifier = counts.get("Verifier", 0)

    n_reject = sum(
        1 for m in transcript
        if m.get("role") == "assistant" and "JUDGMENT: REVISE" in m.get("content", "")
    )
    last_text = _last_assistant_text(transcript)
    consecutive_reject = 1.0 if "JUDGMENT: REVISE" in last_text else 0.0

    total_chars = sum(len(m.get("content", "")) for m in transcript)
    log_chars = math.log1p(total_chars) / 10.0  # soft normalize
    log_chars = min(log_chars, 1.0)

    n_assistant = n_thinker + n_worker + n_verifier

    has_final = 1.0 if "FINAL:" in last_text else 0.0
    has_accept = 1.0 if "JUDGMENT: ACCEPT" in last_text else 0.0
    has_reject = 1.0 if "JUDGMENT: REVISE" in last_text else 0.0

    difficulty = (task.difficulty - 1) / 4.0 if task is not None else 0.5
    turn_norm = turn_idx / max(1, max_turns)

    vec = np.array(
        [
            turn_norm,
            *one_hot,
            float(n_thinker),
            float(n_worker),
            float(n_verifier),
            float(n_reject),
            consecutive_reject,
            log_chars,
            float(n_assistant) / max(1, max_turns),
            has_final,
            has_accept,
            has_reject,
            difficulty,
            1.0,  # bias
        ],
        dtype=np.float32,
    )
    assert vec.shape == (FEATURE_DIM,), f"got {vec.shape}"
    return vec


def label_to_action(model_idx: int, role_idx: int) -> tuple[str, str]:
    """Decode router logits to (model_key, role)."""
    return f"Model_{'A' if model_idx == 0 else 'B'}", IDX_TO_ROLE[role_idx]


def action_to_label(model_key: str, role: str) -> tuple[int, int]:
    model_idx = 0 if model_key.endswith("A") else 1
    role_idx = ROLE_TO_IDX[role]
    return model_idx, role_idx
