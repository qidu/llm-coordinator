"""Tests that require the Qwen3-0.6B-Base backbone to be DOWNLOADED.

These cover:
  - Real Qwen3-0.6B hidden-state extraction (single layer)
  - Batched fitness evaluation against the real backbone
  - dtype preservation when calling set_params on a bfloat16 head
  - end-to-end forward pass through the real backbone

For tests that DO NOT need the backbone (toy regime, head architectures,
SVF math, etc.), see `tests/test_no_qwen.py` — those run in <5s without
ever touching the HuggingFace cache.

Run:
    # (One-time) install + download the Qwen backbone
    pip install -r requirements.txt
    huggingface-cli download Qwen/Qwen3-0.6B-Base

    # Run the tests
    python tests/test_smoke.py

Expected: 5/5 PASS (after the 1.2GB backbone download).

If the backbone is NOT downloaded, those tests that require it raise a
Skipped exception and are counted as SKIP.  Exit code is 0 as long as
no test actually fails.

The full set of unit tests is split across:
    tests/test_no_qwen.py    — 20 tests, no download needed
    tests/test_smoke.py      —  5 tests, needs Qwen3-0.6B-Base
"""

from __future__ import annotations
import sys
import os

import numpy as np
import torch

# allow running as plain script
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.llm_pool import LLMPool
from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig


class Skipped(Exception):
    """Raised by a test when it cannot run because a resource is missing."""
    pass


def _skip(reason: str) -> None:
    """Raise Skipped to tell the runner this test should be skipped."""
    raise Skipped(reason)


# ------------------------------------------------------------------
# Helper: check whether the Qwen3-0.6B backbone has been downloaded.
# ------------------------------------------------------------------

def _qwen3_available() -> bool:
    from pathlib import Path
    cache_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots"
    if not cache_dir.exists():
        return False
    snapshots = list(cache_dir.glob("*/model.safetensors"))
    return bool(snapshots)


# ------------------------------------------------------------------
# Tests (5 total)
# ------------------------------------------------------------------

def test_qwen_router_loads_and_features():
    """Smoke test: load Qwen3-0.6B-Base, extract a hidden state, run head.

    Skipped automatically if the backbone isn't downloaded yet.
    """
    if not _qwen3_available():
        _skip("Qwen3-0.6B-Base not downloaded — run: huggingface-cli download Qwen/Qwen3-0.6B-Base")

    cfg = QwenCoordinatorConfig(
        model_id="Qwen/Qwen3-0.6B-Base",
        head="linear",
        n_outputs=6,  # 2 models * 3 roles
        use_argmax=False,
        device="cpu",
        dtype=torch.float32,
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=True)
    pool = LLMPool()
    coord.configure_outputs(pool.keys, ["Thinker", "Worker", "Verifier"])

    # extract a feature
    h = coord.router.features("What is 2+2?\nassistant: The answer is 4.")
    assert h.shape == (1024,), f"hidden state shape: {h.shape}"

    # round-trip params
    p = coord.get_params()
    assert p.shape[0] == 1024 * 6
    coord.set_params(np.zeros_like(p))
    p2 = coord.get_params()
    assert np.allclose(p2, 0)

    # .act() should return (model_idx, role)
    act = coord.act("Question: how many vowels in 'trinity'?")
    assert act[0] in (0, 1)
    assert act[1] in ("Thinker", "Worker", "Verifier")


def test_extract_hidden_state_batch():
    """features_batch should give same per-row result as features, plus
    a small shape sanity check. Uses the real Qwen backbone."""
    if not _qwen3_available():
        _skip("Qwen3-0.6B-Base not downloaded — run: huggingface-cli download Qwen/Qwen3-0.6B-Base")

    cfg = QwenCoordinatorConfig(
        model_id="Qwen/Qwen3-0.6B-Base", head="linear", n_outputs=6,
        device="cpu", use_argmax=False,
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=True)
    _ = coord.router.features("warm up")  # lazy load
    ctxs = ["Q: 2+2?", "Q: 3*3?", "Q: capital of France?"]
    feats = coord.router.features_batch(ctxs)
    assert feats.shape == (3, coord.cfg.hidden_size), feats.shape
    # Per-row match with single features()
    for i, c in enumerate(ctxs):
        single = coord.router.features(c)
        assert np.allclose(feats[i], single, atol=2e-4), \
            f"row {i} mismatch (max diff {np.max(np.abs(feats[i] - single))})"


def test_batched_fitness_matches_naive():
    """The batched fitness function should give exactly the same scores
    as the naive per-candidate fitness function for a small test setup."""
    if not _qwen3_available():
        _skip("Qwen3-0.6B-Base not downloaded — run: huggingface-cli download Qwen/Qwen3-0.6B-Base")

    from src.evolution import make_batched_qwen_fitness_fn, make_fitness_fn
    from src.tasks import make_dataset

    pool = LLMPool()
    models = pool.keys
    roles = ["Thinker", "Worker", "Verifier"]
    n_outputs = len(models) * len(roles)

    cfg = QwenCoordinatorConfig(
        model_id="Qwen/Qwen3-0.6B-Base", head="linear", n_outputs=n_outputs,
        device="cpu", use_argmax=False,
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=True)
    coord.configure_outputs(models, roles)
    _ = coord.router.features("warm up")

    rng = np.random.default_rng(0)
    tasks = make_dataset(2, seed=42, difficulty_range=(1, 2))
    fit_batched = make_batched_qwen_fitness_fn(
        tasks, pool, max_turns=2, coord_template=coord, use_early_bonus=False,
    )

    def factory(p):
        coord.set_params(p)
        return coord
    fit_naive = make_fitness_fn(
        tasks, pool, max_turns=2, coord_factory=factory, use_early_bonus=False,
    )

    init = coord.get_params()
    X = rng.standard_normal((4, init.size)).astype(np.float32) * 0.1
    fits_b = fit_batched(X)
    fits_n = np.array([fit_naive(x) for x in X])
    assert np.allclose(fits_b, fits_n, atol=1e-6), \
        f"batched vs naive mismatch: {fits_b} vs {fits_n}"


def test_batched_fitness_shape():
    """batched_fitness must return a 1-D array of length pop."""
    if not _qwen3_available():
        _skip("Qwen3-0.6B-Base not downloaded — run: huggingface-cli download Qwen/Qwen3-0.6B-Base")

    from src.evolution import make_batched_qwen_fitness_fn
    from src.tasks import make_dataset

    pool = LLMPool()
    models = pool.keys
    roles = ["Thinker", "Worker", "Verifier"]
    n_outputs = len(models) * len(roles)
    cfg = QwenCoordinatorConfig(
        model_id="Qwen/Qwen3-0.6B-Base", head="linear", n_outputs=n_outputs,
        device="cpu", use_argmax=False,
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=True)
    coord.configure_outputs(models, roles)
    _ = coord.router.features("warm up")
    tasks = make_dataset(2, seed=0, difficulty_range=(1, 2))
    fit = make_batched_qwen_fitness_fn(
        tasks, pool, max_turns=2, coord_template=coord, use_early_bonus=False,
    )
    init = coord.get_params()
    X = np.random.default_rng(0).standard_normal((6, init.size)).astype(np.float32) * 0.1
    out = fit(X)
    assert out.shape == (6,), out.shape
    assert np.all(np.isfinite(out)), "non-finite fitness"


def test_set_params_dtype_preservation():
    """set_params should respect the head's existing dtype (e.g. bfloat16)."""
    if not _qwen3_available():
        _skip("Qwen3-0.6B-Base not downloaded — run: huggingface-cli download Qwen/Qwen3-0.6B-Base")

    cfg = QwenCoordinatorConfig(
        model_id="Qwen/Qwen3-0.6B-Base", head="linear", n_outputs=6,
        device="cpu", use_argmax=False, dtype=torch.bfloat16,
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=True)
    # Lazy init
    _ = coord.router.features("warm up")
    init = coord.get_params()  # should always return float32 (CMA-ES uses fp32)
    # Now set back — head weights should still be bf16
    coord.set_params(init)
    for p in coord.router.head.parameters():
        assert p.dtype == torch.bfloat16, f"head param dtype = {p.dtype}, expected bf16"


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items()
            if k.startswith("test_") and callable(v)]
    fail = skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Skipped as e:
            skipped += 1
            print(f"SKIP  {t.__name__}: {e}")
        except Exception as e:
            fail += 1
            print(f"FAIL  {t.__name__}: {e}")
    total = len(tests)
    passed = total - fail - skipped
    print(f"\n{passed}/{total} passed, {skipped} skipped, {fail} failed")
    # Exit 0 if no failures (SKIPs are acceptable — backbone just not downloaded)
    sys.exit(0 if fail == 0 else 1)
