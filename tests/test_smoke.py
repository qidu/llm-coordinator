"""Smoke tests for the prototype. Run with `python -m pytest -q` or just `python tests/test_smoke.py`."""

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

from src.tasks import make_task, make_dataset, extract_final_answer, is_correct
from src.llm_pool import MockSmallLLM, MockStrongLLM, LLMPool
from src.coordinator import HeuristicCoordinator, MLPCoordinator, CoordinatorConfig
from src.trinity_system import TrinitySystem
from src.features import extract_features, FEATURE_DIM
from src.heads import make_head, HeadConfig
from src.evolution import random_search, recommended_pop_size
from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig


def test_task_generation():
    rng = np.random.default_rng(0)
    import random as pyr
    for _ in range(20):
        t = make_task(pyr.Random(int(rng.integers(0, 1 << 30))))
        assert t.prompt
        assert t.answer
        assert t.kind in ("arithmetic", "logic", "string")


def test_extract_final():
    assert extract_final_answer("blah\nFINAL: 42") == "42"
    # trailing dot is stripped
    assert extract_final_answer("blah\nFINAL: 42.") == "42"
    assert extract_final_answer("the answer is 7") == "7"
    # last non-empty line fallback
    assert extract_final_answer("nothing here") == "nothing here"
    # empty input
    assert extract_final_answer("") is None
    assert extract_final_answer("   \n  ") is None


def test_is_correct():
    class T:
        answer = "42"
    from src.tasks import Task
    t = Task(id="x", prompt="?", answer="42", kind="arithmetic")
    assert is_correct(t, "FINAL: 42")
    assert is_correct(t, "FINAL: 42.0")
    assert not is_correct(t, "FINAL: 41")


def test_features_shape():
    t = make_task()
    f = extract_features([{"role": "user", "content": t.prompt}], t, max_turns=6)
    assert f.shape == (FEATURE_DIM,)
    assert f.dtype == np.float32
    assert float(f[-1]) == 1.0  # bias


def test_heuristic_runs():
    pool = LLMPool(MockSmallLLM(seed=1), MockStrongLLM(seed=2))
    sys_ = TrinitySystem(HeuristicCoordinator(), pool, max_turns=6)
    rng = np.random.default_rng(0)
    import random as pyr
    correct = 0
    for i in range(10):
        t = make_task(pyr.Random(int(rng.integers(0, 1 << 30))))
        r = sys_.solve(t)
        if r.correct:
            correct += 1
    # we don't assert a specific rate — just that it runs
    assert 0 <= correct <= 10


def test_mlp_coord_runs():
    pool = LLMPool(MockSmallLLM(seed=3), MockStrongLLM(seed=4))
    coord = MLPCoordinator()
    sys_ = TrinitySystem(coord, pool, max_turns=6)
    t = make_task()
    r = sys_.solve(t)
    assert r.turns >= 1
    assert len(r.decisions) == r.turns


def test_mlp_param_io():
    coord = MLPCoordinator()
    p1 = coord.get_params()
    assert p1.ndim == 1
    assert p1.size == coord.num_parameters()
    p2 = p1 + 0.01
    coord.set_params(p2)
    p3 = coord.get_params()
    assert np.allclose(p2, p3)


def test_dataset():
    ds = make_dataset(8, seed=42)
    assert len(ds) == 8
    assert all(t.id for t in ds)


def test_head_architectures():
    """All 4 paper head architectures should produce (B, n_a) logits."""
    import torch
    n_a = 5  # 2 models + 3 roles in our toy pool
    for kind in ("linear", "low_rank", "sparse", "block_diag", "mlp"):
        cfg = HeadConfig(in_dim=FEATURE_DIM, n_outputs=n_a, kind=kind, n_blocks=n_a)
        h = make_head(cfg)
        x = torch.randn(4, FEATURE_DIM)
        y = h(x)
        assert y.shape == (4, n_a), f"{kind} produced {y.shape}"
        n_params = h.num_parameters()
        assert n_params > 0
    # paper-faithful sizes: block_diag with n_blocks=n_a should be smallest
    cfg_bd = HeadConfig(in_dim=FEATURE_DIM, n_outputs=n_a, kind="block_diag", n_blocks=n_a)
    cfg_lin = HeadConfig(in_dim=FEATURE_DIM, n_outputs=n_a, kind="linear")
    assert make_head(cfg_bd).num_parameters() < make_head(cfg_lin).num_parameters()


def test_mlp_with_different_heads():
    """MLPCoordinator should accept all head kinds and route successfully."""
    pool = LLMPool(MockSmallLLM(seed=5), MockStrongLLM(seed=6))
    for kind in ("linear", "block_diag", "mlp"):
        cfg = CoordinatorConfig(head=kind, n_blocks=5, use_argmax=True)
        coord = MLPCoordinator(cfg=cfg)
        sys_ = TrinitySystem(coord, pool, max_turns=4)
        t = make_task()
        r = sys_.solve(t)
        assert r.turns >= 1


def test_argmax_vs_softmax():
    """Argmax mode should be deterministic across calls (same coord, no param
    mutation in between). Mock LLM randomness is shared, so test on the
    coordinator's output directly."""
    pool = LLMPool(MockSmallLLM(seed=42), MockStrongLLM(seed=43))
    cfg_argmax = CoordinatorConfig(use_argmax=True, deterministic=True)
    c_arg = MLPCoordinator(cfg=cfg_argmax)
    # run twice with same coord object, decisions should match
    decisions1 = []
    decisions2 = []
    c_arg.reset()
    for turn in range(1, 4):
        d, _ = c_arg.route(turn, [{"role": "user", "content": "Q"}])
        decisions1.append(d)
    c_arg.reset()
    for turn in range(1, 4):
        d, _ = c_arg.route(turn, [{"role": "user", "content": "Q"}])
        decisions2.append(d)
    assert decisions1 == decisions2, f"{decisions1} != {decisions2}"


def test_recommended_pop_size():
    # paper: n=10240 -> lambda=32; n=1024 -> lambda=25; n=100 -> lambda=18
    assert recommended_pop_size(10240) == 32
    assert recommended_pop_size(1024) == 25
    assert 16 <= recommended_pop_size(100) <= 20


def test_random_search_runs():
    pool = LLMPool(MockSmallLLM(seed=7), MockStrongLLM(seed=8))
    from src.evolution import CMAESConfig
    cfg = CoordinatorConfig(head="block_diag", n_blocks=5, use_argmax=True)
    init_coord = MLPCoordinator(cfg=cfg)
    init = init_coord.get_params()
    train_tasks = make_dataset(4, seed=0, difficulty_range=(1, 2))
    from src.evolution import make_fitness_fn
    fit = make_fitness_fn(train_tasks, pool, max_turns=3, coord_cfg=cfg)
    es_cfg = CMAESConfig(n_dim=init.size, pop_size=4, generations=2, fitness_fn=fit)
    best = random_search(es_cfg, init, verbose=False)
    assert best.shape == init.shape


def test_qwen_router_loads_and_features():
    """Smoke test: load Qwen3-0.6B-Base, extract a hidden state, run head.

    Skipped automatically if the backbone isn't downloaded yet.
    """
    from pathlib import Path
    cache_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots"
    if not cache_dir.exists():
        print("SKIP  test_qwen_router_loads_and_features (model not downloaded)")
        return
    snapshots = list(cache_dir.glob("*/model.safetensors"))
    if not snapshots:
        print("SKIP  test_qwen_router_loads_and_features (safetensors not yet downloaded)")
        return

    from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
    from src.llm_pool import LLMPool
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


def test_qwen_router_with_fake_backbone():
    """Test the QwenRouter API end-to-end with a fake backbone (no download)."""
    import src.qwen_router as qr

    class FakeBackbone:
        class cfg:
            hidden_size = 1024
            num_hidden_layers = 28
        def __call__(self, **kwargs):
            from types import SimpleNamespace
            h = torch.zeros(1, 5, 1024)
            h[0, -1, :] = 1.0
            return SimpleNamespace(hidden_states=[h] * 29)

    class FakeTok:
        def __call__(self, text, **kw):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    original = qr.load_qwen3
    qr.load_qwen3 = lambda *a, **kw: (FakeBackbone(), FakeTok(), 1024, 28)
    try:
        cfg = QwenCoordinatorConfig(head="linear", n_outputs=6,
                                    device="cpu", use_argmax=False)
        coord = QwenCoordinator(cfg=cfg, deterministic=True)
        pool = LLMPool()
        coord.configure_outputs(pool.keys, ["Thinker", "Worker", "Verifier"])
        # touch a feature to lazy-load
        _ = coord.router.features("warm up")
        assert coord.num_parameters() == 1024 * 6, coord.num_parameters()
        # round-trip
        p = coord.get_params()
        coord.set_params(np.zeros_like(p))
        assert np.allclose(coord.get_params(), 0)
        # act produces a valid pair
        m, r = coord.act("Q: hi")
        assert m in (0, 1)
        assert r in ("Thinker", "Worker", "Verifier")
        # different head kinds
        for head in ("low_rank", "sparse", "block_diag", "mlp"):
            cfg2 = QwenCoordinatorConfig(head=head, n_outputs=6,
                                        device="cpu", use_argmax=False)
            coord2 = QwenCoordinator(cfg=cfg2)
            coord2.configure_outputs(pool.keys, ["Thinker", "Worker", "Verifier"])
            _ = coord2.router.features("warm up")
            assert coord2.num_parameters() > 0
        # reset() exists and is a no-op (TrinitySystem calls it per episode)
        coord.reset()
        # route(turn, transcript, task, max_turns) returns (model_key, role)
        # — TrinitySystem calls this contract.
        m_key, role = coord.route(
            1,
            [{"role": "user", "content": "Q: 2+2?"}],
            task=None, max_turns=4,
        )
        assert m_key in pool.keys
        assert role in ("Thinker", "Worker", "Verifier")
    finally:
        qr.load_qwen3 = original


# NOTE: if __name__ == "__main__" runner block intentionally stays at the
# END of this file so it can discover all test_* functions defined below.


# ------------------------------------------------------------------
# Tests for the batched feature extractor + batched fitness
# ------------------------------------------------------------------

def test_extract_hidden_state_batch():
    """features_batch should give same per-row result as features, plus
    a small shape sanity check."""
    from src.qwen_router import extract_hidden_state_batch
    router_mod = sys.modules["src.qwen_router"]
    # If the model isn't loaded, skip — but mark with a print.
    from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
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
        assert np.allclose(feats[i], single, atol=1e-4), \
            f"row {i} mismatch (max diff {np.max(np.abs(feats[i] - single))})"


def test_batched_fitness_matches_naive():
    """The batched fitness function should give exactly the same scores
    as the naive per-candidate fitness function for a small test setup."""
    from src.evolution import make_batched_qwen_fitness_fn, make_fitness_fn
    from src.tasks import make_dataset
    from src.llm_pool import LLMPool
    from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig

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
    from src.evolution import make_batched_qwen_fitness_fn
    from src.tasks import make_dataset
    from src.llm_pool import LLMPool
    from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
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
    from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
    import torch
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


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    fail = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            fail += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - fail}/{len(tests)} passed")
    sys.exit(0 if fail == 0 else 1)
