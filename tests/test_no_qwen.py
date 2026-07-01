"""Tests that DO NOT need the Qwen3-0.6B backbone to be downloaded.

These cover:
  - Toy task generation + answer matching
  - Feature extraction (handcrafted 16-dim)
  - Heuristic / MLPCoordinator end-to-end on toy tasks
  - All 4 paper head architectures + MLP
  - Evolution: recommended_pop_size + random_search on toy tasks
  - SVF (Singular Value Fine-tuning) — pure torch, no model download
  - QwenRouter / QwenCoordinator with a FAKE backbone (monkey-patched)

Run:
    python tests/test_no_qwen.py

Expected: 20/20 PASS without ever touching the HuggingFace cache.

For tests that DO need the real Qwen3-0.6B-Base (~1.2GB) downloaded, see
`tests/test_smoke.py` (Qwen-dependent tests).
"""

from __future__ import annotations
import sys
import os
import random as pyr

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


# ------------------------------------------------------------------
# Section 1: tasks / answer matching
# ------------------------------------------------------------------

def test_task_generation():
    rng = np.random.default_rng(0)
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


# ------------------------------------------------------------------
# Section 2: end-to-end on toy tasks (heuristic / MLP)
# ------------------------------------------------------------------

def test_heuristic_runs():
    pool = LLMPool(MockSmallLLM(seed=1), MockStrongLLM(seed=2))
    sys_ = TrinitySystem(HeuristicCoordinator(), pool, max_turns=6)
    rng = np.random.default_rng(0)
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


# ------------------------------------------------------------------
# Section 3: head architectures (Appendix A.4)
# ------------------------------------------------------------------

def test_head_architectures():
    """All 4 paper head architectures should produce (B, n_a) logits."""
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
    mutation in between)."""
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


# ------------------------------------------------------------------
# Section 4: evolution (toy regime)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Section 5: QwenRouter / QwenCoordinator with FAKE backbone
# (monkey-patches load_qwen3 so no HuggingFace download is triggered)
# ------------------------------------------------------------------

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
        m_key, role = coord.route(
            1,
            [{"role": "user", "content": "Q: 2+2?"}],
            task=None, max_turns=4,
        )
        assert m_key in pool.keys
        assert role in ("Thinker", "Worker", "Verifier")
    finally:
        qr.load_qwen3 = original


def test_qwen_router_multi_layer_concat():
    """Multi-layer feature extraction: config with layer_idxs=(-2,-4) should
    produce a 2*hidden_size feature vector and a correspondingly larger head.
    """
    import src.qwen_router as qr

    class FakeBackbone:
        class cfg:
            hidden_size = 1024
            num_hidden_layers = 28
        def __call__(self, **kwargs):
            from types import SimpleNamespace
            # Make layers differentiable so concat is observable
            h = [torch.zeros(1, 5, 1024) for _ in range(29)]
            h[-1][0, -1, :] = 1.0      # last layer: all ones at last token
            h[-2][0, -1, :] = 2.0      # second-to-last: all twos
            h[-3][0, -1, :] = 3.0
            h[-4][0, -1, :] = 4.0      # fourth-to-last: all fours
            return SimpleNamespace(hidden_states=h)

    class FakeTok:
        def __call__(self, text, **kw):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    original = qr.load_qwen3
    qr.load_qwen3 = lambda *a, **kw: (FakeBackbone(), FakeTok(), 1024, 28)
    try:
        # Single layer (paper default) — should give a 1024-dim feature
        cfg1 = QwenCoordinatorConfig(head="linear", n_outputs=6,
                                     layer_idxs=(-2,), device="cpu")
        c1 = QwenCoordinator(cfg=cfg1, deterministic=True)
        c1.configure_outputs(["Model_A", "Model_B"], ["Thinker", "Worker", "Verifier"])
        h1 = c1.router.features("warm up")
        assert h1.shape == (1024,), f"single-layer: {h1.shape}"
        assert np.allclose(h1, 2.0), "single-layer should equal h_{-2} = 2.0"

        # Two-layer concat (-2, -4) — 2048-dim
        cfg2 = QwenCoordinatorConfig(head="linear", n_outputs=6,
                                     layer_idxs=(-2, -4), device="cpu")
        c2 = QwenCoordinator(cfg=cfg2, deterministic=True)
        c2.configure_outputs(["Model_A", "Model_B"], ["Thinker", "Worker", "Verifier"])
        h2 = c2.router.features("warm up")
        assert h2.shape == (2048,), f"two-layer: {h2.shape}"
        assert np.allclose(h2[:1024], 2.0)
        assert np.allclose(h2[1024:], 4.0)

        # Three layers (-2, -4, -6) — 3072-dim
        cfg3 = QwenCoordinatorConfig(head="linear", n_outputs=6,
                                     layer_idxs=(-2, -4, -6), device="cpu")
        c3 = QwenCoordinator(cfg=cfg3, deterministic=True)
        c3.configure_outputs(["Model_A", "Model_B"], ["Thinker", "Worker", "Verifier"])
        h3 = c3.router.features("warm up")
        assert h3.shape == (3072,), f"three-layer: {h3.shape}"
        assert np.allclose(h3[:1024], 2.0)
        assert np.allclose(h3[1024:2048], 4.0)
        assert np.allclose(h3[2048:], 0.0)  # layer -6 = h[29-6]=h[23] is zeros

        # Head param count must scale with in_dim (linear: n_outputs * in_dim)
        assert c1.num_parameters() == 6 * 1024
        assert c2.num_parameters() == 6 * 2048
        assert c3.num_parameters() == 6 * 3072

        # Batched path: features_batch should also concat layers
        feats_b = c2.router.features_batch(["Q1", "Q2"])
        assert feats_b.shape == (2, 2048)
    finally:
        qr.load_qwen3 = original


# ------------------------------------------------------------------
# Section 6: SVF (Singular Value Fine-tuning) — paper Section 3.1
# Pure torch, no model download.
# ------------------------------------------------------------------

def test_svf_linear_initial_equals_base():
    """SVFLinear with delta=0 should reproduce its base_linear exactly.

    This is the central correctness invariant: at construction, the
    SVF-wrapped layer is a no-op replacement of the original nn.Linear.
    """
    from src.qwen_router import SVFLinear
    torch.manual_seed(0)
    base = torch.nn.Linear(64, 32, bias=True)
    base.eval()
    svf = SVFLinear(base, rank=64)
    svf.eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        y_base = base(x)
        y_svf = svf(x)
    assert torch.allclose(y_base, y_svf, atol=1e-5), \
        f"max diff = {(y_base - y_svf).abs().max().item()}"
    # The delta buffer is the only trainable piece, and starts at 0
    assert torch.all(svf.delta == 0.0)
    assert svf.num_trainable() == 64


def test_svf_linear_perturbation_changes_output():
    """Setting delta != 0 should change the output (verify parameter is live)."""
    from src.qwen_router import SVFLinear
    torch.manual_seed(1)
    base = torch.nn.Linear(32, 16, bias=False)
    svf = SVFLinear(base, rank=32)
    x = torch.randn(2, 32)
    with torch.no_grad():
        y0 = svf(x).clone()
        svf.delta.data += 0.1  # perturb every singular value
        y1 = svf(x)
    diff = (y0 - y1).abs().max().item()
    assert diff > 1e-3, f"delta perturbation had no effect (max diff = {diff})"
    # reset_to_identity should restore y0
    svf.reset_to_identity()
    with torch.no_grad():
        y2 = svf(x)
    assert torch.allclose(y0, y2, atol=1e-5)


def test_svf_linear_uses_only_top_r_singular_values():
    """Truncating rank should still give a reasonable approximation, and
    trainable count must equal rank."""
    from src.qwen_router import SVFLinear
    torch.manual_seed(2)
    base = torch.nn.Linear(128, 64, bias=True)
    svf_full = SVFLinear(base, rank=None)        # all 64 singular values
    svf_low = SVFLinear(base, rank=8)            # only top-8
    assert svf_full.num_trainable() == 64
    assert svf_low.num_trainable() == 8
    x = torch.randn(2, 128)
    with torch.no_grad():
        y_full = svf_full(x)
        y_low = svf_low(x)
        y_base = base(x)
    # full-rank SVF should exactly equal base at delta=0
    assert torch.allclose(y_base, y_full, atol=1e-5)
    # low-rank should be a (possibly rough) approximation
    approx_err = (y_base - y_low).abs().mean().item()
    assert approx_err < 5.0, f"low-rank approx error too large: {approx_err}"


def test_attach_svf_wraps_last_n_blocks():
    """attach_svf should install exactly n_blocks SVFLinear wrappers in
    the last n_blocks transformer layers, with n_blocks*rank total params
    — matching the paper's 9,216 budget for Qwen3-0.6B with rank=1024
    and n_blocks=9."""
    from src.qwen_router import attach_svf, SVFLinear
    import torch.nn as nn

    class FakeLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.02)
            self.bias = nn.Parameter(torch.zeros(out_f))

    class FakeAttn(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = FakeLinear(dim, dim)

    class FakeLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.self_attn = FakeAttn(dim)

    class FakeInner(nn.Module):
        def __init__(self, n_layers, dim):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(dim) for _ in range(n_layers)])

    class FakeModel(nn.Module):
        def __init__(self, n_layers=28, dim=1024):
            super().__init__()
            self.model = FakeInner(n_layers, dim)
        def eval(self):
            return self

    m = FakeModel(n_layers=28, dim=1024)
    n_blocks, rank = 9, 1024
    installed = attach_svf(m, target="last_k_q_proj",
                           rank=rank, n_blocks=n_blocks)
    assert len(installed) == n_blocks
    for svf in installed:
        assert isinstance(svf, SVFLinear)
        assert svf.num_trainable() == rank
    # Only the LAST n_blocks' q_proj should be wrapped; earlier ones stay plain
    for i, layer in enumerate(m.model.layers):
        q = layer.self_attn.q_proj
        is_wrapped = isinstance(q, SVFLinear)
        if i < 28 - n_blocks:
            assert not is_wrapped, f"layer {i} should NOT be wrapped"
        else:
            assert is_wrapped, f"layer {i} should be wrapped"
    # Total SVF trainable params = n_blocks * rank = 9 * 1024 = 9,216 (paper)
    total_svf = sum(s.num_trainable() for s in installed)
    assert total_svf == n_blocks * rank == 9216, total_svf


def test_qwen_router_svf_param_count_and_roundtrip():
    """QwenRouter with use_svf=True should have head + 9,216 SVF params
    in its flat get_params vector, and set_params should round-trip."""
    import src.qwen_router as qr
    import torch.nn as nn

    class FakeAttnInner(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.q_proj = nn.Linear(dim, dim)

    class FakeLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.self_attn = FakeAttnInner(dim)

    class FakeInner(nn.Module):
        def __init__(self, n_layers, dim):
            super().__init__()
            self.layers = nn.ModuleList([FakeLayer(dim) for _ in range(n_layers)])

    class FakeBackbone(nn.Module):
        class cfg:
            hidden_size = 1024
            num_hidden_layers = 28
        def __init__(self):
            super().__init__()
            self.model = FakeInner(28, 1024)
        def __call__(self, **kw):
            from types import SimpleNamespace
            h = torch.zeros(1, 5, 1024)
            h[0, -1, :] = 1.0
            return SimpleNamespace(hidden_states=[h] * 29)
        def named_parameters(self):
            return super().named_parameters()
        def eval(self):
            return self
        def to(self, *a, **kw):
            return self

    class FakeTok:
        def __call__(self, text, **kw):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    original = qr.load_qwen3
    qr.load_qwen3 = lambda *a, **kw: (FakeBackbone(), FakeTok(), 1024, 28)
    try:
        cfg = QwenCoordinatorConfig(
            head="linear", n_outputs=6,
            layer_idxs=(-2,),
            use_svf=True, svf_rank=1024, svf_n_blocks=9,
            device="cpu",
        )
        coord = QwenCoordinator(cfg=cfg, deterministic=True)
        coord.configure_outputs(["Model_A", "Model_B"], ["Thinker", "Worker", "Verifier"])
        # Lazy init
        _ = coord.router.features("warm up")
        # num_head_parameters = 6*1024 = 6144
        # num_svf_parameters = 9*1024 = 9216
        # total = 15360
        assert coord.router.num_head_parameters() == 6 * 1024
        assert coord.router.num_svf_parameters() == 9 * 1024 == 9216
        assert coord.router.num_parameters() == 6 * 1024 + 9 * 1024
        p = coord.get_params()
        assert p.shape[0] == 6 * 1024 + 9 * 1024
        # The first 6*1024 entries are head weights, the last 9*1024 are SVF deltas
        head_p = coord.router.head_parameters()
        n_head = sum(x.numel() for x in head_p)
        # Head section: equal to actual head parameters
        head_section = p[:n_head]
        actual_head = np.concatenate([x.detach().cpu().float().numpy().ravel()
                                       for x in head_p])
        assert np.allclose(head_section, actual_head)
        # SVF section: equals 0 (delta=0 initialization)
        svf_section = p[n_head:]
        assert np.allclose(svf_section, 0.0), "SVF deltas should init to 0"
        # Round-trip: set_params(ones) should change the params
        coord.set_params(np.ones_like(p))
        p2 = coord.get_params()
        assert np.allclose(p2, 1.0)
        # SVF deltas non-zero means backbone output differs from base
        for svf in coord.router._svf_modules:
            assert not torch.all(svf.delta == 0)
    finally:
        qr.load_qwen3 = original


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

if __name__ == "__main__":
    class Skipped(Exception):
        """Raised by a test when it cannot run because a resource is missing."""
        pass

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
