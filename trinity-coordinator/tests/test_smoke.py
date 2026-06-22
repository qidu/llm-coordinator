"""Smoke tests for the prototype. Run with `python -m pytest -q` or just `python tests/test_smoke.py`."""

from __future__ import annotations
import sys
import os
import numpy as np

# allow running as plain script
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.tasks import make_task, make_dataset, extract_final_answer, is_correct
from src.llm_pool import MockSmallLLM, MockStrongLLM, LLMPool
from src.coordinator import HeuristicCoordinator, MLPCoordinator
from src.trinity_system import TrinitySystem
from src.features import extract_features, FEATURE_DIM


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
