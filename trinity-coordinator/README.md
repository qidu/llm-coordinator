# TRINITY Coordinator — Prototype

A standalone Python prototype of **TRINITY: An Evolved LLM Coordinator**
(ICLR 2026, Sakana AI), focused on Section 4's state machine and the
evolutionary training of the coordinator's routing head.

## What's in the box

| Module | What it does |
|---|---|
| `src/tasks.py` | Toy tasks (arithmetic / logic / string / two-step) with verifiable answers |
| `src/llm_pool.py` | `MockSmallLLM` (80% reliable) and `MockStrongLLM` (98% reliable) — drop-in interface for real LLMs |
| `src/features.py` | 16-dim transcript feature vector (turn #, last role, reject count, etc.) |
| `src/coordinator.py` | `HeuristicCoordinator`, `MLPRouter` + `MLPCoordinator` (paper-faithful small head), `QwenCoordinator` (optional) |
| `src/trinity_system.py` | The Section 4 state machine: per turn, coordinator picks `(model, role)`, the LLM answers, verifier may terminate |
| `src/evolution.py` | **sep-CMA-ES** trainer — pure NumPy, ~150 LOC, no external CMA library |
| `src/eval.py` | Compare two coordinators on a task batch |

The MLP router is **1,765 parameters** for the default `hidden=32` config —
well under the paper's "~10K parameters" head.

## How to run

```bash
# smoke tests
python tests/test_smoke.py

# heuristic baseline
python examples/run_heuristic.py

# train MLP router with sep-CMA-ES, then compare
python examples/train_router.py --gens 15 --pop 12 --train 16 --eval 32

# Qwen2-0.5B backbone (optional, needs transformers)
pip install transformers
python examples/run_qwen.py
```

Sample output from a small training run on the toy task suite:

```
heuristic     acc=88%  accept=88%  avg_turns=3.88
mlp-untrained acc=0%   accept=0%   avg_turns=6.00  (random router, workers loop forever)
mlp-trained   acc=100% accept=100% avg_turns=2.00  (learned: Worker_B -> Verifier_B)
```

The trained MLP discovered that for this task mix, skipping the Thinker
(Model_B can both solve and verify cheaply) outperforms the hand-coded
Thinker → Worker → Verifier loop.

## Architecture notes

### Why MLP, not LLM, for the coordinator?

The paper uses a 0.6B Qwen as a *feature extractor* — last-token hidden state
fed to a linear head. That's the right choice when transcripts are long and
rich, and when you have a pre-trained checkpoint handy.

For a prototype, we skip the LLM dependency and hand-engineer a 16-dim
feature vector that captures everything a state-machine router needs to
make decisions. The MLP head is the same size and shape as the paper's
linear head (~1.7K params vs ~10K), and the training procedure
(sep-CMA-ES on rollout reward) is identical.

If you have a small Qwen checkpoint, `QwenCoordinator` is the drop-in
upgrade path.

### Why sep-CMA-ES instead of RL?

From `src/evolution.py` docstring:

- Sparse binary reward (correct / not correct)
- Stochastic LLM outputs make policy gradients high-variance
- The paper's whole point: black-box, noise-tolerant, gradient-free

The implementation collapses the full CMA-ES covariance matrix update to a
per-dimension step-size update (`sigma` is now a vector, not a matrix).
This is O(d) per generation, d ~ 1.7K, so the difference matters.

### Fitness function

```python
score = 1.0  if correct else 0.0
score += 0.3 * (max_turns - turns_used) / max_turns  # early-termination bonus
score += 0.1  if used Verifier at least once          # role-diversity bonus
```

## Limitations (and what the paper does better)

- **No real LLM**: we ship mocks with controllable reliability so the
  coordinator has *something* to learn. With real LLMs the gap between
  "small" and "strong" is more nuanced (cost, latency, capability).
- **No Qwen backbone in the loop**: optional, requires `transformers` and a
  model download.
- **Task suite is small**: ~30 tasks at difficulty 1-4. The paper trains on
  full benchmarks (MATH, HumanEval, etc.).
- **Reward is binary**: full TRINITY probably uses partial credit for
  trajectory-level signals (e.g. "verifier caught the error").
- **No real-CMA-ES library**: we ship a pure-NumPy sep-CMA-ES. For serious
  experiments, swap in `pycma`.

## File map

```
trinity-coordinator/
├── README.md
├── requirements.txt
├── notes/paper_summary.md       # paper notes
├── src/
│   ├── __init__.py
│   ├── tasks.py                  # toy tasks + answer matching
│   ├── llm_pool.py               # mock LLMs
│   ├── features.py               # 16-dim transcript features
│   ├── coordinator.py            # heuristic / MLP / Qwen coordinators
│   ├── trinity_system.py         # Section 4 state machine
│   ├── evolution.py              # sep-CMA-ES
│   └── eval.py                   # comparison harness
├── tests/
│   └── test_smoke.py
├── examples/
│   ├── run_heuristic.py
│   ├── train_router.py
│   └── run_qwen.py
├── data/                         # (for future: real task suites)
├── artifacts/                    # trained params, logs
└── notes/
    └── paper_summary.md
```
