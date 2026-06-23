# TRINITY Coordinator — Prototype

A standalone Python prototype of **TRINITY: An Evolved LLM Coordinator**
(ICLR 2026, Sakana AI, [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)),
focused on Section 3 (state machine + head architectures) and Section 4.8
(sep-CMA-ES vs random search).

## What's in the box

| Module | What it does |
|---|---|
| `src/tasks.py` | Toy tasks (arithmetic / logic / string / two-step) with verifiable answers |
| `src/llm_pool.py` | `MockSmallLLM` (80% reliable) and `MockStrongLLM` (98% reliable) — drop-in interface for real LLMs |
| `src/features.py` | 16-dim transcript feature vector (turn #, last role, reject count, etc.) |
| `src/heads.py` | **4 paper-faithful head architectures** (Appendix A.4): Linear, Low-rank, Sparse, **Block-diagonal** + a generic MLP |
| `src/coordinator.py` | `HeuristicCoordinator`, `MLPCoordinator` (any head), `QwenCoordinator` (optional) |
| `src/trinity_system.py` | The Section 3.2 state machine: per turn, coordinator picks `(model, role)`, the LLM answers, verifier may terminate |
| `src/evolution.py` | **sep-CMA-ES** (pure NumPy, ~150 LOC) + **Random Search baseline** (paper S4.8) |
| `src/eval.py` | Compare two coordinators on a task batch |

## Head architectures (Appendix A.4)

The paper studies 4 head architectures. The default in this repo is
**block-diagonal** (the paper's block-diagonal-10 with argmax output is
their most parameter-efficient choice).

| Head | Param formula | Toy config (d=16, n_a=5) |
|---|---|---|
| **Linear** | $d_h \cdot n_a$ | 80 |
| **Low-rank** (r=14) | $r \cdot d_h + n_a \cdot r$ | 294 |
| **Sparse** | $d_h \cdot n_a + d_h + 2$ | 98 |
| **Block-diagonal-10** | $d_h$ | 16 (one per output) |
| MLP (not in paper) | varies | 1,765 |

Paper Table 6 (real setup: $d_h=1024$, $n_a=10$): linear=10,240 / sparse=11,266 / block-diag-2=5,120 / block-diag-10=**1,024**.

## How to run

```bash
# smoke tests (13/13)
python tests/test_smoke.py

# heuristic baseline
python examples/run_heuristic.py

# train + compare (default: block_diag head, sep-CMA-ES, 15 gens)
python examples/train_router.py

# sweep across head architectures
python examples/train_router.py --head linear --gens 15
python examples/train_router.py --head block_diag --no-argmax --gens 15

# S4.8 reproduction: sep-CMA-ES vs Random Search (multi-seed average)
python examples/reproduce_s4_8.py --gens 20 --train 24 --n-seeds 3

# Qwen2-0.5B backbone (optional, needs transformers)
pip install transformers
python examples/run_qwen.py
```

### Sample run — head-architecture ablation (default settings)

```
heuristic     acc=92%  accept=79%  avg_turns=3.42
linear-untrained   acc=4%   accept=96%  avg_turns=1.83
linear-trained-cma acc=92%  accept=92%  avg_turns=2.92
```

### Sample run — S4.8 reproduction (3 seeds, MLP head)

```
sep-CMA-ES  acc=99.17% ± 1.18%   turns=3.05
RS          acc=94.17% ± 2.36%   turns=2.73
✓ sep-CMA-ES wins by 5.0pp on average (matches paper's claim in S4.8)
```

The trained router learned: "Model_B Worker → Model_B Verifier" — the
strong model is reliable enough that planning is unnecessary on this task mix.

## Architecture notes

### Why engineered features instead of Qwen hidden state?

The paper uses Qwen3-0.6B as a feature extractor (last-token hidden state)
because their 16GB benchmarks need rich representations. For a prototype,
a hand-crafted 16-dim feature vector (turn #, last role, reject count, etc.)
captures the routing signal cheaply. The MLPCoordinator is the same
whether features come from an LLM or engineered code — `QwenCoordinator`
is the drop-in upgrade.

### Why sep-CMA-ES instead of RL?

- Sparse binary reward (correct / not correct)
- Stochastic LLM outputs make policy gradients high-variance
- The paper's whole point: black-box, noise-tolerant, gradient-free
- Theoretical: in block-$\varepsilon$-separable landscapes, sep-CMA-ES
  improves linearly with T while RS improves only as $\ln T$

The implementation collapses CMA-ES's full covariance matrix update to a
per-dimension step-size vector. O(d) per generation, d ~ 1.7K for our MLP.

### Fitness function

```python
score = 1.0            if correct
score += 0.3 * turn_eff                  # early-termination bonus
score += 0.1  if used Verifier           # role-diversity bonus
```

Without the turn-efficiency bonus the model collapses to "Worker loops
until lucky", never learning to use the Verifier. With it, the model
learns to stop as soon as it has a good answer.

## Honest gaps to the paper

| Aspect | Paper | This repo |
|---|---|---|
| SLM backbone | Qwen3-0.6B + SVF (9,216 trainable params) | 16-dim engineered features |
| Head | 4 architectures + argmax/softmax | All 4 implemented + argmax; also MLP for toy regime |
| Model pool | 7 real LLMs (GPT-5, Gemini-2.5-Pro, Claude-4, Gemma-3-27B, DeepSeek-R1-Distill-32B, Qwen3-32B x2) | 2 mocks (small, strong) |
| $n_a$ | 10 (7 models + 3 roles) | 5 (2 models + 3 roles) |
| Max turns K | 5 | 5 (configurable) |
| Population size | $\lceil 4 + 3 \ln n \rceil$ (32 for n=10K) | Auto-computed via `recommended_pop_size` |
| Real benchmarks | MATH500, MMLU, RLPR, LiveCodeBench | Toy suite (arithmetic, logic, string) |
| sep-CMA-ES | ✅ | ✅ |
| Random Search baseline (S4.8) | ✅ (Table 4) | ✅ (implemented, reproduced) |
| REINFORCE baseline | ✅ | ❌ (could add as future) |
| SFT baseline | ✅ | ❌ (could add as future) |

## File map

```
llm-coordinator/
├── README.md
├── requirements.txt
├── notes/paper_summary.md       # paper notes (verified against arXiv:2512.04695v3)
├── src/
│   ├── __init__.py
│   ├── tasks.py                  # toy tasks + answer matching
│   ├── llm_pool.py               # mock LLMs
│   ├── features.py               # 16-dim transcript features
│   ├── heads.py                  # 4 paper head architectures + MLP
│   ├── coordinator.py            # heuristic / MLP / Qwen coordinators
│   ├── trinity_system.py         # Section 3.2 state machine
│   ├── evolution.py              # sep-CMA-ES + RS baseline
│   └── eval.py                   # comparison harness
├── tests/
│   └── test_smoke.py             # 13 tests
├── examples/
│   ├── run_heuristic.py
│   ├── train_router.py
│   ├── reproduce_s4_8.py         # sep-CMA-ES vs RS, multi-seed
│   ├── run_qwen.py
│   └── plot_curve.py
├── data/                         # (for future: real task suites)
├── artifacts/                    # trained params, logs
└── notes/
    └── paper_summary.md
```
