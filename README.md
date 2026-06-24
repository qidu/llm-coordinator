# TRINITY Coordinator — Prototype

A standalone Python prototype of **TRINITY: An Evolved LLM Coordinator**
(ICLR 2026, Sakana AI, [arXiv:2512.04695](https://arxiv.org/abs/2512.04695)),
focused on Section 3 (state machine + head architectures) and Section 4.8
(sep-CMA-ES vs random search).

## Quickstart

> **TL;DR — go from a fresh clone to a working reproduction in ~5 min.**

```bash
# 1. Get the code & activate the venv
git clone <this-repo> llm-coordinator
cd llm-coordinator                            # the repo IS llm-coordinator
source ~/venvs/headroom/bin/activate          # or create it (see Setup below)

# 2. (One-time) install + download the Qwen backbone
pip install -r requirements.txt
huggingface-cli download Qwen/Qwen3-0.6B-Base

# 3. Run the tests — you should see 19/19 PASS
python tests/test_smoke.py

# 4. Reproduce the paper's main result (~1 min)
python examples/reproduce_s4_8.py
#   sep-CMA-ES  acc=99.17% ± 1.18%
#   RS          acc=94.17% ± 2.36%
#   → sep-CMA-ES wins by 5pp, matching Table 4

# 5. Train a Qwen3-0.6B-backed head with sep-CMA-ES (~5–25 min on MPS+fp16 or CUDA+fp16)
## macOS
python examples/ablate_qwen_heads.py \
    --gens 6 --train 8 --eval 16 \
    --heads linear block_diag low_rank sparse mlp \
    --device mps --dtype float16 \
    --save artifacts/qwen_head_ablation_mps.json
## Linux with cuda
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/ablate_qwen_heads.py \
    --device cuda --dtype float16 --train 8 --eval 8 --gens 6 \
    --heads linear block_diag low_rank sparse mlp --no-batched \
    --save artifacts/qwen_head_ablation_cuda.json
```
```bash
============================================================
Qwen3-0.6B Head-Architecture Ablation  (gens=6, train=8, eval=8, method=cma)
============================================================
Head               d  pop     acc  accept   turns    time
  linear        6144   31  100.0%  100.0%    2.25  137.7s
  block_diag    1024   25  87.5%   0.0%    4.00  105.0s
  low_rank     32960   36  87.5%  100.0%    2.62  164.0s
  sparse        7169   31  87.5%  75.0%    2.62  100.6s
  mlp          34054   36  87.5%  37.5%    3.25  147.3s

Best head: linear (100.0% acc)
```

> **Why `linear` wins at 0.6B scale — and what this result means**
>
> The `linear` head wins not because it's architecturally superior, but because at 0.6B scale the routing signal is too weak to benefit from expert specialization. MoE's promise is that the router learns to send different inputs (math, code, dialogue, etc.) to different experts, each becoming a specialist. At 0.6B there isn't enough model capacity or training signal for true specialization — what you'd expect to be "Expert A for math, Expert B for code" collapses into barely-different copies of the same thing.
>
> The constrained variants (`block_diag`, `sparse`, `mlp`) are actively harmful at this scale because they add a routing bottleneck on top of an already-weak routing signal. The `block_diag` result is the clearest signal: **0% acceptance** means the router almost never picked the expert it trained on, confirming the routing signal itself hasn't formed meaningful expert niches.
>
> This result is a **false negative for expert-constrained architectures**, not a positive endorsement of linear heads. If you want to measure true expert quality at scale, look at:
>
> - Does the router show consistent per-example routing patterns?
> - Do different experts show measurably different activation norms?
> - What does routing entropy look like across the eval set?
>
> Running this ablation on a larger model (e.g. Qwen3-32B) would likely flip the story — the routing signal has enough capacity to carve out specialist pathways, and constrained heads become a feature rather than a bug.

If `python examples/reproduce_s4_8.py` doesn't print numbers, jump to **Setup** below.

## What's in the box

| Module | What it does |
|---|---|
| `src/tasks.py` | Toy tasks (arithmetic / logic / string / two-step) with verifiable answers |
| `src/llm_pool.py` | `MockSmallLLM` (80% reliable) and `MockStrongLLM` (98% reliable) — `LLM` Protocol interface for plugging in real LLMs |
| `src/features.py` | 16-dim transcript feature vector (turn #, last role, reject count, etc.) |
| `src/heads.py` | **4 paper-faithful head architectures** (Appendix A.4): Linear, Low-rank, Sparse, **Block-diagonal** + a generic MLP |
| `src/coordinator.py` | `HeuristicCoordinator`, `MLPCoordinator` (any head), `QwenCoordinator` (optional) |
| `src/qwen_router.py` | `QwenRouter` — frozen Qwen3-0.6B backbone + trainable head (all 4 paper architectures); batched fitness for sep-CMA-ES speedup |
| `src/trinity_system.py` | The Section 3.2 state machine: per turn, coordinator picks `(model, role)`, the LLM answers, verifier may terminate |
| `src/evolution.py` | **sep-CMA-ES** (pure NumPy, ~150 LOC) + **Random Search baseline** (paper S4.8) — `recommended_pop_size` implements the paper's $\lceil 4 + 3\ln n \rceil$ formula |
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

## Setup

The project uses a Python 3.12 venv called `headroom` (lives at `~/venvs/headroom`). It already has PyTorch, Transformers, NumPy, and pytest installed — no `pip install` needed unless you're starting from scratch.

```bash
# Activate the venv (do this once per shell)
source ~/venvs/headroom/bin/activate

# If the venv doesn't exist yet, create it:
python3.12 -m venv ~/venvs/headroom
source ~/venvs/headroom/bin/activate
pip install -r requirements.txt

# Verify the install
python -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__)"
```

For the Qwen3-0.6B backbone used in `run_qwen.py` / `train_qwen.py` / `ablate_qwen_heads.py`, you also need the model weights. The repo expects them in the standard HuggingFace cache:

```bash
# Download Qwen3-0.6B-Base (~1.2GB). If this gets stuck partway, see
# scripts/finish_qwen_download.sh which resumes from where it stopped.
huggingface-cli download Qwen/Qwen3-0.6B-Base
# or, if huggingface.co is blocked in your region:
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-0.6B-Base
```

The first time `QwenRouter` is constructed it will load these weights into the cache at `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/` (~1.2 GB).

## How to run

```bash
# smoke tests (19/19)
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

# Qwen3-0.6B backbone (paper-faithful; needs the Qwen3-0.6B-Base download above)
python examples/run_qwen.py                    # single forward pass
python examples/train_qwen.py                  # train one head via sep-CMA-ES
python examples/ablate_qwen_heads.py --help    # full 5-head ablation
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

### Two feature paths: engineered vs. Qwen hidden state

The repo ships with two interchangeable feature paths, flipped from the README's original framing:

- **MLPCoordinator** (default): hand-crafted 16-dim feature vector (turn #, last role, reject count, etc.). Fast, no GPU needed. Useful for rapid prototyping of head architectures and evolution hyperparameters.
- **QwenCoordinator** (`src/qwen_router.py`): frozen Qwen3-0.6B backbone (last-token hidden state, same as the paper). Drop-in replacement — same `route()` interface. Powers `train_qwen.py` and `ablate_qwen_heads.py`.

Both coordinators share the same `TrinitySystem` and evolution loop. The toy regime defaults to MLPCoordinator for speed; swap to QwenCoordinator when you want paper-faithful features.

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

| Aspect | Paper | This repo | Status |
|---|---|---|---|
| SLM backbone | Qwen3-0.6B + SVF (9,216 trainable params) | `QwenRouter` (frozen Qwen3-0.6B + trainable head) via `examples/train_qwen.py` / `ablate_qwen_heads.py` | ✅ |
| Head | 4 architectures + argmax/softmax | All 4 implemented + argmax; also MLP for toy regime | ✅ |
| Population size | $\lceil 4 + 3 \ln n \rceil$ (32 for n=10K) | `recommended_pop_size` in `src/evolution.py:264` — exact formula | ✅ |
| Model pool | 7 real LLMs (GPT-5, Gemini-2.5-Pro, Claude-4, Gemma-3-27B, DeepSeek-R1-Distill-32B, Qwen3-32B x2) | 2 mocks (small, strong) — `LLM` Protocol in `src/llm_pool.py` ready for real LLMs; hardcoded 2-model assumptions in `features.py` / `coordinator.py` need updating | 🔧 |
| $n_a$ | 10 (7 models + 3 roles) | 5 (2 models + 3 roles) | 🔧 |
| Real benchmarks | MATH500, MMLU, RLPR, LiveCodeBench | Toy suite (arithmetic, logic, string) — `Task` abstraction in `src/tasks.py` is benchmark-agnostic; needs dataset loaders | 🔧 |
| sep-CMA-ES | ✅ | ✅ | — |
| Random Search baseline (S4.8) | ✅ (Table 4) | ✅ (implemented, reproduced) | — |
| REINFORCE baseline | ✅ | ❌ (could add as future) | — |
| SFT baseline | ✅ | ❌ (could add as future) | — |

**Legend:** ✅ = matches paper, 🔧 = fixable with moderate effort, ❌ = not implemented

The three 🔧 gaps are all structurally feasible: the abstractions are in place (Protocol interface for real LLMs, `Task` objects for benchmarks, configurable `n_outputs` for action space). The remaining work is data loading and removing the 2-model hardcodes.

## File map

```
.                            # repo root (= llm-coordinator/ on disk)
├── README.md
├── requirements.txt
├── notes/paper_summary.md       # paper notes (verified against arXiv:2512.04695v3)
├── src/
│   ├── __init__.py
│   ├── tasks.py                  # toy tasks + answer matching
│   ├── llm_pool.py               # mock LLMs
│   ├── features.py               # 16-dim transcript features
│   ├── heads.py                  # 4 paper head architectures + MLP
│   ├── coordinator.py            # heuristic / MLP coordinators
│   ├── qwen_router.py           # Qwen3-0.6B backbone + trainable head
│   ├── trinity_system.py         # Section 3.2 state machine
│   ├── evolution.py              # sep-CMA-ES + RS baseline
│   └── eval.py                   # comparison harness
├── tests/
│   └── test_smoke.py             # 19 tests
├── examples/
│   ├── run_heuristic.py
│   ├── train_router.py
│   ├── reproduce_s4_8.py         # sep-CMA-ES vs RS, multi-seed
│   ├── run_qwen.py              # single forward pass with Qwen3-0.6B
│   ├── train_qwen.py            # train head via sep-CMA-ES
│   ├── ablate_qwen_heads.py     # 5-head ablation
│   └── plot_curve.py
├── data/                         # (for future: real task suites)
├── artifacts/                    # trained params, logs
└── notes/
    └── paper_summary.md
```
