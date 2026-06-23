#!/usr/bin/env python3
"""Qwen-based coordinator example — paper-faithful version.

Uses Qwen3-0.6B-Base as the SLM backbone (paper's choice).
Extracts the SECOND-TO-LAST layer hidden state at the last transcript token
(paper: "h_{n-1}" from the penultimate transformer layer), then routes
through a flat linear head (1024 -> n_outputs).

This script demonstrates the FORWARD pass (untrained head = random routing).
To TRAIN the head, use `train_qwen.py`.

Run:
    python examples/run_qwen.py                         # default: 4 toy tasks
    python examples/run_qwen.py --n 10 --difficulty 1-3
    python examples/run_qwen.py --head linear --block-size 10  # block-diag
"""

from __future__ import annotations
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.llm_pool import LLMPool
from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
from src.tasks import make_dataset
from src.trinity_system import TrinitySystem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base",
                    help="HF model id (paper used Qwen3-0.6B)")
    ap.add_argument("--head", default="linear",
                    choices=["linear", "low_rank", "sparse", "block_diag", "mlp"])
    ap.add_argument("--n-models", type=int, default=2,
                    help="toy: 2 mock LLMs in the pool")
    ap.add_argument("--n-roles", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--difficulty", type=str, default="1-3",
                    help="e.g. '1-3'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu",
                    help="cpu, mps, or cuda")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.difficulty.split("-"))
    pool = LLMPool()
    models = pool.keys
    roles = ["Thinker", "Worker", "Verifier"][:args.n_roles]

    n_outputs = len(models) * len(roles)
    cfg = QwenCoordinatorConfig(
        model_id=args.model,
        head=args.head,
        n_outputs=n_outputs,
        use_argmax=False,
        deterministic=args.deterministic,
        device=args.device,
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=args.deterministic)
    coord.configure_outputs(models, roles)

    print(f"Loading {args.model} on {args.device} ...")
    t0 = time.time()
    # touch a feature to trigger lazy load + warm-up
    _ = coord.router.features("warm up")
    load_time = time.time() - t0
    n_head = coord.num_parameters()
    print(f"Loaded in {load_time:.1f}s. Head: {args.head} ({n_head} trainable params, "
          f"n_outputs={n_outputs})")

    rng_seed = args.seed
    tasks = make_dataset(args.n, seed=rng_seed, difficulty_range=(lo, hi))
    sys_ = TrinitySystem(coord, pool, max_turns=args.max_turns)
    n_correct, n_accept = 0, 0
    total_time = 0.0
    for t in tasks:
        ts = time.time()
        r = sys_.solve(t)
        dt = time.time() - ts
        total_time += dt
        ok = r.correct
        n_correct += int(ok)
        n_accept += int(r.accepted)
        print(f"  {t.id:30s}  accepted={r.accepted}  correct={ok}  final={r.final_answer}  "
              f"turns={r.turns}  decisions={r.decisions}  ({dt:.2f}s)")

    print(f"\n{n_correct}/{len(tasks)} correct, {n_accept}/{len(tasks)} accepted "
          f"(untrained head → random routing). Avg solve time: "
          f"{total_time/max(1,len(tasks)):.2f}s/task")


if __name__ == "__main__":
    main()
