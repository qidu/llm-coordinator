#!/usr/bin/env python3
"""Train the MLP coordinator with sep-CMA-ES, then evaluate against heuristic.

Usage:
    python examples/train_router.py            # quick run
    python examples/train_router.py --gens 30 --pop 16 --train 24
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.coordinator import HeuristicCoordinator, MLPCoordinator
from src.evolution import train_router
from src.eval import compare
from src.llm_pool import LLMPool
from src.tasks import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--train", type=int, default=12, help="training tasks per gen")
    ap.add_argument("--eval", type=int, default=24, help="held-out eval tasks")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="artifacts/router_params.json")
    args = ap.parse_args()

    pool = LLMPool()

    t0 = time.time()
    best_params, logs = train_router(
        pool,
        n_train=args.train,
        n_eval=args.eval,
        pop_size=args.pop,
        generations=args.gens,
        max_turns=args.max_turns,
        hidden=args.hidden,
        seed=args.seed,
        save_path=args.save,
    )
    train_time = time.time() - t0

    print(f"\n=== Training done in {train_time:.1f}s ===")
    print(f"Final train fitness: {logs[-1].best:.3f}")

    # ---- compare on held-out ----
    eval_tasks = make_dataset(args.eval, seed=args.seed + 9999, difficulty_range=(1, 5))

    def make_mlp():
        c = MLPCoordinator(params=best_params, deterministic=True)
        return c

    def make_heur():
        return HeuristicCoordinator()

    def make_random_mlp():
        c = MLPCoordinator(deterministic=True)
        return c

    print("\n=== Comparison on held-out tasks ===")
    reports = compare(
        [
            ("heuristic", make_heur),
            ("mlp-untrained", make_random_mlp),
            ("mlp-trained", make_mlp),
        ],
        "trained vs heuristic",
        eval_tasks,
        pool,
        max_turns=args.max_turns,
    )
    print("\nSummary:")
    for r in reports:
        print(f"  {r.name:20s}  acc={r.accuracy:.0%}  accept={r.accept_rate:.0%}  "
              f"avg_turns={r.avg_turns:.2f}")


if __name__ == "__main__":
    main()
