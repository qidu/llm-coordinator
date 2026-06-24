#!/usr/bin/env python3
"""Train the coordinator with sep-CMA-ES or RS baseline, then evaluate.

Reproduces the S4.8 comparison from the paper (RS vs sep-CMA-ES) on the
toy task suite, plus a head-architecture ablation (linear / block_diag / mlp).

Usage:
    python examples/train_router.py                          # quick run
    python examples/train_router.py --gens 20 --pop 16
    python examples/train_router.py --head block_diag --argmax
    python examples/train_router.py --method rs --gens 20    # RS baseline
    python examples/train_router.py --ablate                 # head ablation
"""

from __future__ import annotations
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.coordinator import HeuristicCoordinator, MLPCoordinator, CoordinatorConfig
from src.evolution import (
    train_router, random_search, CMAESConfig, make_fitness_fn,
    recommended_pop_size,
)
from src.eval import compare
from src.features import set_model_keys
from src.llm_pool import LLMPool, make_real_pool
from src.tasks import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--pop", type=int, default=None,
                    help="population size (default: paper formula ⌈4+3 ln n⌉)")
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--eval", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=5,
                    help="paper default K=5")
    ap.add_argument("--head", type=str, default="block_diag",
                    choices=["linear", "low_rank", "sparse", "block_diag", "mlp"])
    ap.add_argument("--n-blocks", type=int, default=None,
                    help="block-diag blocks (default = n_outputs)")
    ap.add_argument("--argmax", action="store_true", default=True,
                    help="use argmax output (paper's best for block_diag)")
    ap.add_argument("--no-argmax", dest="argmax", action="store_false")
    ap.add_argument("--method", type=str, default="cma",
                    choices=["cma", "rs"],
                    help="training algorithm: sep-CMA-ES or random search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="artifacts/router_params.json")
    ap.add_argument("--ablate", action="store_true",
                    help="run a head-architecture ablation instead of comparison")
    ap.add_argument("--real", action="store_true",
                    help="use real LLMs (deepseek-v4-flash + max-m3 on localhost:8788) "
                         "instead of mocks")
    ap.add_argument("--endpoint", type=str, default="http://localhost:8788/v1",
                    help="API endpoint for --real mode")
    ap.add_argument("--benchmark", type=str, default=None,
                    choices=["math500", "mmlu", "livecodebench"],
                    help="use a real benchmark instead of toy tasks")
    ap.add_argument("--bench-cache", type=str, default="data",
                    help="cache directory for benchmark downloads")
    args = ap.parse_args()

    # Wire real LLMs before instantiating any coordinator.
    if args.real:
        pool = make_real_pool(endpoint=args.endpoint)
        set_model_keys(pool.keys)
        print(f"[real] pool={pool.keys}  endpoint={args.endpoint}")
    else:
        pool = LLMPool()
        set_model_keys(pool.keys)
        print(f"[mock] pool={pool.keys}")

    # Wire real benchmarks if requested.
    if args.benchmark:
        from src.benchmarks import load_benchmark
        train_tasks = load_benchmark(args.benchmark, cache_dir=args.bench_cache)
        eval_tasks = load_benchmark(args.benchmark, cache_dir=args.bench_cache)
        print(f"[benchmark] {args.benchmark}: {len(train_tasks)} train, "
              f"{len(eval_tasks)} eval tasks")
        # re-seed for reproducibility
        import random
        rng = random.Random(args.seed)
        rng.shuffle(train_tasks)
        rng.shuffle(eval_tasks)
    else:
        train_tasks = make_dataset(args.train, seed=args.seed, difficulty_range=(1, 4))
        eval_tasks = make_dataset(args.eval, seed=args.seed + 9999, difficulty_range=(1, 5))

    t0 = time.time()
    best, logs, coord_cfg = train_router(
        pool,
        n_train=args.train,
        n_eval=args.eval,
        pop_size=args.pop,
        generations=args.gens,
        max_turns=args.max_turns,
        head=args.head,
        n_blocks=args.n_blocks or 5,
        use_argmax=args.argmax,
        method=args.method,
        seed=args.seed,
        save_path=args.save,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
    )
    train_time = time.time() - t0

    print(f"\n=== {args.method.upper()} training done in {train_time:.1f}s ===")
    print(f"Head: {args.head} (n_blocks={coord_cfg.n_blocks}, argmax={args.argmax})")
    print(f"Final train fitness: {logs[-1].best:.3f}")

    # ---- held-out evaluation ----
    def make_mlp():
        return MLPCoordinator(params=best, cfg=coord_cfg)

    def make_heur():
        return HeuristicCoordinator()

    def make_random_mlp():
        return MLPCoordinator(cfg=coord_cfg)

    print("\n=== Comparison on held-out tasks ===")
    reports = compare(
        [
            ("heuristic", make_heur),
            (f"{args.head}-untrained", make_random_mlp),
            (f"{args.head}-trained-{args.method}", make_mlp),
        ],
        f"{args.method} on {args.head}",
        eval_tasks,
        pool,
        max_turns=args.max_turns,
    )
    print("\nSummary:")
    for r in reports:
        print(f"  {r.name:32s}  acc={r.accuracy:.0%}  accept={r.accept_rate:.0%}  "
              f"avg_turns={r.avg_turns:.2f}")


if __name__ == "__main__":
    main()
