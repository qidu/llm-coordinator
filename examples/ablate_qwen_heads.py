#!/usr/bin/env python3
"""Head-architecture ablation on Qwen3-0.6B features.

Train all 5 head architectures (linear, low_rank, sparse, block_diag, mlp)
on the same task suite and report train + held-out accuracy.

Run:  python examples/ablate_qwen_heads.py --gens 10 --train 12 --eval 20
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import json
import torch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.coordinator import HeuristicCoordinator
from src.evolution import (
    CMAESConfig, make_fitness_fn, random_search, sep_cma_es,
    recommended_pop_size, make_batched_qwen_fitness_fn,
)
from src.eval import evaluate
from src.llm_pool import LLMPool
from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
from src.tasks import make_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--method", default="cma", choices=["cma", "rs"])
    ap.add_argument("--no-batched", action="store_true",
                    help="Disable the batched fitness path (slower but "
                         "byte-for-byte identical to the naive per-candidate loop).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                    help="Compute device for the Qwen backbone. "
                         "cuda for NVIDIA GPUs, mps for Apple Silicon.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                    help="Model precision. float16/bfloat16 halves memory + speeds up MPS.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heads", nargs="+",
                    default=["linear", "block_diag", "mlp"],
                    help="head architectures to compare")
    ap.add_argument("--save", default="artifacts/qwen_head_ablation.json")
    args = ap.parse_args()

    pool = LLMPool()
    models = pool.keys
    roles = ["Thinker", "Worker", "Verifier"]
    n_outputs = len(models) * len(roles)

    train_tasks = make_dataset(args.train, seed=args.seed, difficulty_range=(1, 3))
    eval_tasks = make_dataset(args.eval, seed=args.seed + 9999, difficulty_range=(1, 4))

    results = {}
    for head in args.heads:
        print(f"\n{'='*60}\nHead: {head}\n{'='*60}")
        cfg = QwenCoordinatorConfig(
            model_id=args.model,
            head=head,
            n_outputs=n_outputs,
            use_argmax=False,
            deterministic=True,
            device=args.device,
            dtype={"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[args.dtype],
        )
        coord = QwenCoordinator(cfg=cfg, deterministic=True)
        coord.configure_outputs(models, roles)

        # load backbone once per head (it's cached, so ~0s after first)
        t0 = time.time()
        init_params = coord.get_params()
        load_time = time.time() - t0
        d = init_params.size
        pop = recommended_pop_size(d)
        print(f"  d={d} params, pop={pop}, device={args.device}, "
              f"dtype={args.dtype}, load={load_time:.1f}s")

        def factory(params):
            coord.set_params(params)
            return coord

        # ---- build the fitness function ----
        # By default use the batched path: one Qwen3 forward per (turn)
        # over all (pop, task) pairs, instead of pop * tasks forwards.
        # The batched path gives ~2x speedup at small pop and ~1.2x at
        # larger pop on CPU. With --no-batched we fall back to the naive
        # per-candidate loop (useful as a correctness sanity check).
        if args.no_batched:
            fit_naive = make_fitness_fn(
                train_tasks, pool, max_turns=args.max_turns,
                coord_factory=factory,
            )
            es_cfg = CMAESConfig(n_dim=d, pop_size=pop, generations=args.gens,
                                 sigma_init=0.15, fitness_fn=fit_naive)
        else:
            fit_batched = make_batched_qwen_fitness_fn(
                train_tasks, pool, max_turns=args.max_turns,
                coord_template=coord,
            )
            es_cfg = CMAESConfig(n_dim=d, pop_size=pop, generations=args.gens,
                                 sigma_init=0.15, batched_fitness_fn=fit_batched)
        ts = time.time()
        if args.method == "cma":
            best = sep_cma_es(es_cfg, init_params, verbose=False)
        else:
            best = random_search(es_cfg, init_params, verbose=False)
        train_time = time.time() - ts
        print(f"  trained in {train_time:.1f}s ({train_time/max(1,args.gens):.1f}s/gen)")

        # evaluate
        coord.set_params(best)
        eval_report = evaluate(lambda: coord, f"{head}-trained", eval_tasks, pool,
                               max_turns=args.max_turns)
        print(f"  -> eval acc={eval_report.accuracy:.2%}  accept={eval_report.accept_rate:.2%}  "
              f"avg_turns={eval_report.avg_turns:.2f}")
        results[head] = {
            "acc": eval_report.accuracy,
            "accept": eval_report.accept_rate,
            "avg_turns": eval_report.avg_turns,
            "train_time_s": train_time,
            "d": d,
            "pop": pop,
        }

    # ---- summary ----
    print("\n" + "=" * 60)
    print(f"Qwen3-0.6B Head-Architecture Ablation  (gens={args.gens}, "
          f"train={args.train}, eval={args.eval}, method={args.method})")
    print("=" * 60)
    print(f"{'Head':12s}  {'d':>6s}  {'pop':>3s}  {'acc':>6s}  {'accept':>6s}  "
          f"{'turns':>6s}  {'time':>6s}")
    for head, r in results.items():
        print(f"  {head:10s}  {r['d']:>6d}  {r['pop']:>3d}  {r['acc']:>5.1%}  "
              f"{r['accept']:>5.1%}  {r['avg_turns']:>6.2f}  {r['train_time_s']:>5.1f}s")

    # best head
    best_head = max(results, key=lambda h: results[h]["acc"])
    print(f"\nBest head: {best_head} ({results[best_head]['acc']:.1%} acc)")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump({"results": results, "best": best_head,
                   "gens": args.gens, "train": args.train,
                   "eval": args.eval, "method": args.method,
                   "model": args.model}, f, indent=2)
    print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
