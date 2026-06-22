#!/usr/bin/env python3
"""Train a Qwen3-0.6B-backed linear head with sep-CMA-ES.

This is the paper-faithful training loop:
    - Frozen Qwen3-0.6B-Base backbone (extracts 1024-dim hidden state)
    - Second-to-last layer, last token
    - Trainable head (1024 -> n_outputs): Linear, Block-diagonal, MLP, etc.
    - sep-CMA-ES black-box optimization on fitness (correctness + turn bonus)

Run:
    python examples/train_qwen.py --gens 8 --train 8 --eval 16
    python examples/train_qwen.py --head block_diag --n-blocks 5 --argmax
    python examples/train_qwen.py --method rs --gens 8   # S4.8 baseline
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.coordinator import HeuristicCoordinator
from src.evolution import (
    CMAESConfig, make_fitness_fn, random_search, sep_cma_es,
    recommended_pop_size,
)
from src.eval import compare, evaluate
from src.llm_pool import LLMPool
from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
from src.tasks import make_dataset
from src.trinity_system import TrinitySystem


def build_coord(args, n_outputs: int, models, roles):
    """Build a QwenCoordinator (lazy-loads backbone on first .features())."""
    cfg = QwenCoordinatorConfig(
        model_id=args.model,
        head=args.head,
        n_outputs=n_outputs,
        use_argmax=args.argmax,
        deterministic=True,
        device=args.device,
        dtype=torch.float32,
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=True)
    coord.configure_outputs(models, roles)
    return coord


import torch  # late import so argparse --help runs without torch imported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--head", default="linear",
                    choices=["linear", "low_rank", "sparse", "block_diag", "mlp"])
    ap.add_argument("--argmax", action="store_true", default=False,
                    help="use argmax output (paper's best for block_diag-10)")
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--pop", type=int, default=None,
                    help="auto: paper formula ⌈4+3 ln n⌉")
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--method", type=str, default="cma", choices=["cma", "rs"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--difficulty", default="1-3")
    ap.add_argument("--save", default="artifacts/qwen_head_params.json")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.difficulty.split("-"))

    pool = LLMPool()
    models = pool.keys
    roles = ["Thinker", "Worker", "Verifier"]
    n_outputs = len(models) * len(roles)
    print(f"Pool: {models}, Roles: {roles}, n_outputs: {n_outputs}")

    train_tasks = make_dataset(args.train, seed=args.seed, difficulty_range=(lo, hi))
    eval_tasks = make_dataset(args.eval, seed=args.seed + 9999, difficulty_range=(lo, hi + 1))

    # ---- build the Qwen coord ONCE (loads backbone) and reuse it ----
    coord = build_coord(args, n_outputs, models, roles)
    print("Loading Qwen backbone...")
    t_load = time.time()
    init_params = coord.get_params()  # triggers lazy load
    load_time = time.time() - t_load
    print(f"Loaded in {load_time:.1f}s")
    d = init_params.size
    pop = args.pop or recommended_pop_size(d)
    print(f"Head: {args.head} (d={d} params), pop={pop}, gens={args.gens}, "
          f"train={args.train}, eval={args.eval}, max_turns={args.max_turns}, "
          f"method={args.method}")

    # factory: reuse the SAME loaded coord, just swap head params
    def coord_factory(params):
        coord.set_params(params)
        return coord

    fit_fn = make_fitness_fn(train_tasks, pool, max_turns=args.max_turns,
                             coord_factory=coord_factory)

    es_cfg = CMAESConfig(
        n_dim=d,
        pop_size=pop,
        sigma_init=0.15,
        generations=args.gens,
        fitness_fn=fit_fn,
    )

    print("\n>>> Training (this will run Qwen forward passes — slow on CPU) <<<")
    t0 = time.time()
    if args.method == "cma":
        best = sep_cma_es(es_cfg, init_params, verbose=True)
    else:
        best = random_search(es_cfg, init_params, verbose=True)
    train_time = time.time() - t0
    print(f"\nTraining done in {train_time:.1f}s ({train_time/max(1,args.gens):.1f}s/gen)")

    # ---- save ----
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump({
            "params": best.tolist(),
            "head": args.head,
            "d": d,
            "n_outputs": n_outputs,
            "models": models,
            "roles": roles,
            "model_id": args.model,
            "layer_idx": -2,
            "use_argmax": args.argmax,
            "method": args.method,
            "gens": args.gens,
            "train_time_s": train_time,
        }, f)
    print(f"Saved head params to {args.save}")

    # ---- held-out evaluation ----
    print("\n=== Held-out evaluation ===")
    def make_trained():
        c = build_coord(args, n_outputs, models, roles)
        c.set_params(best)
        return c
    def make_untrained():
        return build_coord(args, n_outputs, models, roles)
    def make_heur():
        return HeuristicCoordinator()

    reports = compare(
        [
            ("heuristic", make_heur),
            (f"{args.head}-untrained", make_untrained),
            (f"{args.head}-trained-{args.method}", make_trained),
        ],
        f"Qwen3 {args.head} head",
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
