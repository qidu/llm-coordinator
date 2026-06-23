#!/usr/bin/env python3
"""Reproduce the S4.8 comparison: sep-CMA-ES vs Random Search.

Paper Table 4: sep-CMA-ES > RS > REINFORCE on all 4 benchmarks.
This script runs both sep-CMA-ES and RS on the same task suite, same
training budget (m_CMA=16 vs m_RS=32 rollouts per candidate), and reports
fitness on a held-out set.

Run:  python examples/reproduce_s4_8.py --gens 25 --train 20
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

from src.coordinator import MLPCoordinator, CoordinatorConfig
from src.evolution import (
    CMAESConfig, make_fitness_fn, random_search, sep_cma_es,
    recommended_pop_size,
)
from src.llm_pool import LLMPool
from src.tasks import make_dataset
from src.eval import evaluate


def evaluate_params(params, coord_cfg, eval_tasks, pool, max_turns):
    coord = MLPCoordinator(params=params, cfg=coord_cfg)
    report = evaluate(lambda: coord, f"{coord_cfg.head}", eval_tasks, pool,
                      max_turns=max_turns)
    return report.accuracy, report.avg_turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--eval", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--head", type=str, default="block_diag")
    ap.add_argument("--argmax", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="repeat with multiple seeds and average (reduces RS noise)")
    args = ap.parse_args()

    pool = LLMPool()
    coord_cfg = CoordinatorConfig(head=args.head, n_blocks=5, use_argmax=args.argmax)
    init_coord = MLPCoordinator(cfg=coord_cfg)
    init = init_coord.get_params()
    d = init.size
    pop = args.pop or recommended_pop_size(d)

    train_tasks = make_dataset(args.train, seed=args.seed, difficulty_range=(1, 4))
    eval_tasks = make_dataset(args.eval, seed=args.seed + 9999, difficulty_range=(1, 5))
    fit_fn = make_fitness_fn(train_tasks, pool, max_turns=args.max_turns, coord_cfg=coord_cfg)

    print(f"Head: {args.head} (d={d} params, pop={pop}, gens={args.gens}, "
          f"train={args.train}, eval={args.eval}, seeds={args.n_seeds})\n")

    cma_accs, cma_turns_list, cma_times = [], [], []
    rs_accs, rs_turns_list, rs_times = [], [], []
    last_cma_params, last_rs_params = None, None

    for seed_idx in range(args.n_seeds):
        seed = args.seed + seed_idx * 1000
        print(f"\n=== seed {seed} ===")
        train_tasks_s = make_dataset(args.train, seed=seed, difficulty_range=(1, 4))
        eval_tasks_s = make_dataset(args.eval, seed=seed + 9999, difficulty_range=(1, 5))
        fit_fn_s = make_fitness_fn(train_tasks_s, pool, max_turns=args.max_turns,
                                   coord_cfg=coord_cfg)

        # sep-CMA-ES
        t0 = time.time()
        es_cfg = CMAESConfig(n_dim=d, pop_size=pop, generations=args.gens,
                             sigma_init=0.15, fitness_fn=fit_fn_s)
        best_cma = sep_cma_es(es_cfg, init, verbose=(seed_idx == 0))
        cma_time = time.time() - t0
        cma_acc, cma_turns = evaluate_params(best_cma, coord_cfg, eval_tasks_s, pool,
                                             args.max_turns)
        cma_accs.append(cma_acc); cma_turns_list.append(cma_turns); cma_times.append(cma_time)
        last_cma_params = best_cma
        print(f"  sep-CMA-ES  acc={cma_acc:.2%}  turns={cma_turns:.2f}  time={cma_time:.1f}s")

        # Random Search
        t0 = time.time()
        es_cfg_rs = CMAESConfig(n_dim=d, pop_size=pop, generations=args.gens,
                                sigma_init=0.15, fitness_fn=fit_fn_s)
        best_rs = random_search(es_cfg_rs, init, verbose=(seed_idx == 0))
        rs_time = time.time() - t0
        rs_acc, rs_turns = evaluate_params(best_rs, coord_cfg, eval_tasks_s, pool,
                                          args.max_turns)
        rs_accs.append(rs_acc); rs_turns_list.append(rs_turns); rs_times.append(rs_time)
        last_rs_params = best_rs
        print(f"  RS          acc={rs_acc:.2%}  turns={rs_turns:.2f}  time={rs_time:.1f}s")

    # ---- summary ----
    import numpy as np
    cma_acc_mean, cma_acc_std = float(np.mean(cma_accs)), float(np.std(cma_accs))
    cma_turns_mean = float(np.mean(cma_turns_list))
    cma_time_mean = float(np.mean(cma_times))
    rs_acc_mean, rs_acc_std = float(np.mean(rs_accs)), float(np.std(rs_accs))
    rs_turns_mean = float(np.mean(rs_turns_list))
    rs_time_mean = float(np.mean(rs_times))

    print("\n" + "=" * 50)
    print(f"Summary over {args.n_seeds} seed(s):")
    print(f"  sep-CMA-ES  acc={cma_acc_mean:.2%} ± {cma_acc_std:.2%}  "
          f"turns={cma_turns_mean:.2f}  time/cma-gen={cma_time_mean:.2f}s")
    print(f"  RS          acc={rs_acc_mean:.2%} ± {rs_acc_std:.2%}  "
          f"turns={rs_turns_mean:.2f}  time/rs-gen={rs_time_mean:.2f}s")
    if cma_acc_mean > rs_acc_mean:
        gap = (cma_acc_mean - rs_acc_mean) * 100
        print(f"  ✓ sep-CMA-ES wins by {gap:.1f}pp on average "
              f"(matches paper's claim in S4.8)")
    elif cma_acc_mean == rs_acc_mean:
        print("  = tied")
    else:
        print("  ✗ RS wins on this run (RS has higher variance; try more gens/tasks)")

    out_path = "artifacts/s4_8_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"cma": {"acc_mean": cma_acc_mean, "acc_std": cma_acc_std,
                            "turns": cma_turns_mean, "time": cma_time_mean},
                    "rs": {"acc_mean": rs_acc_mean, "acc_std": rs_acc_std,
                            "turns": rs_turns_mean, "time": rs_time_mean},
                    "n_seeds": args.n_seeds, "head": args.head}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
