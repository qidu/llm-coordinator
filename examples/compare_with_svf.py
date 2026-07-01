#!/usr/bin/env python3
"""Compare frozen-backbone vs SVF-backbone on the head-architecture ablation.

The key question: does letting the backbone learn (via SVF — paper Section 3.1)
change which head architecture wins? Or does the "linear wins at 0.6B"
observation (README §Honest gaps) hold even with the routing signal boosted?

We train each (head, svf) configuration for a fixed compute budget, on the
same task suite, and report held-out accuracy.

Run:
    python examples/compare_with_svf.py \\
        --gens 6 --train 8 --eval 16 \\
        --heads linear block_diag low_rank sparse mlp \\
        --device cpu

The output is a markdown-friendly table comparing frozen vs SVF accuracy
side-by-side, with trainable-parameter counts next to each result.
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
    CMAESConfig, recommended_pop_size, sep_cma_es,
    make_batched_qwen_fitness_fn,
)
from src.eval import evaluate
from src.llm_pool import LLMPool
from src.qwen_router import QwenCoordinator, QwenCoordinatorConfig
from src.tasks import make_dataset


def run_one(head: str, use_svf: bool, args, layer_idxs, models, roles,
            n_outputs, train_tasks, eval_tasks, pool):
    cfg = QwenCoordinatorConfig(
        model_id=args.model,
        head=head,
        n_outputs=n_outputs,
        use_argmax=False,
        deterministic=True,
        layer_idxs=layer_idxs,
        use_svf=use_svf,
        svf_rank=args.svf_rank,
        svf_n_blocks=args.svf_n_blocks,
        device=args.device,
        dtype={"float32": torch.float32, "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype],
    )
    coord = QwenCoordinator(cfg=cfg, deterministic=True)
    coord.configure_outputs(models, roles)
    t0 = time.time()
    init_params = coord.get_params()
    load_time = time.time() - t0
    d = init_params.size
    pop = recommended_pop_size(d)
    n_head = coord.router.num_head_parameters()
    n_svf = coord.router.num_svf_parameters()
    print(f"  head={head:10s} svf={use_svf}  d={d:>6d} (head={n_head}, "
          f"svf={n_svf}) pop={pop} load={load_time:.1f}s")

    def factory(p):
        coord.set_params(p)
        return coord

    fit_batched = make_batched_qwen_fitness_fn(
        train_tasks, pool, max_turns=args.max_turns,
        coord_template=coord,
    )
    es_cfg = CMAESConfig(
        n_dim=d, pop_size=pop, generations=args.gens,
        sigma_init=0.15, batched_fitness_fn=fit_batched,
    )
    ts = time.time()
    best = sep_cma_es(es_cfg, init_params, verbose=False)
    train_time = time.time() - ts

    coord.set_params(best)
    rep = evaluate(lambda: coord, f"{head}{'-svf' if use_svf else '-frozen'}",
                   eval_tasks, pool, max_turns=args.max_turns)
    return {
        "head": head, "use_svf": use_svf,
        "acc": rep.accuracy, "accept": rep.accept_rate,
        "avg_turns": rep.avg_turns,
        "d": d, "n_head": n_head, "n_svf": n_svf,
        "train_time_s": train_time, "gens": args.gens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--eval", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--layers", default="-2",
                    help="Comma-separated layer indices, default '-2' (paper).")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--svf-rank", type=int, default=1024,
                    help="Top-r singular values per SVF target. "
                         "Default 1024 (paper's per-matrix budget for 0.6B).")
    ap.add_argument("--svf-n-blocks", type=int, default=9,
                    help="Number of trailing transformer blocks to wrap. "
                         "Default 9 → 9 * 1024 = 9216 SVF params (paper's "
                         "exact count for 0.6B).")
    ap.add_argument("--heads", nargs="+",
                    default=["linear", "block_diag", "mlp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="artifacts/svf_vs_frozen.json")
    args = ap.parse_args()

    layer_idxs = tuple(int(x) for x in args.layers.split(","))
    pool = LLMPool()
    models = pool.keys
    roles = ["Thinker", "Worker", "Verifier"]
    n_outputs = len(models) * len(roles)

    train_tasks = make_dataset(args.train, seed=args.seed, difficulty_range=(1, 3))
    eval_tasks = make_dataset(args.eval, seed=args.seed + 9999,
                              difficulty_range=(1, 4))

    print(f"\n{'='*70}")
    print(f"SVF vs FROZEN backbone — head ablation on {args.model}")
    print(f"  layers={layer_idxs}  svf_rank={args.svf_rank}  "
          f"svf_n_blocks={args.svf_n_blocks}")
    print(f"  gens={args.gens}  train={args.train}  eval={args.eval}  "
          f"max_turns={args.max_turns}")
    print(f"{'='*70}\n")

    results = []
    for head in args.heads:
        print(f"\n--- head={head} ---")
        for use_svf in (False, True):
            r = run_one(head, use_svf, args, layer_idxs, models, roles,
                        n_outputs, train_tasks, eval_tasks, pool)
            results.append(r)
            print(f"    → acc={r['acc']:.2%}  accept={r['accept']:.2%}  "
                  f"turns={r['avg_turns']:.2f}  time={r['train_time_s']:.1f}s")

    # ---- side-by-side summary ----
    print(f"\n{'='*70}")
    print("Summary: FROZEN vs SVF backbone (Qwen3-0.6B)")
    print(f"{'='*70}")
    print(f"{'Head':10s}  {'FROZEN acc':>11s}  {'SVF acc':>9s}  "
          f"{'Δpp':>5s}  {'#head':>5s}  {'#svf':>5s}")
    by_head = {}
    for r in results:
        by_head.setdefault(r["head"], {})[r["use_svf"]] = r
    for head, pair in by_head.items():
        fr = pair.get(False, {}).get("acc")
        sv = pair.get(True, {}).get("acc")
        n_h = pair[False]["n_head"]
        n_s = pair[True]["n_svf"]
        delta = (sv - fr) * 100 if (sv is not None and fr is not None) else 0.0
        marker = ""
        if sv is not None and fr is not None:
            if sv > fr + 0.01:
                marker = " ✓ SVF wins"
            elif sv < fr - 0.01:
                marker = " ✗ SVF hurts"
        print(f"  {head:8s}  "
              f"{(f'{fr:.1%}' if fr is not None else 'N/A'):>11s}  "
              f"{(f'{sv:.1%}' if sv is not None else 'N/A'):>9s}  "
              f"{delta:>+5.1f}  {n_h:>5d}  {n_s:>5d}{marker}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    with open(args.save, "w") as f:
        json.dump({
            "args": vars(args),
            "layer_idxs": list(layer_idxs),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
