#!/usr/bin/env python3
"""Quick demo: heuristic coordinator on a few toy tasks."""

from __future__ import annotations
import os
import sys
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.coordinator import HeuristicCoordinator
from src.llm_pool import LLMPool
from src.tasks import make_dataset
from src.trinity_system import TrinitySystem
from src.eval import evaluate


def main():
    rng = random.Random(0)
    tasks = make_dataset(8, seed=0, difficulty_range=(1, 4))
    pool = LLMPool()

    print("=== Heuristic Coordinator on 8 toy tasks ===\n")
    report = evaluate(
        lambda: HeuristicCoordinator(),
        "heuristic",
        tasks,
        pool,
        max_turns=6,
    )
    print(f"\nFinal: acc={report.accuracy:.0%}  accept={report.accept_rate:.0%}  "
          f"avg_turns={report.avg_turns:.1f}")

    print("\n--- one full run transcript ---")
    t = tasks[0]
    sys_ = TrinitySystem(HeuristicCoordinator(), pool, max_turns=6)
    r = sys_.solve(t)
    for i, m in enumerate(r.transcript):
        print(f"\n[{i:02d}] {m['role']}: {m['content'][:200]}")
    print(f"\n-> accepted={r.accepted}  correct={r.correct}  final={r.final_answer}")


if __name__ == "__main__":
    main()
