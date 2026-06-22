#!/usr/bin/env python3
"""Qwen-based coordinator example.

Faithful to the paper: takes the last hidden state of a small Qwen model
through a linear head to produce routing logits. Defaults to Qwen2-0.5B-Instruct
(a close stand-in for the unavailable Qwen-0.6B from the paper).

Note: this only runs the FORWARD pass. To TRAIN the linear head with
sep-CMA-ES, you'd feed hidden states instead of engineered features — left
as a future extension (see notes/paper_summary.md).
"""

from __future__ import annotations
import os
import sys
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.coordinator import QwenCoordinator
from src.llm_pool import LLMPool
from src.tasks import make_dataset
from src.trinity_system import TrinitySystem


def main():
    pool = LLMPool()
    coord = QwenCoordinator(model_name="Qwen/Qwen2-0.5B-Instruct")
    print("Loading Qwen backbone (one-time cost)...")
    coord.load()
    print(f"Backbone + head: {coord.num_parameters()} trainable params (head only)")

    rng = random.Random(0)
    tasks = make_dataset(4, seed=0, difficulty_range=(1, 3))
    sys_ = TrinitySystem(coord, pool, max_turns=4)
    n_correct = 0
    for t in tasks:
        r = sys_.solve(t)
        ok = r.correct
        n_correct += int(ok)
        print(f"  {t.id:30s}  accepted={r.accepted}  correct={ok}  final={r.final_answer}  "
              f"turns={r.turns}  decisions={r.decisions}")
    print(f"\n{n_correct}/{len(tasks)} correct with Qwen-coord (untrained head — random routing)")


if __name__ == "__main__":
    main()
