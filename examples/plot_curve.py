#!/usr/bin/env python3
"""Plot the CMA-ES learning curve (matplotlib, no savefig needed if no display)."""

from __future__ import annotations
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def plot_from_log(log_path: str = "artifacts/router_params_log.json",
                  out_path: str = "artifacts/learning_curve.png"):
    if not os.path.exists(log_path):
        print(f"no log at {log_path}")
        return
    with open(log_path) as f:
        log = json.load(f)
    gens = [g["gen"] for g in log]
    best = [g["best"] for g in log]
    mean = [g["mean"] for g in log]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; printing text summary")
        for g, b, m in zip(gens, best, mean):
            print(f"  gen {g:02d}  best={b:.3f}  mean={m:.3f}")
        return

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(gens, best, "o-", label="best")
    ax[0].plot(gens, mean, "s--", label="mean", alpha=0.6)
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("fitness")
    ax[0].set_title("sep-CMA-ES on MLP router")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    sigmas = [g["sigma_mean"] for g in log]
    ax[1].plot(gens, sigmas, "o-", color="tab:orange")
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("mean sigma (step size)")
    ax[1].set_title("step size adaptation")
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    plot_from_log()
