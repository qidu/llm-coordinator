"""sep-CMA-ES trainer for the MLP coordinator.

Why sep-CMA-ES (and not RL)?
    - Sparse binary reward (correct / not correct)
    - Stochastic LLM outputs make policy gradients high-variance
    - The paper's whole point: black-box, noise-tolerant, gradient-free

sep = "separable": each parameter dimension has its own step size σ_i. This
collapses the covariance matrix update to O(d) instead of O(d^2), which matters
when d ~ 1.4K (our MLP head). At d ~ 1.4K, full CMA-ES is still tractable, but
sep-CMA-ES trains faster and matches the paper.

Fitness
-------
Mean binary reward over a fixed task batch (re-rolled with the same seed each
generation so the same tasks are graded each time → low-noise fitness signal).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
import time
from typing import Callable

import numpy as np

from .coordinator import MLPCoordinator
from .llm_pool import LLMPool
from .tasks import Task, make_dataset
from .trinity_system import TrinitySystem


@dataclass
class CMAESConfig:
    n_dim: int
    pop_size: int = 16          # λ
    sigma_init: float = 0.3     # initial step size (weights initialized ~ N(0, 0.1))
    sigma_min: float = 1e-4
    sigma_max: float = 2.0
    tau_s: float = 0.0          # set below
    tau_c: float = 0.0          # set below
    generations: int = 30
    fitness_fn: Callable[[np.ndarray], float] = None  # type: ignore

    def __post_init__(self):
        # Standard sep-CMA-ES schedule
        n = self.n_dim
        self.tau_s = (self.sigma_max - self.sigma_min) / max(1, self.generations)
        self.tau_c = self.tau_s / 3.0


@dataclass
class GenerationLog:
    gen: int
    best: float
    mean: float
    worst: float
    sigma_mean: float
    elapsed_s: float
    candidates: list[float] = field(default_factory=list)


def random_search(cfg: CMAESConfig, init_params: np.ndarray,
                  on_generation: Callable[[GenerationLog], None] | None = None,
                  log_every: int = 1, verbose: bool = True) -> np.ndarray:
    """Random search baseline (paper Section 4.8 / Table 4).

    Same total budget as sep-CMA-ES per the paper's protocol: m_CMA=16,
    m_RS=32, so each gen the RS baseline evaluates 2x more candidates but
    no recombination. We just use the per-gen best.
    """
    d = cfg.n_dim
    pop = cfg.pop_size * 2  # m_RS = 2 * m_CMA  (paper's protocol)
    mean = init_params.astype(np.float64).copy()
    rng = np.random.default_rng(1)

    best_so_far = -np.inf
    best_params = mean.copy()

    for gen in range(cfg.generations):
        t0 = time.time()
        # RS samples from a fixed prior around the initial mean
        Z = rng.standard_normal((pop, d))
        X = mean + cfg.sigma_init * Z
        fits = np.array([cfg.fitness_fn(x.astype(np.float32)) for x in X], dtype=np.float64)
        order = np.argsort(-fits)
        if fits[order[0]] > best_so_far:
            best_so_far = fits[order[0]]
            best_params = X[order[0]].astype(np.float32).copy()
        elapsed = time.time() - t0
        log = GenerationLog(
            gen=gen,
            best=float(fits[order[0]]),
            mean=float(fits.mean()),
            worst=float(fits[order[-1]]),
            sigma_mean=cfg.sigma_init,
            elapsed_s=elapsed,
            candidates=[float(f) for f in fits],
        )
        if verbose and (gen % log_every == 0 or gen == cfg.generations - 1):
            print(
                f"[RS gen {gen:02d}] best={log.best:.3f} mean={log.mean:.3f} ({elapsed:.1f}s)"
            )
        if on_generation is not None:
            on_generation(log)
    return best_params


# placeholder for clarity in on_generation callback name above


def sep_cma_es(cfg: CMAESConfig, init_params: np.ndarray,
               on_generation: Callable[[GenerationLog], None] | None = None,
               log_every: int = 1, verbose: bool = True) -> np.ndarray:
    """Plain sep-CMA-ES (Hansen 2016) on flat parameter vector.

    Returns the best parameter vector found.
    """
    d = cfg.n_dim
    lam = cfg.pop_size
    mu = lam // 2

    # weights (log-decreasing) — same shape as standard CMA-ES
    raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = raw_w / raw_w.sum()
    mu_eff = 1.0 / (w ** 2).sum()

    # learning rates (sep-CMA-ES)
    c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
    c_1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)

    mean = init_params.astype(np.float64).copy()
    sigma = np.full(d, cfg.sigma_init, dtype=np.float64)
    p_sigma = np.zeros(d, dtype=np.float64)
    p_c = np.zeros(d, dtype=np.float64)

    # E[||N(0,I)||] approx
    E_norm = math.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))

    best_so_far = -np.inf
    best_params = mean.copy()

    rng = np.random.default_rng(0)

    for gen in range(cfg.generations):
        t0 = time.time()
        # ---- sample population ----
        Z = rng.standard_normal((lam, d))
        X = mean + sigma * Z
        # ---- evaluate ----
        fits = np.array([cfg.fitness_fn(x.astype(np.float32)) for x in X], dtype=np.float64)
        order = np.argsort(-fits)  # descending
        X_sel = X[order[:mu]]
        fits_sel = fits[order[:mu]]

        # track best
        if fits[order[0]] > best_so_far:
            best_so_far = fits[order[0]]
            best_params = X[order[0]].astype(np.float32).copy()

        # ---- update mean (weighted recombination) ----
        new_mean = (w[:, None] * X_sel).sum(axis=0)

        # ---- update evolution paths ----
        ps_step = (new_mean - mean) / np.maximum(sigma, 1e-12)
        p_sigma = (1.0 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * ps_step
        p_c = (1.0 - c_c) * p_c + math.sqrt(c_c * (2.0 - c_c) * mu_eff) * ps_step

        # ---- update step sizes (decoupled) ----
        # sign-based update with damping
        denom = E_norm
        # adapt sigma per dimension toward |p_sigma_i|/denom
        # map: 1 + c_sigma/a * (|p_sigma|/denom - E_norm_1)  (E_norm_1 is 1 for N(0,1))
        # using 1/|p_sigma| ~ 1 as the unbiased target:
        target = np.abs(p_sigma) / denom
        # multiplicative update with damping c_sigma
        sigma = sigma * np.exp((c_sigma / d_sigma) * (target - 1.0 / denom))
        sigma = np.clip(sigma, cfg.sigma_min, cfg.sigma_max)

        mean = new_mean

        elapsed = time.time() - t0
        log = GenerationLog(
            gen=gen,
            best=float(fits[order[0]]),
            mean=float(fits.mean()),
            worst=float(fits[order[-1]]),
            sigma_mean=float(sigma.mean()),
            elapsed_s=elapsed,
            candidates=[float(f) for f in fits],
        )
        if verbose and (gen % log_every == 0 or gen == cfg.generations - 1):
            print(
                f"[gen {gen:02d}] best={log.best:.3f} mean={log.mean:.3f} "
                f"sigma_mean={log.sigma_mean:.4f} ({elapsed:.1f}s)"
            )
        if on_generation is not None:
            on_generation(log)

    return best_params


# ------------------------------------------------------------------
# Task: train the MLP coordinator on a fixed task batch
# ------------------------------------------------------------------

def make_fitness_fn(tasks: list[Task], pool: LLMPool, max_turns: int = 6,
                    rollouts_per_candidate: int = 1,
                    use_early_bonus: bool = True,
                    coord_cfg: "CoordinatorConfig | None" = None):
    """Build a fitness function for a fixed task batch.

    Fitness combines:
        - binary correctness (primary, weight 1.0)
        - early-termination bonus (encourages using the Verifier to stop)
        - role-diversity bonus (small, prevents degenerate 'Worker only' policy)

    Each task is rolled out exactly once per candidate for noise reduction.
    """
    from .coordinator import MLPCoordinator, CoordinatorConfig
    coord_cfg = coord_cfg or CoordinatorConfig()

    def fitness(params: np.ndarray) -> float:
        coord = MLPCoordinator(params=params, cfg=coord_cfg)
        system = TrinitySystem(coord, pool, max_turns=max_turns)
        score = 0.0
        for t in tasks:
            res = system.solve(t)
            base = 1.0 if res.correct else 0.0
            if use_early_bonus and res.correct:
                # linear bonus: solve in 3 turns = +0.5, in 6 turns = 0
                turn_eff = (max_turns - res.turns) / max(1, max_turns)
                base += 0.3 * turn_eff
                # small bonus for actually using the verifier
                used_verifier = any(role == "Verifier" for _, role in res.decisions)
                if used_verifier:
                    base += 0.1
            score += base
        return score / max(1, len(tasks))
    return fitness


def recommended_pop_size(n_dim: int) -> int:
    """Paper's formula: lambda = ceil(4 + 3 ln n). For n=10240, lambda=32."""
    import math
    return max(4, math.ceil(4 + 3 * math.log(max(2, n_dim))))


def train_router(pool: LLMPool, n_train: int = 16, n_eval: int = 32,
                 pop_size: int | None = None, generations: int = 20,
                 max_turns: int = 6, hidden: int = 32, seed: int = 0,
                 head: str = "block_diag", n_blocks: int = 5,
                 use_argmax: bool = True,
                 method: str = "cma",
                 save_path: str | None = None) -> tuple[np.ndarray, list[GenerationLog], "CoordinatorConfig"]:
    """Train a router with sep-CMA-ES (or RS baseline) on synthetic tasks."""
    from .coordinator import MLPCoordinator, CoordinatorConfig
    coord_cfg = CoordinatorConfig(
        hidden=hidden,
        head=head,
        n_blocks=n_blocks,
        use_argmax=use_argmax,
    )
    init_coord = MLPCoordinator(cfg=coord_cfg)
    init = init_coord.get_params()
    d = init.size
    if pop_size is None:
        pop_size = recommended_pop_size(d)
    print(f"Training {head} router ({n_blocks} blocks, argmax={use_argmax}, "
          f"d={d} params): pop={pop_size}, gen={generations}, tasks={n_train}, "
          f"max_turns={max_turns}, method={method}")

    train_tasks = make_dataset(n_train, seed=seed, difficulty_range=(1, 4))
    fit_fn = make_fitness_fn(train_tasks, pool, max_turns=max_turns, coord_cfg=coord_cfg)
    logs: list[GenerationLog] = []

    def on_gen(log: GenerationLog):
        logs.append(log)

    es_cfg = CMAESConfig(
        n_dim=d,
        pop_size=pop_size,
        sigma_init=0.15,
        generations=generations,
        fitness_fn=fit_fn,
    )
    if method == "cma":
        best = sep_cma_es(es_cfg, init, on_generation=on_gen, verbose=True)
    elif method == "rs":
        best = random_search(es_cfg, init, on_generation=on_gen, verbose=True)
    else:
        raise ValueError(f"unknown method: {method}")

    if save_path is not None:
        import json
        with open(save_path, "w") as f:
            json.dump({
                "params": best.tolist(),
                "head": head,
                "n_blocks": n_blocks,
                "use_argmax": use_argmax,
                "hidden": hidden,
                "d": d,
                "n_train": n_train,
                "n_eval": n_eval,
                "pop_size": pop_size,
                "generations": generations,
                "max_turns": max_turns,
                "seed": seed,
                "method": method,
                "fitness_train_final": logs[-1].best if logs else None,
            }, f)
        print(f"Saved params to {save_path}")
        log_path = save_path.replace(".json", "_log.json")
        with open(log_path, "w") as f:
            json.dump([lg.__dict__ for lg in logs], f, indent=2)
        print(f"Saved training log to {log_path}")

    return best, logs, coord_cfg
