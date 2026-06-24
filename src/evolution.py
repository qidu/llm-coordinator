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
    batched_fitness_fn: Callable[[np.ndarray], np.ndarray] = None  # type: ignore
    # If set, takes (pop, d) and returns (pop,) fitness. Takes precedence
    # over per-candidate fitness_fn. Used by the Qwen batched path.

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
        if cfg.batched_fitness_fn is not None:
            fits = np.asarray(cfg.batched_fitness_fn(X.astype(np.float32)), dtype=np.float64)
        else:
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
        if cfg.batched_fitness_fn is not None:
            fits = np.asarray(cfg.batched_fitness_fn(X.astype(np.float32)), dtype=np.float64)
        else:
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
                    coord_cfg: "CoordinatorConfig | None" = None,
                    coord_factory: "Callable[[np.ndarray], object] | None" = None):
    """Build a fitness function for a fixed task batch.

    Fitness combines:
        - binary correctness (primary, weight 1.0)
        - early-termination bonus (encourages using the Verifier to stop)
        - role-diversity bonus (small, prevents degenerate 'Worker only' policy)

    Each task is rolled out exactly once per candidate for noise reduction.

    coord_factory(params) -> Coordinator is the optional explicit hook
    (used by the Qwen router so we don't re-instantiate the heavy backbone
    on every CMA-ES candidate). If omitted, defaults to MLPCoordinator.
    """
    from .coordinator import MLPCoordinator, CoordinatorConfig
    coord_cfg = coord_cfg or CoordinatorConfig()

    def make_coord(params: np.ndarray):
        if coord_factory is not None:
            return coord_factory(params)
        return MLPCoordinator(params=params, cfg=coord_cfg)

    def fitness(params: np.ndarray) -> float:
        coord = make_coord(params)
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


# ------------------------------------------------------------------
# Batched fitness for QwenCoordinator
# ------------------------------------------------------------------
#
# The naive fitness() loops pop × task × turn Qwen3 forwards. The bulk
# of wall-clock is the Qwen3 backbone, not the head. We can amortize:
# for each (task, turn), the transcript only depends on the head params
# via the action chosen at the PREVIOUS turn. So if we share transcripts
# across candidates that picked the same previous action, we get
# feature reuse.
#
# In practice, the simplest big win is **batching the Qwen3 forward over
# all pop candidates at each (task, turn)**. Transcripts differ across
# candidates (they took different actions), but they're usually within a
# factor of 2 in length, and PyTorch's batched attention is far more
# efficient than N separate forwards.
#
# This function evaluates a whole CMA-ES generation in one call.

def make_batched_qwen_fitness_fn(
    tasks: list,
    pool,
    max_turns: int = 6,
    use_early_bonus: bool = True,
    coord_template = None,  # QwenCoordinator instance (params will be overwritten)
    batch_size: int = 0,    # chunk size for Qwen3 forwards (0 = no chunking)
):
    """Build a fitness function that evaluates a BATCH of head params at once.

    coord_template: a QwenCoordinator whose backbone is loaded. We only
    mutate its head parameters; backbone stays frozen. The caller must
    pass a numpy array of shape (pop, d_params).

    batch_size: if > 0, splits Qwen3 forward passes into chunks of this
    many contexts to avoid GPU OOM. Tune downward (e.g. 4-16) for
    smaller GPUs.
    """
    from .prompts import THINKER_PROMPT, WORKER_PROMPT, VERIFIER_PROMPT
    from .tasks import extract_final_answer, is_correct
    ROLE_PROMPTS = {"Thinker": THINKER_PROMPT, "Worker": WORKER_PROMPT,
                    "Verifier": VERIFIER_PROMPT}

    router = coord_template.router
    # Match the deterministic behaviour of the naive path: use_argmax
    # wins, else fall back to deterministic argmax, else sample.
    deterministic = bool(getattr(coord_template.cfg, "use_argmax", False)
                         or getattr(coord_template.cfg, "deterministic", True))
    models = list(coord_template._models)
    roles = list(coord_template._roles)

    def _format_context(task, transcript, turn, max_turns):
        """Mirror QwenCoordinator.route() so the cached features match."""
        lines = []
        if task is not None:
            lines.append(f"Original Question: {task.prompt}")
            lines.append("")
        for m in transcript:
            lines.append(f"{m['role']}: {m['content']}")
        lines.append(f"\n[turn {turn}/{max_turns}] Choose next (model, role):")
        return "\n".join(lines)

    def _action_to_pair(action: int):
        n_roles = len(roles)
        return action // n_roles, roles[action % n_roles]

    def batched_fitness(params_batch: np.ndarray) -> np.ndarray:
        """Return a (pop,) array of fitness scores.

        Simulates pop parallel rollouts of all tasks. For each (task, turn):
          1. Build transcript-text for every (candidate, task) pair
          2. features_batch() -> (pop*len(tasks), d_hidden)
          3. For each candidate, apply its head params to its slice
          4. Pick actions, run LLM pool, update transcripts
        """
        # sep_cma_es sometimes calls fitness with a single 1-D vector; we
        # always work in 2-D (pop, d).
        if params_batch.ndim == 1:
            params_batch = params_batch[None, :]
        pop = params_batch.shape[0]
        nt = len(tasks)
        # Per-candidate state: a list (over tasks) of transcripts.
        transcripts = [[
            [{"role": "user", "content": f"Original Question: {t.prompt}"}]
            for t in tasks
        ] for _ in range(pop)]
        # decisions[ci][ti] = list of (model_key, role) pairs for that rollout
        decisions = [[[] for _ in range(nt)] for _ in range(pop)]
        # done[ci][ti] = True once this rollout is finished (ACCEPT or max_turns)
        done = [[False] * nt for _ in range(pop)]

        for turn in range(1, max_turns + 1):
            # Active (candidate, task) pairs
            active_pairs = [(ci, ti) for ci in range(pop)
                            for ti in range(nt) if not done[ci][ti]]
            if not active_pairs:
                break
            # 1. Build context strings for all active pairs
            ctxs = [_format_context(tasks[ti], transcripts[ci][ti], turn, max_turns)
                    for ci, ti in active_pairs]
            # 2. Batched Qwen3 forward
            features = router.features_batch(ctxs, batch_size=batch_size)  # (active_pairs, H)
            # 3. For each candidate, gather its slices of features (one per task)
            #    then run head forward once per candidate.
            offset = 0
            # group features by candidate
            per_cand_indices: dict[int, list[tuple[int, int]]] = {}
            for k, (ci, ti) in enumerate(active_pairs):
                per_cand_indices.setdefault(ci, []).append((k, ti))
            for ci, items in per_cand_indices.items():
                # set head params for this candidate
                router.set_params(params_batch[ci])
                idxs = [k for k, _ in items]
                feat_ci = features[idxs]                            # (n_active_for_cand, H)
                logits = router.forward_batched_features(feat_ci)  # (n_active_for_cand, n_actions)
                actions = router.head_select(logits, deterministic=deterministic)
                # 4. For each (candidate, task) pair, take the action
                for (k, ti), action in zip(items, actions):
                    t = tasks[ti]
                    m_idx, role = _action_to_pair(action)
                    model_key = models[m_idx]
                    system = ROLE_PROMPTS[role]
                    context = "\n".join(
                        f"{m['role']}: {m['content']}" for m in transcripts[ci][ti]
                    )
                    output = pool.generate(model_key, system, context, role, task=t)
                    tag = f"[{role} {model_key}]:"
                    transcripts[ci][ti].append(
                        {"role": "assistant", "content": f"{tag} {output}"}
                    )
                    decisions[ci][ti].append((model_key, role))
                    if role == "Verifier" and "JUDGMENT: ACCEPT" in output:
                        done[ci][ti] = True
            # candidates that ran out of turns -> mark done
            if turn == max_turns:
                for ci, ti in active_pairs:
                    done[ci][ti] = True

        # Score each candidate: mean per-task fitness
        per_cand_task_scores = np.zeros((pop, nt), dtype=np.float64)
        for ci in range(pop):
            for ti, t in enumerate(tasks):
                # find last Worker's final answer in this transcript
                last_final = None
                for m in transcripts[ci][ti]:
                    if m["role"] == "assistant" and "Worker" in m["content"]:
                        last_final = extract_final_answer(m["content"])
                correct = is_correct(t, last_final or "")
                base = 1.0 if correct else 0.0
                if use_early_bonus and correct:
                    turn_eff = (max_turns - len(decisions[ci][ti])) / max(1, max_turns)
                    base += 0.3 * turn_eff
                    used_verifier = any(r == "Verifier" for _, r in decisions[ci][ti])
                    if used_verifier:
                        base += 0.1
                per_cand_task_scores[ci, ti] = base
        return per_cand_task_scores.mean(axis=1)

    return batched_fitness


def make_batched_qwen_es_cfg(
    tasks: list,
    pool,
    coord_template,  # QwenCoordinator (backbone loaded, head dummy)
    pop_size: int = 31,
    generations: int = 15,
    max_turns: int = 4,
    sigma_init: float = 0.15,
):
    """Build a CMAESConfig whose fitness_fn evaluates the WHOLE generation
    in one batched call.

    Returns (cfg, init_params) where init_params is the head's param vector
    (used as the CMA-ES mean).
    """
    init_params = coord_template.get_params()
    d = init_params.size
    fit_fn = make_batched_qwen_fitness_fn(
        tasks, pool, max_turns=max_turns, coord_template=coord_template,
    )
    cfg = CMAESConfig(
        n_dim=d,
        pop_size=pop_size,
        sigma_init=sigma_init,
        generations=generations,
        batched_fitness_fn=fit_fn,
    )
    return cfg, init_params
