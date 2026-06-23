"""Evaluation utilities: compare heuristic vs trained MLP router."""

from __future__ import annotations
from dataclasses import dataclass

from .coordinator import HeuristicCoordinator, MLPCoordinator
from .llm_pool import LLMPool
from .tasks import Task, make_dataset
from .trinity_system import RunResult, TrinitySystem


@dataclass
class EvalReport:
    name: str
    n_tasks: int
    accuracy: float
    accept_rate: float
    avg_turns: float
    decisions: dict[str, int]


def evaluate(coordinator_factory, name: str, tasks: list[Task], pool: LLMPool,
             max_turns: int = 6) -> EvalReport:
    coord = coordinator_factory()
    system = TrinitySystem(coord, pool, max_turns=max_turns)
    n_correct = 0
    n_accept = 0
    total_turns = 0
    decisions: dict[str, int] = {}
    for t in tasks:
        res = system.solve(t)
        if res.correct:
            n_correct += 1
        if res.accepted:
            n_accept += 1
        total_turns += res.turns
        for model, role in res.decisions:
            key = f"{model}->{role}"
            decisions[key] = decisions.get(key, 0) + 1
    return EvalReport(
        name=name,
        n_tasks=len(tasks),
        accuracy=n_correct / max(1, len(tasks)),
        accept_rate=n_accept / max(1, len(tasks)),
        avg_turns=total_turns / max(1, len(tasks)),
        decisions=decisions,
    )


def compare(coord_specs: list[tuple[str, callable]], name: str, tasks: list[Task],
            pool: LLMPool, max_turns: int = 6) -> list[EvalReport]:
    reports = []
    for nm, factory in coord_specs:
        r = evaluate(factory, nm, tasks, pool, max_turns=max_turns)
        reports.append(r)
        print(f"\n[{r.name}] acc={r.accuracy:.2%}  accept={r.accept_rate:.2%}  "
              f"avg_turns={r.avg_turns:.2f}")
        for k, v in sorted(r.decisions.items(), key=lambda kv: -kv[1]):
            print(f"   {k:30s}  {v}")
    return reports
