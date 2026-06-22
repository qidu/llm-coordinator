"""End-to-end TRINITY multi-turn solver.

This is the actual loop from Section 4 of the paper, but pluggable:

    coordinator  : HeuristicCoordinator | MLPCoordinator | QwenCoordinator
    pool         : LLMPool  (mock or real)
    task         : Task     (for answer checking)

Each turn:
    1. coordinator.route(turn, transcript, task) -> (model_key, role)
    2. pool.generate(model_key, system_prompt, context, role, task) -> output
    3. transcript.append(...)
    4. if role == Verifier and "JUDGMENT: ACCEPT" in output -> return final

Run via .solve(task) -> dict with transcript, turns_used, accepted, correct.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .coordinator import HeuristicCoordinator, MLPCoordinator
from .features import IDX_TO_ROLE
from .llm_pool import LLMPool
from .prompts import THINKER_PROMPT, WORKER_PROMPT, VERIFIER_PROMPT
from .tasks import Task, extract_final_answer, is_correct

ROLE_PROMPTS = {
    "Thinker": THINKER_PROMPT,
    "Worker": WORKER_PROMPT,
    "Verifier": VERIFIER_PROMPT,
}


@dataclass
class RunResult:
    task: Task
    transcript: list[dict] = field(default_factory=list)
    turns: int = 0
    accepted: bool = False
    correct: bool = False
    final_answer: str | None = None
    decisions: list[tuple[str, str]] = field(default_factory=list)  # (model, role) per turn


class TrinitySystem:
    def __init__(self, coordinator, pool: LLMPool, max_turns: int = 6):
        self.coordinator = coordinator
        self.pool = pool
        self.max_turns = max_turns

    def solve(self, task: Task) -> RunResult:
        self.coordinator.reset()
        transcript: list[dict] = [
            {"role": "user", "content": f"Original Question: {task.prompt}"}
        ]
        decisions: list[tuple[str, str]] = []
        accepted = False
        final_answer = None

        for turn in range(1, self.max_turns + 1):
            model_key, role = self.coordinator.route(turn, transcript, task=task,
                                                     max_turns=self.max_turns)
            decisions.append((model_key, role))
            system = ROLE_PROMPTS[role]
            context = "\n".join(f"{m['role']}: {m['content']}" for m in transcript)
            output = self.pool.generate(model_key, system, context, role, task=task)

            tag = f"[{role} {model_key}]:"
            transcript.append({"role": "assistant", "content": f"{tag} {output}"})

            # final answer capture: prefer last Worker's FINAL
            if role == "Worker":
                final_answer = extract_final_answer(output)
            if role == "Verifier" and "JUDGMENT: ACCEPT" in output:
                accepted = True
                break

        correct = is_correct(task, final_answer or "")
        return RunResult(
            task=task,
            transcript=transcript,
            turns=len(decisions),
            accepted=accepted,
            correct=correct,
            final_answer=final_answer,
            decisions=decisions,
        )
