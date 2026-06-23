"""Mock LLM pool.

In the paper the model pool mixes different LLM providers. For the prototype we
ship two *deterministic* mocks so training has zero noise and tests are
reproducible. Real LLMs can be plugged in later behind the same interface.

Interface:
    class LLMPool:
        def generate(self, model_key: str, system: str, user: str) -> str

Mock strategy
-------------
Model A = "small"  -> fast, less reliable. 80% chance of correct answer
Model B = "strong" -> slower, more reliable. 95% chance of correct answer

Both:
- Always follow the role prompt structure (Thinker bullet, Worker FINAL:, Verifier JUDGMENT:)
- Quality varies by (model_key, difficulty) so the coordinator has something to learn
"""

from __future__ import annotations
import random
import re
from typing import Protocol

from .prompts import THINKER_PROMPT, WORKER_PROMPT, VERIFIER_PROMPT
from .tasks import Task, extract_final_answer, is_correct


class LLM(Protocol):
    def generate(self, system: str, user: str, role: str, task: Task | None = None) -> str: ...


class MockSmallLLM:
    """Fast, ~80% reliable. Sometimes makes arithmetic slips."""

    def __init__(self, seed: int = 0, reliability: float = 0.80):
        self.rng = random.Random(seed)
        self.reliability = reliability

    def generate(self, system: str, user: str, role: str, task: Task | None = None) -> str:
        if role == "Thinker":
            return self._think(user, task)
        if role == "Worker":
            return self._work(user, task)
        if role == "Verifier":
            return self._verify(user, task)
        return ""

    # ---- role handlers ----

    def _think(self, user: str, task: Task | None) -> str:
        # always produces a reasonable plan
        if task is None:
            return "- Decompose the problem\n- Solve each part\n- Verify"
        if task.kind == "arithmetic":
            return "- Identify the operator\n- Compute step by step\n- Report FINAL: <value>"
        if task.kind == "logic":
            return "- Evaluate each boolean\n- Apply the connective in order\n- Report FINAL: True/False"
        if task.kind == "string":
            return "- Identify the operation (count/reverse/vowels)\n- Apply carefully\n- Report FINAL: <value>"
        return "- Decompose\n- Solve\n- Verify"

    def _work(self, user: str, task: Task | None) -> str:
        if task is None or self.rng.random() > self.reliability:
            # noisy wrong answer
            try:
                wrong = str(int(task.answer) + self.rng.choice([-2, -1, 1, 2, 3]))
            except (ValueError, AttributeError):
                wrong = "?"
            return f"Working it out...\nGot {wrong}.\nFINAL: {wrong}"
        return f"Solving: {task.prompt}\nThe answer is {task.answer}.\nFINAL: {task.answer}"

    def _verify(self, user: str, task: Task | None) -> str:
        # small model is optimistic: 70% accept, 30% reject
        accept = self.rng.random() < 0.7
        if accept:
            return f"JUDGMENT: ACCEPT\nDIAGNOSIS: looks good."
        return "JUDGMENT: REVISE\nDIAGNOSIS: recheck the arithmetic."


class MockStrongLLM:
    """Slow, ~98% reliable. Rare slip on difficulty>=4."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed + 999)

    def generate(self, system: str, user: str, role: str, task: Task | None = None) -> str:
        if role == "Thinker":
            return self._think(user, task)
        if role == "Worker":
            return self._work(user, task)
        if role == "Verifier":
            return self._verify(user, task)
        return ""

    def _think(self, user: str, task: Task | None) -> str:
        if task is None:
            return "- Decompose\n- Solve\n- Cross-check"
        if task.kind == "arithmetic":
            return (
                "- Parse the expression exactly as written\n"
                "- Respect operator precedence\n"
                "- State FINAL: <value>"
            )
        if task.kind == "logic":
            return (
                "- Evaluate inner boolean expressions first\n"
                "- Apply the outer connective\n"
                "- State FINAL: True/False"
            )
        if task.kind == "string":
            return (
                "- Identify the operation requested\n"
                "- Apply it character by character if needed\n"
                "- State FINAL: <value>"
            )
        return "- Decompose, solve, verify"

    def _work(self, user: str, task: Task | None) -> str:
        if task is None:
            return "FINAL: ?"
        # very small chance to slip on hard tasks
        slip = self.rng.random() < 0.02 * max(1, task.difficulty - 2)
        if slip:
            try:
                wrong = str(int(task.answer) + self.rng.choice([-1, 1]))
            except (ValueError, AttributeError):
                wrong = "?"
            return f"Working...\nResult: {wrong}\nFINAL: {wrong}"
        return f"Computation:\n{task.answer}\nFINAL: {task.answer}"

    def _verify(self, user: str, task: Task | None) -> str:
        # strong model is strict: parses worker's FINAL and compares
        # The user text contains the full transcript; find the last FINAL: line
        # produced by any worker.
        finals = re.findall(r"FINAL:\s*([^\n]+)", user)
        worker_text = ("\n".join(finals)) if finals else ""
        if task is not None and not is_correct(task, worker_text):
            return "JUDGMENT: REVISE\nDIAGNOSIS: worker's FINAL does not match the problem."
        if not finals and task is not None:
            return "JUDGMENT: REVISE\nDIAGNOSIS: worker did not produce a FINAL line."
        return "JUDGMENT: ACCEPT\nDIAGNOSIS: solution checks out."


class LLMPool:
    """Routes (model_key, role) -> a real/mock LLM.generate call."""

    def __init__(self, small: LLM | None = None, strong: LLM | None = None):
        self.llms = {
            "Model_A": small or MockSmallLLM(),
            "Model_B": strong or MockStrongLLM(),
        }

    def generate(self, model_key: str, system: str, user: str, role: str, task: Task | None = None) -> str:
        return self.llms[model_key].generate(system, user, role, task=task)

    @property
    def keys(self) -> list[str]:
        return list(self.llms.keys())
