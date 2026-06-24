"""LLM pool — mock and real.

Interface:
    class LLLM:
        def generate(self, system: str, user: str, role: str, task: Task | None) -> str

    class LLMPool:
        def generate(self, model_key: str, system: str, user: str, role: str,
                    task: Task | None) -> str

Mock strategy
-------------
Model A = "small"  -> fast, less reliable. 80% chance of correct answer
Model_B = "strong" -> slower, more reliable. 95% chance of correct answer

Real strategy (make_real_pool)
-------------------------------
deepseek-v4-flash  -> fast, used as "small" (Thinker/Worker routing)
max-m3              -> slow/strong, used as "strong" (Verifier + hard tasks)
Both call localhost:8788 via OpenAI-compatible API.
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


class OpenAICompatibleLLM:
    """Real LLM via an OpenAI-compatible API endpoint.

    model_name: model identifier sent to the API (e.g. "deepseek-v4-flash", "max-m3")
    endpoint:  base URL, e.g. "http://localhost:8788/v1"
    temperature, max_tokens: generation params.
    timeout:  seconds to wait for a response.
    """

    def __init__(self, model_name: str, endpoint: str = "http://localhost:8788/v1",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 timeout: float = 60.0):
        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, system: str, user: str, role: str, task: Task | None = None) -> str:
        import json, urllib.request, urllib.error

        # Build the chat message list, appending the user text as the final message.
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        body = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            return f"[LLM ERROR: {exc}]"


def make_real_pool(endpoint: str = "http://localhost:8788/v1",
                   small_model: str = "deepseek-v4-flash",
                   strong_model: str = "max-m3",
                   small_temp: float = 0.7,
                   strong_temp: float = 0.3,
                   timeout: float = 60.0) -> LLMPool:
    """Factory: build an LLMPool with real OpenAI-compatible LLMs.

    Args:
        endpoint:      base URL of the inference server
        small_model:   model name for the "small" slot (Thinker/Worker)
        strong_model:  model name for the "strong" slot (Verifier, hard tasks)
        small_temp:    temperature for the small model
        strong_temp:   temperature for the strong model (lower = more deterministic)
        timeout:       seconds per API call
    """
    return LLMPool(
        llms={
            "Model_A": OpenAICompatibleLLM(
                model_name=small_model,
                endpoint=endpoint,
                temperature=small_temp,
                timeout=timeout,
            ),
            "Model_B": OpenAICompatibleLLM(
                model_name=strong_model,
                endpoint=endpoint,
                temperature=strong_temp,
                timeout=timeout,
            ),
        }
    )


class LLMPool:
    """Routes (model_key, role) -> a real/mock LLM.generate call."""

    def __init__(self, small: LLM | None = None, strong: LLM | None = None,
                 llms: dict[str, LLM] | None = None):
        if llms is not None:
            self.llms = llms
        else:
            self.llms = {
                "Model_A": small or MockSmallLLM(),
                "Model_B": strong or MockStrongLLM(),
            }

    def generate(self, model_key: str, system: str, user: str, role: str, task: Task | None = None) -> str:
        return self.llms[model_key].generate(system, user, role, task=task)

    @property
    def keys(self) -> list[str]:
        return list(self.llms.keys())
