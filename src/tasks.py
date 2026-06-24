"""Toy tasks with verifiable answers for TRINITY training/eval.

Each task is a dict:
    id:           unique str
    prompt:       the question
    answer:       ground-truth answer (str or numeric)
    kind:         "arithmetic" | "logic" | "string" | "word"
    difficulty:   1-5

Verifier compares extracted final answer to ground truth.
"""

from __future__ import annotations
import random
import re
from dataclasses import dataclass


@dataclass
class Task:
    id: str
    prompt: str
    answer: str
    kind: str
    difficulty: int = 1


# --------------------------- generators ---------------------------

def _arithmetic(rng: random.Random, difficulty: int) -> Task:
    if difficulty == 1:
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        op = rng.choice(["+", "-", "*"])
    elif difficulty == 2:
        a, b = rng.randint(10, 50), rng.randint(10, 50)
        op = rng.choice(["+", "-", "*", "//"])
    elif difficulty == 3:
        a, b = rng.randint(50, 200), rng.randint(2, 12)
        op = rng.choice(["+", "-", "*", "//"])
    elif difficulty == 4:
        a, b = rng.randint(100, 999), rng.randint(3, 25)
        op = rng.choice(["*", "//"])
    else:
        a, b = rng.randint(100, 9999), rng.randint(7, 99)
        op = "*"

    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    elif op == "*":
        ans = a * b
    elif op == "//":
        # ensure divisibility
        ans = a // b
        a = ans * b
    prompt = f"What is {a} {op} {b}?"
    return Task(
        id=f"arith-{a}-{op}-{b}",
        prompt=prompt,
        answer=str(ans),
        kind="arithmetic",
        difficulty=difficulty,
    )


def _two_step_arithmetic(rng: random.Random, difficulty: int) -> Task:
    """Two-step: e.g. (a + b) * c - d. Forces decomposition."""
    a = rng.randint(5, 30)
    b = rng.randint(5, 30)
    c = rng.randint(2, 9)
    d = rng.randint(1, 20)
    exprs = [
        (f"({a} + {b}) * {c} - {d}", (a + b) * c - d),
        (f"({a} - {b}) * {c} + {d}", (a - b) * c + d),
        (f"{a} * {b} + {c} * {d}", a * b + c * d),
        (f"{a} * {b} - {c} * {d}", a * b - c * d),
        (f"({a} + {b} + {c}) * {d}", (a + b + c) * d),
    ]
    expr, ans = rng.choice(exprs)
    prompt = f"Compute: {expr}"
    return Task(
        id=f"two-step-{a}-{b}-{c}-{d}",
        prompt=prompt,
        answer=str(ans),
        kind="arithmetic",
        difficulty=min(5, difficulty + 1),
    )


def _logic(rng: random.Random, difficulty: int) -> Task:
    """Simple propositional logic puzzles."""
    a = rng.choice([True, False])
    b = rng.choice([True, False])
    c = rng.choice([True, False])
    if difficulty == 1:
        ans = a and b
        prompt = f"Is the following statement true? ({a} AND {b})"
    elif difficulty == 2:
        ans = a or b
        prompt = f"Is the following statement true? ({a} OR {b})"
    elif difficulty == 3:
        ans = (a and b) or c
        prompt = f"Is the following statement true? (({a} AND {b}) OR {c})"
    else:
        ans = (a or b) and (not c)
        prompt = f"Is the following statement true? (({a} OR {b}) AND (NOT {c}))"
    return Task(
        id=f"logic-{a}-{b}-{c}",
        prompt=prompt,
        answer=str(ans),
        kind="logic",
        difficulty=difficulty,
    )


def _string(rng: random.Random, difficulty: int) -> Task:
    word = rng.choice(
        ["trinity", "sakana", "router", "evolut", "iclr", "agent",
         "thinker", "worker", "verify", "matrix", "cmaes"]
    )
    if difficulty <= 2:
        ans = len(word)
        prompt = f"How many letters are in the word '{word}'?"
    elif difficulty == 3:
        ans = word[::-1]
        prompt = f"Reverse the word '{word}'."
    else:
        # count vowels
        ans = sum(1 for c in word if c in "aeiou")
        prompt = f"How many vowels are in the word '{word}'?"
    return Task(
        id=f"str-{word}-{difficulty}",
        prompt=prompt,
        answer=str(ans),
        kind="string",
        difficulty=difficulty,
    )


# --------------------------- public API ---------------------------

def make_task(rng: random.Random | None = None, difficulty: int | None = None) -> Task:
    rng = rng or random.Random()
    if difficulty is None:
        difficulty = rng.randint(1, 5)
    kind = rng.choice(["arith", "two_step", "logic", "string"])
    if kind == "arith":
        return _arithmetic(rng, difficulty)
    if kind == "two_step":
        return _two_step_arithmetic(rng, difficulty)
    if kind == "logic":
        return _logic(rng, difficulty)
    return _string(rng, difficulty)


def make_dataset(n: int, seed: int = 0, difficulty_range=(1, 5)) -> list[Task]:
    rng = random.Random(seed)
    out: list[Task] = []
    for i in range(n):
        d = rng.randint(*difficulty_range)
        out.append(make_task(rng, difficulty=d))
    return out


# --------------------------- answer matching ---------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_answer(text: str) -> str | None:
    """Pull the worker's committed answer from its output.

    Priority:
        1. text after 'FINAL:' (TRINITY role convention)
        2. \\boxed{...} (MATH benchmark convention)
        3. last line matching A / B / C / D (MMLU convention)
        4. last number in the text (numeric tasks)
        5. boolean (logic tasks)
        6. last non-empty line (code tasks)
    """
    if not text:
        return None
    # 1. TRINITY FINAL: convention
    m = re.search(r"final\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    # 2. \boxed{...}  (MATH)
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m.group(1).strip()
    # 3. MMLU multiple choice (A/B/C/D on its own line)
    choice_match = re.search(r"^\s*([ABCD])\s*$", text, re.MULTILINE)
    if choice_match:
        return choice_match.group(1).strip()
    # also catch "The answer is (B)" style
    m = re.search(r"\b(?:answer|choice)\s*[:=]\s*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 4. last number (numeric tasks)
    nums = _NUM_RE.findall(text)
    if nums:
        return nums[-1]
    # 5. boolean (logic tasks)
    low = text.lower()
    if "true" in low and "false" not in low:
        return "True"
    if "false" in low and "true" not in low:
        return "False"
    # 6. last non-empty line (code tasks — return last line of code)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        return lines[-1]
    return None


def is_correct(task: Task, response: str) -> bool:
    """Compare extracted answer with ground truth (lenient)."""
    extracted = extract_final_answer(response)
    if extracted is None:
        return False
    extracted_clean = extracted.strip().strip(".")
    target = task.answer.strip().strip(".")
    # numeric compare
    try:
        return abs(float(extracted_clean) - float(target)) < 1e-6
    except ValueError:
        pass
    return extracted_clean.lower() == target.lower()
