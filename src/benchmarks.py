"""Real benchmark task loaders: MATH500, MMLU, LiveCodeBench.

Each function returns a list[Task] that can be passed directly to
TrinitySystem.solve() and the evolution fitness loop.

Answer extraction
-----------------
Each task's answer field is the ground truth.  The LLM's output is parsed
by tasks.extract_final_answer() which handles:
  - FINAL: <value>   (TRINITY role convention)
  - \\boxed{...}     (MATH convention)
  - A / B / C / D    (MMLU convention)
  - bare numbers     (fallback)
"""

from __future__ import annotations
import os
import re
import urllib.request
import urllib.error
import json
from pathlib import Path
from typing import Literal

from .tasks import Task


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _download_json(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Download a JSON file if dest doesn't exist."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-coordinator/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
    except (urllib.error.URLError, Exception) as exc:
        raise RuntimeError(
            f"Failed to download {url}. "
            f"Check your network or download the file manually to {dest}."
        ) from exc


def _strip_boxed(text: str) -> str:
    """Extract content from \\boxed{...} or bare text."""
    text = text.strip()
    # strip markdown code fences
    text = re.sub(r"^\$\$(.*?)\$\$$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^\$(.*?)\$$", r"\1", text)
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m.group(1).strip()
    return text.strip().lstrip("\\")


# ------------------------------------------------------------------
# MATH500
# ------------------------------------------------------------------

def load_math500(cache_dir: str | Path = "data/math500",
                 split: Literal["test", "val"] = "test") -> list[Task]:
    """Load MATH500 (Hendryck et al.) from HuggingFace.

    Dataset: HuggingFaceH4/MATH500
    Each item has: problem, solution, answer, level (difficulty 1-5)

    The answer field is a LaTeX string, e.g. "42" or "2^{10}".
    We store the plain-text answer (stripped of \\boxed).
    Difficulty is derived from the "level" field (1=Easy … 5=Hard).
    """
    cache_dir = Path(cache_dir)
    math_dir = cache_dir / "HuggingFaceH4__MATH500"
    math_dir.mkdir(parents=True, exist_ok=True)
    local_file = math_dir / f"{split}.json"

    url = (
        "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/"
        "resolve/main/test.jsonl"
    )
    if not local_file.exists():
        _download_json(url, local_file)

    tasks: list[Task] = []
    difficulty_map = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    with open(local_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item["problem"]
            # answer may be \boxed{...} or bare
            answer = _strip_boxed(item.get("answer", ""))
            level_str = str(item.get("level", "3"))
            difficulty = difficulty_map.get(level_str, 3)
            tasks.append(Task(
                id=f"math500-{len(tasks)}",
                prompt=prompt,
                answer=answer,
                kind="math",
                difficulty=difficulty,
            ))
    return tasks


# ------------------------------------------------------------------
# MMLU
# ------------------------------------------------------------------

def load_mmlu(cache_dir: str | Path = "data/mmlu",
               subject: str | None = None) -> list[Task]:
    """Load MMLU (Hendryck et al.) from HuggingFace.

    Dataset: alexandrainst/m_mmlu  (13,258 multi-subject test questions)
    Fields:  instruction, option_a/b/c/d, answer (A/B/C/D), id.

    We store the prompt as the instruction + options, and the answer as
    the single-letter choice (A/B/C/D).

    Args:
        cache_dir:  local cache directory
        subject:    if provided, only load items whose id starts with
                    "subject/" (e.g. "math" or "clinical").
                    If None, loads all subjects.
    """
    cache_dir = Path(cache_dir)
    mmlu_dir = cache_dir / "alexandrainst__m_mmlu"
    mmlu_dir.mkdir(parents=True, exist_ok=True)
    local_file = mmlu_dir / "test.jsonl"

    if not local_file.exists():
        url = (
            "https://huggingface.co/datasets/alexandrainst/m_mmlu/"
            "resolve/main/data/en/test.jsonl"
        )
        _download_json(url, local_file)

    tasks: list[Task] = []
    choice_letters = ["A", "B", "C", "D"]
    with open(local_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            instruction = item.get("instruction") or item.get("question", "")
            choices = [
                item.get("option_a", ""),
                item.get("option_b", ""),
                item.get("option_c", ""),
                item.get("option_d", ""),
            ]
            answer_letter = str(item.get("answer", "")).strip().upper()
            if not instruction or not any(choices):
                continue
            # filter by subject prefix in id (e.g. "math/test/0")
            item_id: str = item.get("id", "")
            if subject and not item_id.lower().startswith(subject.lower() + "/"):
                continue
            # format: instruction followed by A) ... B) ... C) ... D) ...
            prompt_lines = [instruction]
            for i, ch in enumerate(choices[:4]):
                if ch:
                    prompt_lines.append(f"{choice_letters[i]}) {ch}")
            prompt = "\n".join(prompt_lines)
            tasks.append(Task(
                id=f"mmlu-{len(tasks)}",
                prompt=prompt,
                answer=answer_letter,
                kind="mmlu",
                difficulty=3,  # MMLU is uniformly difficulty ~3
            ))
    return tasks


# ------------------------------------------------------------------
# LiveCodeBench
# ------------------------------------------------------------------

def load_livecodebench(cache_dir: str | Path = "data/livecodebench",
                        n: int = 200) -> list[Task]:
    """Load LiveCodeBench (Bang et al.) code generation tasks.

    Dataset: livecodebench/livecodebench_original
    Each item has: question_id, question, answer (canonical code solution),
    difficulty, generated_tests (list of test cases).

    We use the problem description as the prompt and the canonical solution
    answer.  The verifier will check whether the LLM's final answer contains
    the expected output or a plausible solution.

    Since LiveCodeBench is primarily judged by code execution (which we can't
    do without a sandbox), we set difficulty from the dataset's difficulty field
    and store the canonical solution as the answer.  The extract_final_answer
    function will fall back to returning the last line of code.

    Args:
        cache_dir:  local cache directory
        n:          maximum number of tasks to load (0 = all)
    """
    cache_dir = Path(cache_dir)
    lcb_dir = cache_dir / "livecodebench__code_generation_lite"
    lcb_dir.mkdir(parents=True, exist_ok=True)
    local_file = lcb_dir / "test.jsonl"

    if not local_file.exists():
        url = (
            "https://huggingface.co/datasets/livecodebench/code_generation_lite/"
            "resolve/main/test.jsonl"
        )
        _download_json(url, local_file)

    tasks: list[Task] = []
    difficulty_map = {"easy": 1, "medium": 3, "hard": 5}
    with open(local_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt = item.get("question", "")
            answer = item.get("answer", "```\n# solution\n```")
            difficulty_str = str(item.get("difficulty", "medium")).lower()
            difficulty = difficulty_map.get(difficulty_str, 3)
            tasks.append(Task(
                id=f"lcb-{item.get('question_id', len(tasks))}",
                prompt=prompt,
                answer=answer,
                kind="code",
                difficulty=difficulty,
            ))
            if 0 < n <= len(tasks):
                break
    return tasks


# ------------------------------------------------------------------
# Convenience loader
# ------------------------------------------------------------------

def load_benchmark(name: Literal["math500", "mmlu", "livecodebench"],
                   cache_dir: str | Path = "data",
                   **kwargs) -> list[Task]:
    """Load a benchmark by name.  Delegates to the specific loader."""
    if name == "math500":
        return load_math500(cache_dir=Path(cache_dir) / "math500", **kwargs)
    if name == "mmlu":
        return load_mmlu(cache_dir=Path(cache_dir) / "mmlu", **kwargs)
    if name == "livecodebench":
        return load_livecodebench(cache_dir=Path(cache_dir) / "livecodebench", **kwargs)
    raise ValueError(f"Unknown benchmark: {name!r}.  Known: math500, mmlu, livecodebench.")
