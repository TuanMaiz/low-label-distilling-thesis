"""Answer-only prompt construction and parsing for WDC entity matching."""
from __future__ import annotations

from typing import Literal

from supervision.config import DEFAULT_PROMPT_VERSION


PROMPT_VERSION = DEFAULT_PROMPT_VERSION
VALID_LABELS = ("match", "non_match")
PromptMode = Literal["teacher_label", "direct_prediction"]

SYSTEM_PROMPT = (
    "You are an entity matching classifier. Return exactly one label and no "
    "other text."
)


def build_answer_only_prompt(pair_row: dict, mode: PromptMode = "teacher_label") -> str:
    """
    Build an answer-only ER prompt for one serialized pair.

    The gold label is intentionally not included. Teacher-label generation and
    direct LLM matching both ask the model to classify the pair from the record
    attributes alone.
    """
    if mode not in {"teacher_label", "direct_prediction"}:
        raise ValueError(f"Unsupported prompt mode: {mode}")

    return "\n".join(
        [
            "Task: decide whether Record A and Record B refer to the same real-world product.",
            "",
            "Return exactly one of these labels:",
            "- match: the two records refer to the same real-world product.",
            "- non_match: the two records refer to different real-world products.",
            "",
            "Do not explain your answer. Do not output JSON. Do not add punctuation.",
            f"pair_id: {pair_row['pair_id']}",
            "",
            "serialized_pair:",
            pair_row["input_text"],
            "",
            "label:",
        ]
    )


def parse_answer_only_label(raw_answer: str | None) -> Literal["match", "non_match"] | None:
    """
    Parse a strict answer-only model response.

    Only canonical labels, optional surrounding quotes/backticks, and the common
    hyphenated ``non-match`` spelling are accepted. Explanations such as
    ``match.`` or ``The answer is match`` remain invalid.
    """
    if raw_answer is None:
        return None

    normalized = raw_answer.strip().lower()
    for quote in ('"', "'", "`"):
        if len(normalized) >= 2 and normalized[0] == normalized[-1] == quote:
            normalized = normalized[1:-1].strip()
            break

    if normalized == "match":
        return "match"
    if normalized in {"non_match", "non-match"}:
        return "non_match"
    return None
