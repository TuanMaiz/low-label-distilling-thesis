"""Validated configuration for compact Entity Matching students."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


StudentArchitecture = Literal["seq2seq", "sequence_classification"]
SUPPORTED_ARCHITECTURES = {"seq2seq", "sequence_classification"}
_STUDENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_EXPECTED_LABELS = {"non-match", "match"}


@dataclass(frozen=True)
class StudentConfig:
    student_id: str
    model_name: str
    architecture: StudentArchitecture
    tokenizer_use_fast: bool
    num_labels: int | None
    label_to_id: dict[str, int]

    @property
    def id_to_label(self) -> dict[int, str]:
        return {label_id: label for label, label_id in self.label_to_id.items()}

    def to_dict(self) -> dict:
        return asdict(self)


def load_student_config(path: Path) -> StudentConfig:
    """Load and validate one JSON student configuration."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Student config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Student config is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Student config must contain a JSON object: {path}")

    required = {
        "student_id",
        "model_name",
        "architecture",
        "tokenizer_use_fast",
        "num_labels",
        "label_to_id",
    }
    missing = sorted(required - payload.keys())
    extra = sorted(payload.keys() - required)
    if missing:
        raise ValueError(f"Student config is missing fields: {', '.join(missing)}")
    if extra:
        raise ValueError(f"Student config has unsupported fields: {', '.join(extra)}")

    student_id = payload["student_id"]
    model_name = payload["model_name"]
    architecture = payload["architecture"]
    tokenizer_use_fast = payload["tokenizer_use_fast"]
    num_labels = payload["num_labels"]
    label_to_id = payload["label_to_id"]

    if not isinstance(student_id, str) or not _STUDENT_ID_PATTERN.fullmatch(student_id):
        raise ValueError("student_id must be lowercase kebab-case")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty Hugging Face model identifier")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported student architecture {architecture!r}; expected one of "
            f"{sorted(SUPPORTED_ARCHITECTURES)}"
        )
    if not isinstance(tokenizer_use_fast, bool):
        raise ValueError("tokenizer_use_fast must be a boolean")
    if architecture == "sequence_classification" and num_labels != 2:
        raise ValueError("sequence_classification students require num_labels=2")
    if architecture == "seq2seq" and num_labels is not None:
        raise ValueError("seq2seq students require num_labels=null")
    if not isinstance(label_to_id, dict) or set(label_to_id) != _EXPECTED_LABELS:
        raise ValueError("label_to_id must define exactly 'non-match' and 'match'")
    if set(label_to_id.values()) != {0, 1}:
        raise ValueError("label_to_id values must be the distinct class IDs 0 and 1")

    return StudentConfig(
        student_id=student_id,
        model_name=model_name,
        architecture=architecture,
        tokenizer_use_fast=tokenizer_use_fast,
        num_labels=num_labels,
        label_to_id={str(label): int(label_id) for label, label_id in label_to_id.items()},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=(
            "student_id",
            "model_name",
            "architecture",
            "tokenizer_use_fast",
            "num_labels",
        ),
    )
    args = parser.parse_args()
    config = load_student_config(args.config)
    if args.field:
        value = getattr(config, args.field)
        if isinstance(value, bool):
            print(str(value).lower())
        elif value is not None:
            print(value)
        return
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
