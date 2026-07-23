"""Validated configuration for compact Entity Matching students."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


StudentArchitecture = Literal[
    "seq2seq",
    "sequence_classification",
    "generative_reranker",
]
SUPPORTED_ARCHITECTURES = {
    "seq2seq",
    "sequence_classification",
    "generative_reranker",
}
_STUDENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_EXPECTED_LABELS = {"non-match", "match"}
_RERANKER_REQUIRED_FIELDS = {
    "reranker_instruction",
    "reranker_positive_token",
    "reranker_negative_token",
    "fine_tuning_method",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "gradient_checkpointing",
}


@dataclass(frozen=True)
class StudentConfig:
    student_id: str
    model_name: str
    architecture: StudentArchitecture
    tokenizer_use_fast: bool
    num_labels: int | None
    label_to_id: dict[str, int]
    model_revision: str | None = None
    max_input_length: int = 512
    input_truncation: bool = True
    reranker_instruction: str | None = None
    reranker_positive_token: str | None = None
    reranker_negative_token: str | None = None
    fine_tuning_method: str | None = None
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules: tuple[str, ...] | None = None
    gradient_checkpointing: bool = False

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
    optional = {
        "max_input_length",
        "input_truncation",
        "model_revision",
        *_RERANKER_REQUIRED_FIELDS,
    }
    extra = sorted(payload.keys() - required - optional)
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
    model_revision = payload.get("model_revision")
    architecture_default_input_length = {
        "seq2seq": 512,
        "sequence_classification": 2400,
        "generative_reranker": 4096,
    }.get(architecture, 512)
    max_input_length = payload.get(
        "max_input_length",
        architecture_default_input_length,
    )
    input_truncation = payload.get(
        "input_truncation",
        architecture == "seq2seq",
    )

    if not isinstance(student_id, str) or not _STUDENT_ID_PATTERN.fullmatch(student_id):
        raise ValueError("student_id must be lowercase kebab-case")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty Hugging Face model identifier")
    if model_revision is not None and (
        not isinstance(model_revision, str) or not model_revision.strip()
    ):
        raise ValueError("model_revision must be a non-empty string when supplied")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported student architecture {architecture!r}; expected one of "
            f"{sorted(SUPPORTED_ARCHITECTURES)}"
        )
    if not isinstance(tokenizer_use_fast, bool):
        raise ValueError("tokenizer_use_fast must be a boolean")
    if architecture == "sequence_classification" and num_labels != 2:
        raise ValueError("sequence_classification students require num_labels=2")
    if architecture == "generative_reranker" and num_labels != 2:
        raise ValueError("generative_reranker students require num_labels=2")
    if architecture == "seq2seq" and num_labels is not None:
        raise ValueError("seq2seq students require num_labels=null")
    if not isinstance(label_to_id, dict) or set(label_to_id) != _EXPECTED_LABELS:
        raise ValueError("label_to_id must define exactly 'non-match' and 'match'")
    if set(label_to_id.values()) != {0, 1}:
        raise ValueError("label_to_id values must be the distinct class IDs 0 and 1")
    if not isinstance(max_input_length, int) or isinstance(max_input_length, bool):
        raise ValueError("max_input_length must be an integer")
    if max_input_length <= 0:
        raise ValueError("max_input_length must be positive")
    if not isinstance(input_truncation, bool):
        raise ValueError("input_truncation must be a boolean")
    if architecture in {"sequence_classification", "generative_reranker"} and input_truncation:
        raise ValueError(f"{architecture} students require input_truncation=false")

    reranker_values = {
        field: payload.get(field)
        for field in _RERANKER_REQUIRED_FIELDS
    }
    supplied_reranker_fields = {
        field for field, value in reranker_values.items() if value is not None
    }
    if architecture == "generative_reranker":
        missing_reranker = sorted(_RERANKER_REQUIRED_FIELDS - payload.keys())
        if missing_reranker:
            raise ValueError(
                "generative_reranker config is missing fields: "
                + ", ".join(missing_reranker)
            )
        for field in (
            "reranker_instruction",
            "reranker_positive_token",
            "reranker_negative_token",
        ):
            value = reranker_values[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if (
            reranker_values["reranker_positive_token"]
            == reranker_values["reranker_negative_token"]
        ):
            raise ValueError("reranker positive and negative tokens must differ")
        if reranker_values["fine_tuning_method"] != "lora":
            raise ValueError("generative_reranker fine_tuning_method must be 'lora'")
        for field in ("lora_rank", "lora_alpha"):
            value = reranker_values[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        lora_dropout = reranker_values["lora_dropout"]
        if (
            not isinstance(lora_dropout, (int, float))
            or isinstance(lora_dropout, bool)
            or not 0.0 <= float(lora_dropout) < 1.0
        ):
            raise ValueError("lora_dropout must be in [0, 1)")
        target_modules = reranker_values["lora_target_modules"]
        if (
            not isinstance(target_modules, list)
            or not target_modules
            or any(not isinstance(module, str) or not module for module in target_modules)
            or len(set(target_modules)) != len(target_modules)
        ):
            raise ValueError("lora_target_modules must be a non-empty list of unique strings")
        if not isinstance(reranker_values["gradient_checkpointing"], bool):
            raise ValueError("gradient_checkpointing must be a boolean")
    elif supplied_reranker_fields:
        raise ValueError(
            f"{architecture} config cannot define generative reranker fields"
        )

    return StudentConfig(
        student_id=student_id,
        model_name=model_name,
        architecture=architecture,
        tokenizer_use_fast=tokenizer_use_fast,
        num_labels=num_labels,
        label_to_id={str(label): int(label_id) for label, label_id in label_to_id.items()},
        model_revision=model_revision,
        max_input_length=max_input_length,
        input_truncation=input_truncation,
        reranker_instruction=reranker_values["reranker_instruction"],
        reranker_positive_token=reranker_values["reranker_positive_token"],
        reranker_negative_token=reranker_values["reranker_negative_token"],
        fine_tuning_method=reranker_values["fine_tuning_method"],
        lora_rank=reranker_values["lora_rank"],
        lora_alpha=reranker_values["lora_alpha"],
        lora_dropout=(
            float(reranker_values["lora_dropout"])
            if reranker_values["lora_dropout"] is not None
            else None
        ),
        lora_target_modules=(
            tuple(reranker_values["lora_target_modules"])
            if reranker_values["lora_target_modules"] is not None
            else None
        ),
        gradient_checkpointing=bool(reranker_values["gradient_checkpointing"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--field",
        choices=(
            "student_id",
            "model_name",
            "model_revision",
            "architecture",
            "tokenizer_use_fast",
            "num_labels",
            "max_input_length",
            "input_truncation",
            "reranker_instruction",
            "reranker_positive_token",
            "reranker_negative_token",
            "fine_tuning_method",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "gradient_checkpointing",
        ),
    )
    args = parser.parse_args()
    config = load_student_config(args.config)
    if args.field:
        value = getattr(config, args.field)
        if isinstance(value, bool):
            print(str(value).lower())
        elif isinstance(value, tuple):
            print(",".join(value))
        elif value is not None:
            print(value)
        return
    print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
