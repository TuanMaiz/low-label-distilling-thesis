"""PyTorch runtime helpers for reproducible mixed-precision execution."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from typing import ContextManager

import torch


PRECISION_CHOICES = ("auto", "fp32", "fp16", "bf16")


def resolve_precision(device: str, requested: str = "auto") -> str:
    """Resolve an explicit precision or choose a safe default for the device."""
    precision = requested.strip().lower()
    if precision not in PRECISION_CHOICES:
        raise ValueError(
            f"Unsupported precision {requested!r}; choose one of {PRECISION_CHOICES}"
        )

    device_type = torch.device(device).type
    if precision == "auto":
        if device_type != "cuda":
            return "fp32"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if precision in {"fp16", "bf16"} and device_type != "cuda":
        raise ValueError(f"{precision} requires a CUDA device; received {device!r}")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 was requested, but the CUDA device does not report BF16 support")
    return precision


def resolve_validation_batch_size(
    device: str,
    train_batch_size: int,
    requested: int | None = None,
    precision: str = "auto",
) -> int:
    """Resolve the validation-only batch size from the actual runtime precision."""
    if requested is not None:
        if requested <= 0:
            raise ValueError("validation_batch_size must be positive")
        return requested
    if torch.device(device).type != "cuda":
        return train_batch_size
    return 32 if resolve_precision(device, precision) == "bf16" else 16


def runtime_identity(
    device: str,
    precision: str,
    train_batch_size: int,
    validation_batch_size: int | None,
) -> tuple[str, int, str]:
    """Return resolved precision, validation batch, and physical device name."""
    resolved_precision = resolve_precision(device, precision)
    resolved_validation_batch = resolve_validation_batch_size(
        device,
        train_batch_size,
        validation_batch_size,
        precision,
    )
    device_object = torch.device(device)
    device_name = (
        torch.cuda.get_device_name(device_object)
        if device_object.type == "cuda"
        else "cpu"
    )
    return resolved_precision, resolved_validation_batch, device_name


def autocast_context(device: str, precision: str) -> ContextManager:
    """Return the autocast context for an already-resolved precision."""
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def create_grad_scaler(device: str, precision: str):
    """Create an FP16 gradient scaler while retaining PyTorch 2.0 compatibility."""
    if precision != "fp16" or torch.device(device).type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=PRECISION_CHOICES, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--validation-batch-size")
    args = parser.parse_args()
    requested_validation_batch = (
        None
        if args.validation_batch_size in {None, "auto"}
        else int(args.validation_batch_size)
    )
    resolved_precision, resolved_validation_batch, device_name = runtime_identity(
        args.device,
        args.precision,
        args.train_batch_size,
        requested_validation_batch,
    )
    print(resolved_precision)
    print(resolved_validation_batch)
    print(device_name)


if __name__ == "__main__":
    main()
