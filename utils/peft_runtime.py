"""Prepare and validate the optional-dependency surface used by PEFT LoRA."""
from __future__ import annotations

import argparse
from importlib import metadata
import subprocess
import sys


def remove_optional_torchao() -> str | None:
    """Remove Colab's optional TorchAO package because this run does not use it."""
    try:
        version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return None

    print(
        "Removing optional torchao="
        f"{version}; Phase 5 LoRA training does not use TorchAO quantization."
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "--yes",
            "torchao",
        ],
        check=True,
    )
    return version


def validate_lora_injection() -> int:
    """Exercise PEFT's adapter dispatcher without loading a remote model."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model

        class TinyProjection(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Linear(2, 2)

        adapted = get_peft_model(
            TinyProjection(),
            LoraConfig(
                r=1,
                lora_alpha=1,
                target_modules=["projection"],
                bias="none",
            ),
        )
    except ImportError as exc:
        raise RuntimeError(
            "PEFT LoRA adapter injection failed because an optional dependency "
            "is incompatible. Run `python -m utils.peft_runtime sanitize`, then "
            "`python -m utils.peft_runtime check` in the training runtime. "
            f"Original error: {exc}"
        ) from exc

    trainable = sum(
        parameter.numel()
        for name, parameter in adapted.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )
    if trainable <= 0:
        raise RuntimeError("PEFT LoRA smoke check created no trainable adapter weights")
    return trainable


def sanitize_optional_torchao() -> str | None:
    """Remove TorchAO only when it is the proven cause of LoRA injection failure."""
    try:
        version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return None

    try:
        validate_lora_injection()
    except RuntimeError as exc:
        if "incompatible version of torchao" not in str(exc).lower():
            raise
        return remove_optional_torchao()

    print(
        f"Optional torchao={version} is compatible with PEFT; keeping it installed."
    )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("sanitize", "check"))
    args = parser.parse_args()

    try:
        if args.action == "sanitize":
            removed = sanitize_optional_torchao()
            if removed is None:
                print("No incompatible TorchAO installation found; no cleanup needed.")
            return
        trainable = validate_lora_injection()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"PEFT runtime check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"PEFT LoRA runtime check passed: trainable_adapter_parameters={trainable}")


if __name__ == "__main__":
    main()
