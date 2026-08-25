import subprocess
import sys
import types
import unittest
from importlib import metadata
from unittest.mock import patch

from utils.peft_runtime import (
    remove_optional_torchao,
    sanitize_optional_torchao,
    validate_lora_injection,
)


class PeftRuntimeTest(unittest.TestCase):
    def test_remove_optional_torchao_is_a_noop_when_absent(self):
        with (
            patch(
                "utils.peft_runtime.metadata.version",
                side_effect=metadata.PackageNotFoundError,
            ),
            patch("utils.peft_runtime.subprocess.run") as run,
        ):
            self.assertIsNone(remove_optional_torchao())
        run.assert_not_called()

    def test_remove_optional_torchao_uses_the_active_python(self):
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        with (
            patch(
                "utils.peft_runtime.metadata.version",
                return_value="0.10.0",
            ),
            patch(
                "utils.peft_runtime.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            self.assertEqual(remove_optional_torchao(), "0.10.0")
        run.assert_called_once_with(
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

    def test_sanitize_keeps_compatible_torchao(self):
        with (
            patch(
                "utils.peft_runtime.metadata.version",
                return_value="0.16.0",
            ),
            patch(
                "utils.peft_runtime.validate_lora_injection",
                return_value=8,
            ),
            patch("utils.peft_runtime.remove_optional_torchao") as remove,
        ):
            self.assertIsNone(sanitize_optional_torchao())
        remove.assert_not_called()

    def test_sanitize_removes_only_incompatible_torchao(self):
        with (
            patch(
                "utils.peft_runtime.metadata.version",
                return_value="0.10.0",
            ),
            patch(
                "utils.peft_runtime.validate_lora_injection",
                side_effect=RuntimeError(
                    "Original error: Found an incompatible version of torchao"
                ),
            ),
            patch(
                "utils.peft_runtime.remove_optional_torchao",
                return_value="0.10.0",
            ) as remove,
        ):
            self.assertEqual(sanitize_optional_torchao(), "0.10.0")
        remove.assert_called_once_with()

    def test_sanitize_preserves_torchao_for_unrelated_peft_failure(self):
        with (
            patch(
                "utils.peft_runtime.metadata.version",
                return_value="0.10.0",
            ),
            patch(
                "utils.peft_runtime.validate_lora_injection",
                side_effect=RuntimeError("Target module is unsupported"),
            ),
            patch("utils.peft_runtime.remove_optional_torchao") as remove,
        ):
            with self.assertRaisesRegex(RuntimeError, "Target module"):
                sanitize_optional_torchao()
        remove.assert_not_called()

    def test_lora_smoke_check_exercises_peft_adapter_injection(self):
        class FakeModule:
            pass

        class FakeLinear:
            def __init__(self, input_features, output_features):
                self.input_features = input_features
                self.output_features = output_features

        class FakeParameter:
            requires_grad = True

            def numel(self):
                return 8

        class FakeAdaptedModel:
            def named_parameters(self):
                return [("projection.lora_A.default.weight", FakeParameter())]

        observed = {}

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                observed["config"] = kwargs

        def fake_get_peft_model(model, config):
            observed["model"] = model
            observed["adapter_config"] = config
            return FakeAdaptedModel()

        fake_torch = types.SimpleNamespace(
            nn=types.SimpleNamespace(Module=FakeModule, Linear=FakeLinear)
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            get_peft_model=fake_get_peft_model,
        )

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "peft": fake_peft},
        ):
            trainable = validate_lora_injection()

        self.assertEqual(trainable, 8)
        self.assertEqual(observed["config"]["target_modules"], ["projection"])

    def test_lora_smoke_check_explains_optional_dependency_failure(self):
        class FakeModule:
            pass

        class FakeLinear:
            def __init__(self, input_features, output_features):
                pass

        def fail_injection(model, config):
            raise ImportError("Found an incompatible version of torchao")

        fake_torch = types.SimpleNamespace(
            nn=types.SimpleNamespace(Module=FakeModule, Linear=FakeLinear)
        )
        fake_peft = types.SimpleNamespace(
            LoraConfig=lambda **kwargs: object(),
            get_peft_model=fail_injection,
        )

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "peft": fake_peft},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "utils.peft_runtime sanitize",
            ):
                validate_lora_injection()

    def test_lora_smoke_check_wraps_import_time_torchao_failure(self):
        class BrokenPeft(types.ModuleType):
            def __getattr__(self, name):
                if name in {"LoraConfig", "get_peft_model"}:
                    raise ImportError("Found an incompatible version of torchao")
                raise AttributeError(name)

        fake_torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object))
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "peft": BrokenPeft("peft")},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Found an incompatible version of torchao",
            ):
                validate_lora_injection()

if __name__ == "__main__":
    unittest.main()
