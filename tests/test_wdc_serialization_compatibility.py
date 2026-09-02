from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data.er_dataset_loader import WDCProductsConfig, load_wdc_products_pairwise
from data.serialize_pairs import write_serialized_pairs


REPO_ROOT = Path(__file__).resolve().parents[1]


class WdcSerializationCompatibilityTests(unittest.TestCase):
    def test_fresh_compatibility_serialization_is_byte_identical(self) -> None:
        source = REPO_ROOT / "data/raw/wdc_products/80pair.zip"
        existing = REPO_ROOT / "data/cache/wdc_products/serialized"
        if not source.is_file() or not existing.is_dir():
            self.skipTest("local WDC source/cache not available")
        splits = load_wdc_products_pairwise(
            source,
            WDCProductsConfig(corner_cases=80, train_size="small", test_unseen=100),
        )
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary)
            for split in ("train", "validation", "test"):
                output = generated / f"{split}.jsonl"
                write_serialized_pairs(splits[split], output)
                self.assertEqual(
                    output.read_bytes(),
                    (existing / f"{split}.jsonl").read_bytes(),
                    f"WDC {split} serialization changed",
                )


if __name__ == "__main__":
    unittest.main()
