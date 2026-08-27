from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_wdc_target_alignment import AlignmentError, check_alignment


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows(source: str) -> list[dict]:
    return [
        {
            "dataset_id": "wdc",
            "input_text": f"pair input {index}",
            "label_source": source,
            "pair_id": f"p{index}",
            "split": "train",
            "target_text": "match" if source == "gold" or index == 0 else "non-match",
        }
        for index in range(3)
    ]


class WdcTargetAlignmentTests(unittest.TestCase):
    def _check(self, gold_rows: list[dict], llm_rows: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.jsonl"
            llm_path = root / "llm_hard.jsonl"
            _write_rows(gold_path, gold_rows)
            _write_rows(llm_path, llm_rows)
            return check_alignment(gold_path, llm_path, expected_rows=3)

    def test_accepts_aligned_pairs_with_different_labels(self) -> None:
        summary = self._check(_rows("gold"), _rows("llm_hard"))

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["row_count_per_arm"], 3)
        self.assertEqual(summary["label_disagreements"], 2)

    def test_rejects_wrong_row_count(self) -> None:
        with self.assertRaisesRegex(AlignmentError, "expected 3"):
            self._check(_rows("gold")[:-1], _rows("llm_hard"))

    def test_rejects_duplicate_pair_id(self) -> None:
        llm_rows = _rows("llm_hard")
        llm_rows[2]["pair_id"] = "p1"

        with self.assertRaisesRegex(AlignmentError, "repeats pair_id"):
            self._check(_rows("gold"), llm_rows)

    def test_rejects_different_pair_order(self) -> None:
        llm_rows = _rows("llm_hard")
        llm_rows[0], llm_rows[1] = llm_rows[1], llm_rows[0]

        with self.assertRaisesRegex(AlignmentError, "ordered pair_id mismatch"):
            self._check(_rows("gold"), llm_rows)

    def test_rejects_different_pair_input(self) -> None:
        llm_rows = _rows("llm_hard")
        llm_rows[1]["input_text"] = "different pair input"

        with self.assertRaisesRegex(AlignmentError, "input_text"):
            self._check(_rows("gold"), llm_rows)


if __name__ == "__main__":
    unittest.main()
