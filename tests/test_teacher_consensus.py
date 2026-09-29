from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from supervision.build_teacher_consensus import (
    CONSISTENCY_HEADER,
    PREDICTION_HEADER,
    build_teacher_consensus,
)


OUTPUT_NAMES = ("llm_unrefined.csv", "consistency.csv", "summary.json")


def _read_csv_rows(
    path: Path, expected_header: list[str], output_name: str
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise AssertionError(f"{output_name} output has an unexpected header")
        return list(reader)


def _artifact_bytes(output_dir: Path) -> dict[str, bytes]:
    return {name: (output_dir / name).read_bytes() for name in OUTPUT_NAMES}


def _assert_total(summary: dict[str, Any], field: str, expected_count: int) -> None:
    if sum(summary[field].values()) != expected_count:
        raise AssertionError(f"{field.replace('_', ' ')} do not sum to expected count")


def verify_teacher_consensus_files(
    *,
    inputs_path: Path,
    pass_paths: Sequence[Path],
    expected_count: int,
) -> dict[str, Any]:
    """Exercise the builder against caller-supplied artifacts without retaining outputs."""
    with tempfile.TemporaryDirectory(prefix="teacher-consensus-check-") as temporary_directory:
        output_dir = Path(temporary_directory) / "outputs"
        summary = build_teacher_consensus(
            inputs_path=inputs_path,
            pass_paths=pass_paths,
            output_dir=output_dir,
            expected_count=expected_count,
        )
        if summary["row_count"] != expected_count:
            raise AssertionError("summary row count differs from expected count")
        _assert_total(summary, "majority_class_counts", expected_count)
        _assert_total(summary, "consistency_counts", expected_count)

        first_bytes = _artifact_bytes(output_dir)
        majority_rows = _read_csv_rows(
            output_dir / "llm_unrefined.csv", PREDICTION_HEADER, "majority"
        )
        consistency_rows = _read_csv_rows(
            output_dir / "consistency.csv", CONSISTENCY_HEADER, "consistency"
        )
        if len(majority_rows) != expected_count or len(consistency_rows) != expected_count:
            raise AssertionError("output row counts differ from expected count")
        if [row["pair_id"] for row in majority_rows] != [
            row["pair_id"] for row in consistency_rows
        ]:
            raise AssertionError("majority and consistency output pair order differs")

        second_summary = build_teacher_consensus(
            inputs_path=inputs_path,
            pass_paths=pass_paths,
            output_dir=output_dir,
            expected_count=expected_count,
        )
        second_bytes = _artifact_bytes(output_dir)
        if second_summary != summary or second_bytes != first_bytes:
            raise AssertionError("repeated build did not produce identical artifacts")
        return summary


def _real_files_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test teacher consensus against caller-supplied real artifact paths."
    )
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--pass-01", type=Path, required=True)
    parser.add_argument("--pass-02", type=Path, required=True)
    parser.add_argument("--pass-03", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser


class TeacherConsensusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.inputs_path = self.root / "inputs.jsonl"
        self.pass_paths = [self.root / f"pass_{index:02d}.csv" for index in range(1, 4)]
        self.output_dir = self.root / "majority"
        self.pair_ids = ["pair-1", "pair-2", "pair-3", "pair-4"]
        self._write_inputs()
        self._write_passes(
            ["match", "match", "non_match", "non_match"],
            ["match", "non_match", "match", "non_match"],
            ["match", "match", "non_match", "match"],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_inputs(self, *, include_label: bool = False) -> None:
        rows = []
        for pair_id in self.pair_ids:
            row = {"pair_id": pair_id, "input_text": f"input for {pair_id}"}
            if include_label:
                row["label"] = "match"
            rows.append(row)
        self.inputs_path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _write_prediction(self, path: Path, pair_ids: list[str], labels: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["pair_id", "result"])
            writer.writerows(zip(pair_ids, labels, strict=True))

    def _write_passes(self, *passes: list[str]) -> None:
        for path, labels in zip(self.pass_paths, passes, strict=True):
            self._write_prediction(path, self.pair_ids, labels)

    def _build(self) -> dict[str, object]:
        return build_teacher_consensus(
            inputs_path=self.inputs_path,
            pass_paths=self.pass_paths,
            output_dir=self.output_dir,
            expected_count=4,
        )

    def test_builds_majority_and_consistency_artifacts(self) -> None:
        summary = self._build()

        with (self.output_dir / "llm_unrefined.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            majority_rows = list(csv.DictReader(handle))
        self.assertEqual(
            majority_rows,
            [
                {"pair_id": "pair-1", "result": "match"},
                {"pair_id": "pair-2", "result": "match"},
                {"pair_id": "pair-3", "result": "non_match"},
                {"pair_id": "pair-4", "result": "non_match"},
            ],
        )
        with (self.output_dir / "consistency.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            consistency_rows = list(csv.DictReader(handle))
        self.assertEqual(
            [row["consistency"] for row in consistency_rows],
            ["3/3", "2/3", "2/3", "2/3"],
        )
        self.assertEqual(consistency_rows[1]["majority_votes"], "2")
        self.assertEqual(consistency_rows[1]["majority_label"], "match")
        self.assertEqual(consistency_rows[2]["majority_label"], "non_match")
        self.assertEqual(summary["majority_class_counts"], {"match": 2, "non_match": 2})
        self.assertEqual(summary["consistency_counts"], {"3/3": 1, "2/3": 3})

        persisted_summary = json.loads((self.output_dir / "summary.json").read_text("utf-8"))
        for output_name in ("llm_unrefined.csv", "consistency.csv"):
            expected_hash = hashlib.sha256((self.output_dir / output_name).read_bytes()).hexdigest()
            self.assertEqual(persisted_summary["outputs"][output_name]["sha256"], expected_hash)

    def test_repeated_build_is_byte_identical(self) -> None:
        self._build()
        first = {
            name: (self.output_dir / name).read_bytes()
            for name in ("llm_unrefined.csv", "consistency.csv", "summary.json")
        }
        self._build()
        second = {name: (self.output_dir / name).read_bytes() for name in first}
        self.assertEqual(second, first)

    def test_rejects_reordered_missing_duplicate_and_invalid_pass_rows(self) -> None:
        invalid_cases = [
            (self.pair_ids[::-1], ["match"] * 4),
            (self.pair_ids[:-1], ["match"] * 3),
            (["pair-1", "pair-1", "pair-3", "pair-4"], ["match"] * 4),
            (self.pair_ids, ["match", "unknown", "match", "match"]),
        ]
        for pair_ids, labels in invalid_cases:
            with self.subTest(pair_ids=pair_ids, labels=labels):
                self._write_prediction(self.pass_paths[1], pair_ids, labels)
                with self.assertRaises(ValueError):
                    self._build()

    def test_rejects_hidden_label_fields_in_blinded_inputs(self) -> None:
        self._write_inputs(include_label=True)
        with self.assertRaisesRegex(ValueError, "only pair_id and input_text"):
            self._build()
        self.assertFalse(self.output_dir.exists())

    def test_real_file_verifier_exercises_caller_supplied_paths(self) -> None:
        summary = verify_teacher_consensus_files(
            inputs_path=self.inputs_path,
            pass_paths=self.pass_paths,
            expected_count=4,
        )
        self.assertEqual(summary["row_count"], 4)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "real-files":
        arguments = _real_files_parser().parse_args(sys.argv[2:])
        real_summary = verify_teacher_consensus_files(
            inputs_path=arguments.inputs,
            pass_paths=(arguments.pass_01, arguments.pass_02, arguments.pass_03),
            expected_count=arguments.expected_count,
        )
        print(json.dumps(real_summary, indent=2, sort_keys=True))
    else:
        unittest.main()
