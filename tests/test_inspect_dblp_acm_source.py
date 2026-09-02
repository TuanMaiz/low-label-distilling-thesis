from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.inspect_dblp_acm_source import FILENAMES, inspect


class InspectDblpAcmSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.extracted = self.root / "extracted"
        self.direct = self.root / "direct"
        self.extracted.mkdir()
        self.direct.mkdir()
        self.archive = self.root / "source.zip"
        self._write_fixture()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_csv(self, path: Path, header: list[str], rows: list[list[object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)

    def _write_fixture(self) -> None:
        table_header = ["id", "title", "authors", "venue", "year"]
        pair_header = ["ltable_id", "rtable_id", "label"]
        contents = {
            "tableA.csv": (table_header, [[0, "A", "Ann", "V", 2000], [1, "B", "Bob", "W", 2001]]),
            "tableB.csv": (table_header, [[0, "A", "Ann", "V", 2000], [1, "C", "", "X", 2002]]),
            "train.csv": (pair_header, [[0, 0, 1], [1, 1, 0]]),
            "valid.csv": (pair_header, [[0, 1, 0]]),
            "test.csv": (pair_header, [[1, 0, 0]]),
        }
        for filename, (header, rows) in contents.items():
            self._write_csv(self.extracted / filename, header, rows)
            self._write_csv(self.direct / filename, header, rows)
        self._write_archive()

    def _write_archive(self, *, extra_member: bool = False, duplicate_member: bool = False) -> None:
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for filename in FILENAMES:
                archive.write(self.extracted / filename, f"exp_data/{filename}")
            if extra_member:
                archive.writestr("exp_data/extra.csv", "x\n")
            if duplicate_member:
                archive.write(self.extracted / "train.csv", "exp_data/train.csv")

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            archive=self.archive,
            source_root=self.extracted,
            direct_root=self.direct,
            observed_on="2026-09-01",
        )

    def test_deterministic_and_test_lock_boundary(self) -> None:
        first = inspect(self._args())
        second = inspect(self._args())
        self.assertEqual(
            json.dumps(first, sort_keys=True),
            json.dumps(second, sort_keys=True),
        )
        self.assertEqual(first["locked_test"], {
            "header": ["ltable_id", "rtable_id", "label"],
            "row_count": 1,
        })
        self.assertNotIn("test", first["pairs"])
        self.assertEqual(set(first["cross_split_overlap"]), {"train_valid"})

    def test_rejects_direct_copy_mismatch(self) -> None:
        with (self.direct / "train.csv").open("a", encoding="utf-8") as handle:
            handle.write("0,1,0\n")
        with self.assertRaisesRegex(ValueError, "mismatch for train.csv"):
            inspect(self._args())

    def test_rejects_extra_csv_column(self) -> None:
        self._write_csv(
            self.extracted / "tableA.csv",
            ["id", "title", "authors", "venue", "year", "extra"],
            [[0, "A", "Ann", "V", 2000, "x"]],
        )
        self._write_csv(
            self.direct / "tableA.csv",
            ["id", "title", "authors", "venue", "year", "extra"],
            [[0, "A", "Ann", "V", 2000, "x"]],
        )
        self._write_archive()
        with self.assertRaisesRegex(ValueError, "header mismatch"):
            inspect(self._args())

    def test_rejects_extra_archive_member(self) -> None:
        self._write_archive(extra_member=True)
        with self.assertRaisesRegex(ValueError, "archive member mismatch"):
            inspect(self._args())

    def test_rejects_duplicate_archive_member(self) -> None:
        self._write_archive(duplicate_member=True)
        with self.assertRaisesRegex(ValueError, "duplicate members"):
            inspect(self._args())

    def test_rejects_corrupt_archive(self) -> None:
        self.archive.write_bytes(b"not a zip")
        with self.assertRaises(zipfile.BadZipFile):
            inspect(self._args())


if __name__ == "__main__":
    unittest.main()
