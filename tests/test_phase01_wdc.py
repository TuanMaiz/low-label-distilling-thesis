import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from data.er_dataset_loader import WDCProductsConfig, load_wdc_products_pairwise
from data.serialize_pairs import serialize_pair


def _row(idx: int, label: int) -> dict:
    cluster_left = 100 if label else 100 + idx
    cluster_right = cluster_left if label else 900 + idx
    return {
        "id_left": idx * 10,
        "brand_left": "Acme",
        "title_left": f"Acme Camera Model {idx}",
        "description_left": "Compact digital camera",
        "price_left": "99.00",
        "priceCurrency_left": "USD",
        "cluster_id_left": cluster_left,
        "id_right": idx * 10 + 1,
        "brand_right": "Acme",
        "title_right": f"Acme Camera Model {idx}",
        "description_right": "Digital camera compact",
        "price_right": "100.00",
        "priceCurrency_right": "USD",
        "cluster_id_right": cluster_right,
        "pair_id": f"{idx * 10}#{idx * 10 + 1}",
        "label": label,
        "is_hard_negative": False,
    }


def _write_json_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_wdc_zip(tmp_path: Path) -> Path:
    extracted = tmp_path / "wdc"
    extracted.mkdir()
    rows = [_row(i, i % 2) for i in range(1, 17)]
    files = {
        "wdcproducts80cc20rnd000un_train_small.json.gz": rows,
        "wdcproducts80cc20rnd000un_valid_small.json.gz": rows[:4],
        "wdcproducts80cc20rnd100un_gs.json.gz": rows[:4],
    }
    for name, file_rows in files.items():
        _write_json_gz(extracted / name, file_rows)

    zip_path = tmp_path / "80pair.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name in files:
            archive.write(extracted / name, arcname=name)
    return zip_path


class Phase01WDCTest(unittest.TestCase):
    def test_load_wdc_products_from_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = _make_wdc_zip(Path(tmp))

            splits = load_wdc_products_pairwise(
                zip_path,
                config=WDCProductsConfig(corner_cases=80, train_size="small", test_unseen=100),
            )

            self.assertEqual(set(splits), {"train", "validation", "test"})
            self.assertEqual(len(splits["train"]), 16)
            self.assertTrue(splits["train"][0].record_a.attributes["title"].startswith("Acme"))
            self.assertEqual(splits["train"][0].split, "train")

    def test_serialization_preserves_field_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = _make_wdc_zip(Path(tmp))
            pair = load_wdc_products_pairwise(zip_path)["train"][0]

            text = serialize_pair(pair)

            self.assertIn("Record A:", text)
            self.assertIn("- title:", text)
            self.assertIn("- brand:", text)
            self.assertIn("same real-world entity", text)


if __name__ == "__main__":
    unittest.main()
