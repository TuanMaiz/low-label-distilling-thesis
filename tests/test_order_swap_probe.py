import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.order_swap_probe import (
    append_scores,
    build_pairs,
    fuse,
    load_scores,
    run_score,
    run_summarize,
    score_rows,
    sigmoid,
    summarize,
    swap_record_order,
)
from models.generative_reranker_student import (
    GenerativeRerankerModel,
    format_reranker_pair,
)
from models.student_config import load_student_config
from tests.test_generative_reranker import _FakeCausalLM, _FakeTokenizer

CONFIG = Path("configs/students/qwen3_reranker_0_6b.json")


def _text(title_a, title_b):
    return (
        "Task: decide whether Record A and Record B refer to the same real-world entity."
        f"\n\nRecord A:\n- title: {title_a}\n- brand: X"
        f"\n\nRecord B:\n- title: {title_b}\n- brand: Y"
    )


def _rows():
    return [
        {"pair_id": "p1", "label": 1, "input_text": _text("alpha", "alpha beta")},
        {"pair_id": "p2", "label": 0, "input_text": _text("gamma delta", "gamma")},
        {"pair_id": "p3", "label": 0, "input_text": _text("eps", "zeta")},
    ]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fake_model(config, tokenizer):
    return GenerativeRerankerModel(_FakeCausalLM([2.0, -1.0, 0.5]), 6, 7, config.label_to_id)


class SwapRecordOrderTest(unittest.TestCase):
    def test_bodies_exchange_and_header_stays(self):
        text = _text("alpha", "beta")
        swapped = swap_record_order(text)
        self.assertTrue(swapped.startswith("Task: decide"))
        self.assertIn("Record A:\n- title: beta\n- brand: Y", swapped)
        self.assertIn("Record B:\n- title: alpha\n- brand: X", swapped)
        self.assertEqual(swap_record_order(swapped), text)

    def test_swapped_record_b_becomes_query(self):
        prompt = format_reranker_pair(swap_record_order(_text("alpha", "beta")), "Same?")
        query, document = prompt.split("\n<Document>: ")
        self.assertIn("title: beta", query)
        self.assertIn("title: alpha", document)

    def test_missing_boundary_is_rejected(self):
        with self.assertRaises(ValueError):
            swap_record_order("Record A:\nonly one record")


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.config = load_student_config(CONFIG)

    def test_score_rows_records_log_odds_per_order(self):
        tokenizer = _FakeTokenizer()
        collected = []
        written = score_rows(
            _fake_model(self.config, tokenizer), tokenizer, self.config, _rows()[:1],
            "swapped", 4, "cpu", "fp32", 4096, collected.extend,
        )
        self.assertEqual(written, 1)
        record = collected[0]
        self.assertEqual(record["order"], "swapped")
        self.assertAlmostEqual(record["log_odds"], 2.0, places=5)
        self.assertAlmostEqual(record["match_probability"], sigmoid(2.0), places=6)
        self.assertIn("title: alpha beta", tokenizer.prompts[0].split("<Document>")[0])

    def test_run_score_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_jsonl(tmp / "pairs.jsonl", _rows())
            args = argparse.Namespace(
                config=str(CONFIG), model="base", input=str(tmp / "pairs.jsonl"),
                output_dir=str(tmp / "out"), orders=["original", "swapped"],
                pair_ids=None, limit=None, batch_size=2, max_input_length=4096,
                device="cpu", precision="fp32",
            )

            def fake_loader(config, spec):
                tokenizer = _FakeTokenizer()
                return tokenizer, _fake_model(config, tokenizer), {"kind": "fake"}

            with patch("analysis.order_swap_probe.load_scorer", side_effect=fake_loader):
                first = run_score(args)
                second = run_score(args)
            self.assertEqual(first["orders"]["original"]["scored_now"], 3)
            self.assertEqual(second["orders"]["swapped"]["scored_now"], 0)
            scores = load_scores(tmp / "out" / "scores.jsonl")
            self.assertEqual(len(scores), 6)
            run = json.loads((tmp / "out" / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(len(run["sessions"]), 2)

            args.limit = 1
            with patch("analysis.order_swap_probe.load_scorer", side_effect=fake_loader):
                with self.assertRaises(ValueError):
                    run_score(args)

    def test_partial_trailing_line_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.jsonl"
            append_scores(path, [{"pair_id": "p1", "order": "original"}])
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"pair_id": "p2", "ord')
            self.assertEqual(len(load_scores(path)), 1)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


class SummarizeTest(unittest.TestCase):
    def _scores(self):
        values = {
            # pair: (original, swapped) log-odds
            "p1": (3.0, 1.0),    # match; A ("alpha") is vaguer
            "p2": (2.0, -3.0),   # non-match predicted match originally; B is vaguer
            "p3": (-4.0, -4.5),  # non-match; tied title sizes
        }
        rows = []
        for pair_id, (original, swapped) in values.items():
            label = 1 if pair_id == "p1" else 0
            for order, log_odds in (("original", original), ("swapped", swapped)):
                rows.append({"pair_id": pair_id, "order": order, "label": label, "log_odds": log_odds})
        return rows

    def test_groups_direction_and_fusion(self):
        pairs, coverage = build_pairs(self._scores(), _rows())
        self.assertEqual(coverage["pairs_with_both_orders"], 3)
        summary = summarize(pairs, threshold=0.5, select_fused_thresholds=True)
        groups = summary["groups"]
        self.assertEqual((groups["TP"]["n"], groups["FP"]["n"], groups["TN"]["n"]), (1, 1, 1))
        self.assertEqual(groups["FP"]["decision_flips"], 1)
        self.assertEqual(groups["TP"]["decision_flips"], 0)
        by_id = {pair["pair_id"]: pair for pair in pairs}
        # Vaguer record as Query minus specific record as Query.
        self.assertAlmostEqual(by_id["p1"]["direction_delta"], 2.0)
        self.assertAlmostEqual(by_id["p2"]["direction_delta"], -5.0)
        self.assertIsNone(by_id["p3"]["direction_delta"])
        fused = summary["metrics_at_saved_threshold"]["mean"]
        self.assertEqual((fused["tp"], fused["fp"], fused["tn"], fused["fn"]), (1, 0, 2, 0))
        self.assertIn("fused_thresholds", summary)
        self.assertEqual(fuse(1.0, -3.0, "min"), -3.0)

    def test_reproduction_check_and_partial_guard(self):
        pairs, _ = build_pairs(self._scores(), _rows())
        saved = {
            "p1": {"match_probability": sigmoid(3.0), "prediction": 1},
            "p2": {"match_probability": 0.5, "prediction": 0},
            "p3": {"match_probability": sigmoid(-4.0), "prediction": 0},
        }
        check = summarize(pairs, 0.5, saved_predictions=saved)["reproduction_check"]
        self.assertEqual(check["decision_disagreement"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _write_jsonl(tmp / "pairs.jsonl", _rows())
            _write_jsonl(tmp / "scores.jsonl", self._scores()[:4])
            (tmp / "threshold.json").write_text('{"decision_threshold": 0.5}', encoding="utf-8")
            args = argparse.Namespace(
                scores=str(tmp / "scores.jsonl"), input=str(tmp / "pairs.jsonl"),
                threshold_json=str(tmp / "threshold.json"), threshold=None,
                saved_predictions=None,
                select_fused_thresholds=False, fused_thresholds_json=None,
                allow_partial=False, output=str(tmp / "summary.json"), pairs_csv=None,
            )
            with self.assertRaises(ValueError):
                run_summarize(args)
            args.allow_partial = True
            summary = run_summarize(args)
            self.assertEqual(summary["coverage"]["pairs_with_both_orders"], 2)
            args.threshold = 0.5
            with self.assertRaises(ValueError):
                run_summarize(args)
            args.threshold_json = None
            self.assertEqual(run_summarize(args)["saved_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
