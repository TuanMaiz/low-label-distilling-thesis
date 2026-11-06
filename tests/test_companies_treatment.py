"""Offline tests for the Companies treatment pipeline (plan 261106): synthetic data, no API, no GPU."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from supervision import build_category_treated_dataset as core
from supervision import build_category_treated_dataset_companies as companies_builder
from supervision import run_evidence_check_companies as ev

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ev.FIELDS


def record(side, **values):
    return f"Record {side}:\n" + "\n".join(f"- {f}: {values.get(f, '<missing>')}" for f in FIELDS)


def pair_text(a, b):
    return "Task: decide whether Record A and Record B refer to the same real-world entity.\n\n" + record("A", **a) + "\n\n" + record("B", **b)


PAIRS = [  # pair_id, record A, record B, teacher votes, student prediction
    ("madi_companies:train:dbpedia:acme:Acme", {"name": "Acme", "founded": "1950", "revenue": "10"},
     {"name": "Acme Inc", "founded": "1950", "revenue": "12"}, ("match", "match", "match"), "non_match"),
    ("madi_companies:train:fullcontact:beta:b1", {"name": "Beta Group", "founded": "1901"},
     {"name": "Beta Insurance", "founded": "1988"}, ("match", "match", "non_match"), "match"),
    ("madi_companies:train:fullcontact:gamma:g1", {"name": "Gamma"}, {"name": "Gamma Labs"},
     ("non_match", "non_match", "non_match"), "match"),
    ("madi_companies:train:dbpedia:delta:Delta", {"name": "Delta"}, {"name": "Delta"},
     ("match", "match", "match"), "match"),
]


def case(category, *evidence):
    return {"category_id": category, "attribute": "x", "note_vi": "ghi chú",
            "evidence": [{"side": s, "field": f, "quote": q} for s, f, q in evidence]}


DIAGNOSES = {  # selected pairs only
    PAIRS[0][0]: [case("cross_record_conflict", ("A", "revenue", "10"), ("B", "revenue", "12"))],   # supporting only
    PAIRS[1][0]: [case("cross_record_conflict", ("A", "founded", "1901"), ("B", "founded", "1988"))],
    PAIRS[2][0]: [case("ambiguous_variant_bundle", ("A", "name", "Gamma"), ("B", "name", "Gamma Labs"))],
}


class CompaniesTreatmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        d = self.dir = Path(self._tmp.name)
        self.inputs = d / "inputs.jsonl"
        self.inputs.write_text("".join(json.dumps({"pair_id": p, "input_text": pair_text(a, b)}) + "\n"
                                       for p, a, b, _, _ in PAIRS), encoding="utf-8")
        majority = [max(set(v), key=v.count) for _, _, _, v, _ in PAIRS]
        self.majority = d / "llm_unrefined.csv"
        self.majority.write_text("pair_id,result\n" + "".join(f"{p[0]},{m}\n" for p, m in zip(PAIRS, majority)), encoding="utf-8")
        self.consistency = d / "consistency.csv"
        with self.consistency.open("w", encoding="utf-8", newline="") as h:
            w = csv.writer(h, lineterminator="\n")
            w.writerow(["pair_id", "pass_01", "pass_02", "pass_03", "majority_label", "majority_votes", "consistency"])
            for (pid, _, _, votes, _), m in zip(PAIRS, majority):
                n = votes.count(m)
                w.writerow([pid, *votes, m, n, f"{n}/3"])
        rows = []
        for i, ((pid, _, _, _, pred), m) in enumerate(zip(PAIRS, majority)):
            pm = 0.9 if pred == "match" else 0.1
            rows.append([i, pid, 1, "llm_majority", m, int(m == "match"), pred, int(pred == "match"),
                         1 - pm, pm, pm if m == "match" else 1 - pm])
        buf = io.StringIO(newline="")
        w = csv.writer(buf, lineterminator="\n")
        from supervision.build_confident_learning import OOF_HEADER
        w.writerow(OOF_HEADER)
        w.writerows(rows)
        self.oof = d / "oof_predictions.csv"
        self.oof.write_bytes(buf.getvalue().encode("utf-8"))
        self.oof_summary = d / "oof_summary.json"
        self.oof_summary.write_text(json.dumps({
            "artifact_type": "wdc_qwen_oof_predictions", "row_count": len(PAIRS), "dataset_id": "madi_companies",
            "student_id": "qwen3-reranker-0-6b", "class_mapping": {"non_match": 0, "match": 1},
            "probability_column_order": ["non_match_probability", "match_probability"],
            "hidden_gold_accessed": False, "official_validation_accessed": False, "official_test_accessed": False,
            "output": {"sha256": hashlib.sha256(self.oof.read_bytes()).hexdigest()}}), encoding="utf-8")
        self.out = d / "evidence"

    def prepare(self):
        return ev.prepare(self.inputs, self.consistency, self.oof, self.oof_summary, ev.SETTINGS, self.out,
                          expected_count=len(PAIRS))

    def test_selection_is_two_of_three_or_disagreement(self) -> None:
        items, _ = self.prepare()
        self.assertEqual([i["pair_id"] for i in items], [p[0] for p in PAIRS[:3]])
        counts = json.loads((self.out / "contract.json").read_text())["counts"]
        self.assertEqual(counts, {"two_of_three": 1, "disagreement": 2, "overlap": 0, "union": 3})

    def test_payload_uses_flex_sends_records_only(self) -> None:
        items, config = self.prepare()
        payload = ev.payload_for(items[0], config, "prompt")
        self.assertEqual(payload["service_tier"], "flex")
        self.assertEqual(payload["provider"]["max_price"], {"prompt": 2.0, "completion": 10.0})
        user = payload["messages"][1]["content"]
        self.assertTrue(user.startswith("Record A:"))
        self.assertNotIn(items[0]["pair_id"], json.dumps(payload["messages"]))

    def test_run_refuses_draft_prompt(self) -> None:
        items, config = self.prepare()
        self.assertFalse(ev.prompt_signed_off(ev.PROMPT.read_text(encoding="utf-8")))
        with self.assertRaisesRegex(RuntimeError, "DRAFT"):
            ev.run(items, config, self.out, ev.PROMPT.read_text(encoding="utf-8"), client=None)

    def test_ungrounded_quote_rejected(self) -> None:
        items, _ = self.prepare()
        bad = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "status": "categorized", "summary_vi": "s", "uncertainties_vi": [],
            "cases": [case("one_sided_information", ("A", "name", "Not in record"))]})}}]}
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            ev.validate_diagnosis(bad, items[0])

    def write_diagnoses(self, items):
        path = self.dir / "diagnoses.csv"
        with path.open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=ev.DIAGNOSIS_HEADER, lineterminator="\n")
            w.writeheader()
            for it in items:
                w.writerow({k: "" for k in ev.DIAGNOSIS_HEADER} | {
                    "pair_id": it["pair_id"], "status": "categorized", "evidence_json": json.dumps(DIAGNOSES[it["pair_id"]])})
        return path

    def test_category_rule_on_companies(self) -> None:
        items, _ = self.prepare()
        diag = self.write_diagnoses(items)
        src = companies_builder.sources(inputs=self.inputs, majority=self.majority,
                                        selected=self.out / "selected_pairs.json", diagnoses=diag)
        with mock.patch.dict(src, expected_count=len(PAIRS)):
            blobs, manifest = core.render("C", **src)
        decisions = {r["pair_id"]: r for r in csv.DictReader(io.StringIO(blobs["decisions.csv"].decode()))}
        # revenue differs only: supporting info, not a conflict -> match, agrees with teacher -> upweight
        self.assertEqual((decisions[PAIRS[0][0]]["step"], decisions[PAIRS[0][0]]["action"]), ("6", "upweight"))
        # founding year conflict -> non-match, teacher said match -> flip
        self.assertEqual((decisions[PAIRS[1][0]]["step"], decisions[PAIRS[1][0]]["action"]), ("2", "flip"))
        # related-entity ambiguity -> not decidable -> remove
        self.assertEqual((decisions[PAIRS[2][0]]["step"], decisions[PAIRS[2][0]]["action"]), ("3", "remove"))
        self.assertEqual(decisions[PAIRS[3][0]]["action"], "keep")
        self.assertEqual(manifest["contract"], companies_builder.CONTRACT)
        self.assertFalse(manifest["gold_accessed"])
        blobs_b, manifest_b = core.render("B", **{**src, "expected_count": len(PAIRS)})
        self.assertEqual(manifest_b["train_rows"], 1)   # only the undetected pair is kept

    def test_arm_runner_routes_b_and_c(self) -> None:
        env = {**os.environ, "PYTHON": sys.executable}
        for arm in ("B", "C"):
            out = subprocess.run(["bash", "scripts/run_madi_companies_arms.sh", "plan", arm], cwd=ROOT, env=env,
                                 capture_output=True, text=True, check=True).stdout
            self.assertIn(f"refinement/category-rule-{arm}/train.jsonl", out)
            self.assertIn(f"madi_companies_arm_{arm}", out)

    def test_oof_runner_plan_and_guards(self) -> None:
        env = {**os.environ, "PYTHON": sys.executable}
        plan = subprocess.run(["bash", "scripts/run_madi_companies_oof.sh", "plan"], cwd=ROOT, env=env,
                              capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(len(plan), 6)
        for n, line in zip((1, 1, 2, 2, 3, 3), plan):
            self.assertIn(f"oof/majority-3fold/fold_0{n}/", line)
            self.assertIn("--precision fp16", line)
        refused = subprocess.run(["bash", "scripts/run_madi_companies_oof.sh", "train-fold", "1"], cwd=ROOT, env=env,
                                 capture_output=True, text=True)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("--confirm-oof-training", refused.stderr)


if __name__ == "__main__":
    unittest.main()
