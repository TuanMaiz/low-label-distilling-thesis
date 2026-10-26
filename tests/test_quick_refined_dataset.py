"""Quick-tier policy and offline dataset publication tests."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supervision.build_quick_refined_dataset import build, refine
from supervision.run_evidence_check import Diagnosis


def diagnosis(categories, price_only=False):
    cases = []
    for category in categories:
        cases.append({'category_id': category, 'attribute': 'identity', 'note_vi': 'fixture',
                      'evidence': [{'side': 'A', 'field': 'price' if price_only else 'title', 'quote': 'A'},
                                   {'side': 'B', 'field': 'price' if price_only else 'title', 'quote': 'B'}]})
    return Diagnosis.model_validate({'status': 'categorized', 'cases': cases,
                                    'summary_vi': 'fixture', 'uncertainties_vi': []})


class QuickRefinementTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{'pair_id': 'p0', 'input_text': 'record text 0'},
                       {'pair_id': 'p1', 'input_text': 'record text 1'}]
        self.labels = ['match', 'non_match']

    def run_policy(self, diagnoses, ledger=(), selected=None):
        return refine(self.inputs, self.labels, self.inputs if selected is None else selected,
                      diagnoses, list(ledger), 'fixture-dataset')

    def test_prune_stops_before_lower_tiers_and_retains_other_pair(self):
        diag = diagnosis(['representation_difference', 'cross_record_conflict', 'intra_record_contradiction'])
        train, audit = self.run_policy({'p0': diag}, [{'pair_id': 'p0', 'decision': 'non_match', 'reason': 'conflict'}])
        self.assertEqual([r['pair_id'] for r in train], ['p1'])
        self.assertEqual(audit[0]['action'], 'prune')
        self.assertEqual(audit[0]['final_label'], '')
        self.assertTrue(audit[0]['ordered_cases'].startswith('intra_record_contradiction;cross_record_conflict'))

    def test_reviewed_conflict_changes_label_but_not_text(self):
        train, audit = self.run_policy({'p0': diagnosis(['cross_record_conflict', 'representation_difference'])},
                                      [{'pair_id': 'p0', 'decision': 'non_match', 'reason': 'different capacity'}])
        self.assertEqual(train[0]['input_text'], self.inputs[0]['input_text'])
        self.assertEqual((train[0]['label'], train[0]['target_text']), (0, 'non-match'))
        self.assertEqual(audit[0]['action'], 'relabel')

    def test_unreviewed_conflict_and_invalid_diagnosis_do_not_change_labels(self):
        train, audit = self.run_policy({'p0': diagnosis(['cross_record_conflict'])})
        self.assertEqual([r['label'] for r in train], [1, 0])
        self.assertEqual([r['action'] for r in audit], ['keep', 'keep'])
        self.assertFalse(audit[1]['diagnosis_valid'])

    def test_safe_match_changes_only_explicitly_reviewed_pair(self):
        train, audit = self.run_policy({'p1': diagnosis(['representation_difference', 'one_sided_information'])},
                                      [{'pair_id': 'p1', 'decision': 'match', 'reason': 'same complete model'}])
        self.assertEqual(train[1]['label'], 1)
        self.assertEqual(audit[1]['action'], 'relabel')

    def test_unresolved_higher_tier_blocks_match(self):
        for category in ('cross_record_conflict', 'ambiguous_variant_bundle', 'ambiguous_role_scope'):
            with self.subTest(category=category), self.assertRaises(ValueError):
                self.run_policy({'p1': diagnosis(['representation_difference', category])},
                                [{'pair_id': 'p1', 'decision': 'match', 'reason': 'unsafe'}])

    def test_price_only_cannot_force_non_match(self):
        with self.assertRaises(ValueError):
            self.run_policy({'p0': diagnosis(['cross_record_conflict'], price_only=True)},
                            [{'pair_id': 'p0', 'decision': 'non_match', 'reason': 'price differs'}])

    def test_unknown_duplicate_or_invalid_review_rejected(self):
        d = {'pair_id': 'p0', 'decision': 'keep', 'reason': 'unclear'}
        for ledger in ([d, d], [{**d, 'pair_id': 'unknown'}], [{**d, 'decision': 'prune'}]):
            with self.subTest(ledger=ledger), self.assertRaises(ValueError):
                self.run_policy({'p0': diagnosis(['representation_difference'])}, ledger)
        with self.assertRaises(ValueError):
            self.run_policy({}, [d])

    def test_selected_text_or_labels_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.run_policy({}, selected=[{**self.inputs[0], 'input_text': 'tampered'}])
        with self.assertRaises(ValueError):
            refine(self.inputs, ['match'], self.inputs, {}, [], 'fixture')

    def test_unanalysed_rows_remain_present(self):
        train, audit = self.run_policy({}, selected=[])
        self.assertEqual(len(train), 2)
        self.assertTrue(all(not r['selected'] for r in audit))

    def test_publication_rerun_and_tampered_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, labels, review = root/'inputs.jsonl', root/'labels.csv', root/'review.json'
            inputs.write_text(''.join(json.dumps(r)+'\n' for r in self.inputs))
            labels.write_text('pair_id,result\np0,match\np1,non_match\n')
            review.write_text('[]')
            evidence = root/'evidence'
            evidence.mkdir()
            (evidence/'contract.json').write_text(json.dumps({'sources': {}, 'config': {}}))
            (evidence/'selected_pairs.json').write_text(json.dumps(
                [{**self.inputs[0], 'teacher': {'majority_label': 'match'}}]))
            output = root/'output'
            args = (inputs, labels, evidence, review, output, 'fixture', 2)
            with patch('supervision.build_quick_refined_dataset.collect', return_value=(
                    {'p0': diagnosis(['intra_record_contradiction'])}, {}, 0, {})):
                summary = build(*args)
                first = {p.name: p.read_bytes() for p in output.iterdir()}
                self.assertEqual(build(*args), summary)
                self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, first)
                self.assertEqual(summary['retained_count'], 1)
                with (output/'decisions.csv').open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 2)
                with (output/'labels.csv').open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 1)
                self.assertEqual(inputs.read_text(), ''.join(json.dumps(r)+'\n' for r in self.inputs))
                (output/'train.jsonl').write_text('tampered')
                with self.assertRaises(ValueError):
                    build(*args)


if __name__ == '__main__':
    unittest.main()
