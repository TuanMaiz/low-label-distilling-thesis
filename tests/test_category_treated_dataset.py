"""Category-rule treatment (plan 261023): rule, action rule, builder and verifier."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from supervision.build_category_treated_dataset import (
    action_for, build, check_source_path, load_diagnoses, render, treat, verdict_for, verify)


def record(title_a, title_b, price_a='10', price_b='10'):
    return ('Task: decide whether Record A and Record B refer to the same real-world entity.\n\n'
            f'Record A:\n- title: {title_a}\n- brand: <missing>\n- description: <missing>\n'
            f'- price: {price_a}\n- priceCurrency: EUR\n\n'
            f'Record B:\n- title: {title_b}\n- brand: <missing>\n- description: <missing>\n'
            f'- price: {price_b}\n- priceCurrency: EUR')


def case(category, a=('title', 'X1'), b=('title', 'X2')):
    return {'category_id': category, 'attribute': 'a', 'note_vi': 'n',
            'evidence': [{'side': 'A', 'field': a[0], 'quote': a[1]},
                         {'side': 'B', 'field': b[0], 'quote': b[1]}]}


class RuleTests(unittest.TestCase):
    def test_step_order(self):
        self.assertEqual(verdict_for([case('cross_record_conflict'), case('intra_record_contradiction')]), (0, 1))
        self.assertEqual(verdict_for([case('ambiguous_variant_bundle'), case('cross_record_conflict')]), (2, 2))
        self.assertEqual(verdict_for([case('ambiguous_role_scope'), case('one_sided_information')]), (0, 3))
        self.assertEqual(verdict_for([case('one_sided_information'), case('noisy_or_malformed_data'),
                                      case('representation_difference')]), (1, 6))

    def test_price_only_or_one_sided_conflict_is_not_a_conflict(self):
        price = case('cross_record_conflict', ('price', '10'), ('price', '50'))
        mixed = case('cross_record_conflict', ('title', 'X1'), ('price', '50'))
        self.assertEqual(verdict_for([price]), (1, 6))
        self.assertEqual(verdict_for([mixed, case('ambiguous_variant_bundle')]), (0, 3))

    def test_action_rule(self):
        self.assertEqual(action_for(0, 'match'), ('remove', ''))
        self.assertEqual(action_for(2, 'non_match'), ('upweight', 'non_match'))
        self.assertEqual(action_for(2, 'match'), ('flip', 'non_match'))
        self.assertEqual(action_for(1, 'non_match'), ('flip', 'match'))


class TreatTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{'pair_id': f'p{i}', 'input_text': record('X1 a', 'X2 b')} for i in range(5)]
        self.labels = ['match', 'non_match', 'match', 'non_match', 'match']
        self.selected = {'p0', 'p1', 'p2', 'p3'}
        self.diag = {'p0': [case('cross_record_conflict')], 'p1': [case('cross_record_conflict')],
                     'p2': [case('ambiguous_variant_bundle')]}
        self.invalid = {'p3'}

    def run_arm(self, arm):
        return treat(self.inputs, self.labels, self.selected, self.diag, self.invalid, arm, 'ds')

    def test_arm_a_unchanged(self):
        rows, decisions = self.run_arm('A')
        self.assertEqual([r['label'] for r in rows], [1, 0, 1, 0, 1])
        self.assertEqual({d['action'] for d in decisions}, {'keep'})

    def test_arm_b_removes_all_detected(self):
        rows, _ = self.run_arm('B')
        self.assertEqual([r['pair_id'] for r in rows], ['p4'])

    def test_arm_c_actions_and_duplicates(self):
        rows, decisions = self.run_arm('C')
        actions = {d['pair_id']: d['action'] for d in decisions}
        self.assertEqual(actions, {'p0': 'flip', 'p1': 'upweight', 'p2': 'remove', 'p3': 'keep', 'p4': 'keep'})
        self.assertEqual([r['pair_id'] for r in rows], ['p0', 'p1', 'p3', 'p4', 'p1#dup1'])
        self.assertEqual(rows[0]['label'], 0)
        dup, src = rows[-1], rows[1]
        self.assertEqual(dup['duplicate_of'], 'p1')
        self.assertEqual({k: v for k, v in dup.items() if k not in ('pair_id', 'duplicate_of')},
                         {k: v for k, v in src.items() if k != 'pair_id'})

    def test_rejects_diagnosis_outside_detected(self):
        self.diag['p4'] = [case('cross_record_conflict')]
        with self.assertRaises(ValueError):
            self.run_arm('C')


class SourceAndBuildTests(unittest.TestCase):
    def test_gold_paths_refused(self):
        for path in ['data/cache/x/serialized/train.jsonl', 'data/cache/x/full_label_targets/a.jsonl',
                     'data/x/gold.jsonl', 'data/x/validation/a.csv']:
            with self.assertRaises(ValueError):
                check_source_path(Path(path), 'fixture')

    def write_sources(self, root, quote='X1'):
        inputs = [{'pair_id': 'p0', 'input_text': record('X1 a', 'X2 b')},
                  {'pair_id': 'p1', 'input_text': record('Y1', 'Y2')}]
        (root / 'inputs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in inputs))
        (root / 'labels.csv').write_text('pair_id,result\np0,non_match\np1,match\n')
        (root / 'selected.json').write_text(json.dumps([inputs[0]]))
        with (root / 'diag.csv').open('w', newline='') as h:
            w = csv.DictWriter(h, fieldnames=['pair_id', 'status', 'evidence_json'])
            w.writeheader()
            w.writerow({'pair_id': 'p0', 'status': 'categorized',
                        'evidence_json': json.dumps([case('cross_record_conflict', ('title', quote))])})
        return dict(inputs_path=root / 'inputs.jsonl', majority_path=root / 'labels.csv',
                    selected_path=root / 'selected.json', diagnoses_path=root / 'diag.csv',
                    dataset_id='ds', expected_count=2)

    def test_ungrounded_quote_refused(self):
        with tempfile.TemporaryDirectory() as t:
            sources = self.write_sources(Path(t), quote='NOT IN RECORD')
            with self.assertRaises(ValueError):
                render('C', **sources)

    def test_build_verify_and_deterministic_rebuild(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            sources = self.write_sources(root)
            out = root / 'out'
            manifest = build('C', out, **sources)
            self.assertEqual(manifest['actions'], {'keep': 1, 'upweight': 1})
            self.assertEqual(manifest['train_rows'], 3)
            self.assertFalse(manifest['gold_accessed'])
            first = {p.name: p.read_bytes() for p in out.iterdir()}
            build('C', out, **sources)
            self.assertEqual(first, {p.name: p.read_bytes() for p in out.iterdir()})
            verify('C', out, **sources)
            (out / 'train.jsonl').write_bytes(first['train.jsonl'] + b'{}\n')
            with self.assertRaises(ValueError):
                verify('C', out, **sources)


if __name__ == '__main__':
    unittest.main()
