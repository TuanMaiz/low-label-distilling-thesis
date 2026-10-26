from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from supervision.run_evidence_check import (
    Diagnosis, collect, dump, parse_fields, payload_for, prepare, report, run, select_pairs, validate_diagnosis,
)

TEXT = ('Task: decide whether records match.\n\nRecord A:\n- title: SSD 2TB\n- brand: Acme\n'
        '- description: SSD 1TB\n- price: 2E2\n- priceCurrency: USD\n\nRecord B:\n'
        '- title: SSD 2TB\n- brand: <missing>\n- description: SSD 2TB\n- price: 210\n- priceCurrency: USD')
CONFIG = {'settings': {'sol_high': {'model': 'fixture', 'reasoning': {'effort': 'high', 'exclude': True}}},
          'provider_routing': {'only': ['openai'], 'allow_fallbacks': False},
          'pricing_snapshot': {'input_usd_per_million_tokens': 2, 'output_usd_per_million_tokens': 10}}


def item(pid='p0'):
    return {'pair_id': pid, 'input_text': TEXT, 'two_of_three': True, 'disagreement': True,
            'oof': {'given_label': 'match', 'predicted_label': 'non_match', 'match_probability': '0.1'},
            'teacher': {'consistency': '2/3'}}


def raw(cases=None):
    diag = {'status': 'categorized' if cases else 'no_finding', 'cases': cases or [],
            'summary_vi': 'Nhận xét | không kết luận nhãn.', 'uncertainties_vi': []}
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(diag)}}],
            'usage': {'cost': 0.01, 'prompt_tokens': 100, 'completion_tokens': 100}}


def case(category='intra_record_contradiction'):
    return {'category_id': category, 'attribute': 'capacity', 'note_vi': 'Title và description khác dung lượng.',
            'evidence': [{'side': 'A', 'field': 'title', 'quote': '2TB'},
                         {'side': 'A', 'field': 'description', 'quote': '1TB'}]}


class EvidenceCheckTests(unittest.TestCase):
    def test_payload_hides_labels_ids_and_selection(self):
        payload = payload_for(item(), CONFIG, 'definitions')
        self.assertEqual(payload['messages'][1]['content'], 'Record A:' + TEXT.split('Record A:', 1)[1])
        self.assertEqual(set(payload), {'model', 'messages', 'reasoning', 'provider', 'max_tokens', 'response_format'})
        self.assertNotIn('given_label', json.dumps(payload))
        self.assertNotIn('two_of_three', json.dumps(payload))
        self.assertEqual(payload['max_tokens'], 8192)

    def test_union_selection_alignment_and_overlap(self):
        import csv
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'inputs').write_text('\n'.join(json.dumps({'pair_id': f'p{i}', 'input_text': TEXT}) for i in range(4)))
            with (root / 'consistency').open('w', newline='') as handle:
                w = csv.DictWriter(handle, fieldnames=['pair_id', 'pass_01', 'pass_02', 'pass_03', 'majority_label', 'majority_votes', 'consistency'])
                w.writeheader()
                for i in range(4):
                    w.writerow({'pair_id': f'p{i}', 'pass_01': 'match', 'pass_02': 'match',
                                'pass_03': 'non_match' if i in (0, 1) else 'match', 'majority_label': 'match',
                                'majority_votes': 2 if i in (0, 1) else 3, 'consistency': '2/3' if i in (0, 1) else '3/3'})
            oof = [{'pair_id': f'p{i}', 'given_label': 'match', 'predicted_label': 'non_match' if i in (1, 2) else 'match'} for i in range(4)]
            with patch('supervision.run_evidence_check._load_oof', return_value=(oof, None, None, None, None)):
                chosen, counts = select_pairs(root / 'inputs', root / 'consistency', root / 'oof', root / 'summary', 4)
                self.assertEqual([r['pair_id'] for r in chosen], ['p0', 'p1', 'p2'])
                self.assertEqual(counts, {'two_of_three': 2, 'disagreement': 2, 'overlap': 1, 'union': 3})
                oof[0]['given_label'] = 'non_match'
                with self.assertRaisesRegex(ValueError, 'majority mismatch'):
                    select_pairs(root / 'inputs', root / 'consistency', root / 'oof', root / 'summary', 4)

    def test_valid_grounded_diagnosis_and_rejects_invented_quote(self):
        self.assertEqual(validate_diagnosis(raw([case()]), item()).status, 'categorized')
        value = case()
        value['evidence'][0]['quote'] = '4TB'
        with self.assertRaisesRegex(ValueError, 'ungrounded'):
            validate_diagnosis(raw([value]), item())

    def test_requires_both_sides_for_conflict_and_two_quotes_for_internal(self):
        with self.assertRaisesRegex(ValueError, 'both sides'):
            validate_diagnosis(raw([case('cross_record_conflict')]), item())
        value = case()
        value['evidence'] = value['evidence'][:1]
        with self.assertRaisesRegex(ValueError, 'two quotes'):
            validate_diagnosis(raw([value]), item())

    def test_unknown_categories_status_refusal_and_truncation_rejected(self):
        with self.assertRaises(ValueError):
            validate_diagnosis(raw([case('other')]), item())
        value = raw()
        value['choices'][0]['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            validate_diagnosis(value, item())
        with self.assertRaises(ValueError):
            Diagnosis.model_validate({'status': 'unknown', 'cases': [], 'summary_vi': 'x', 'uncertainties_vi': [], 'label': 'match'})

    def test_resume_preserves_malformed_costs_and_readable_files(self):
        class Client:
            calls = 0
            def create(self, payload):
                self.calls += 1
                response = raw()
                if self.calls == 2:
                    response['choices'][0]['message']['content'] = 'malformed'
                return response
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            client = Client()
            rows = [item('p0'), item('p1')]
            summary = run(rows, CONFIG, output, 'prompt', client, workers=1)
            self.assertEqual(summary['valid_count'], 1)
            self.assertEqual(summary['statuses']['invalid'], 1)
            self.assertAlmostEqual(summary['received_response_cost_usd'], .02)
            run(rows, CONFIG, output, 'prompt', client)
            self.assertEqual(client.calls, 2)
            self.assertIn('\\|', (output / 'report.md').read_text())
            self.assertTrue((output / 'diagnoses.csv').exists())

    def test_unresolved_dispatch_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            dump(output / 'requests/0000.json', {'pair_id': 'p0'})
            with self.assertRaisesRegex(RuntimeError, 'unresolved'):
                run([item()], CONFIG, output, 'prompt', object())

    def test_continue_never_attempted_pairs_without_retrying_unresolved(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            dump(output / 'requests/0000.json', {'pair_id': 'p0'})
            class Client:
                calls = 0
                def create(self, payload):
                    self.calls += 1
                    return raw()
            client = Client()
            summary = run([item('p0'), item('p1')], CONFIG, output, 'prompt', client,
                          workers=1, skip_unresolved=True)
            self.assertEqual(client.calls, 1)
            self.assertEqual(summary['statuses']['unresolved'], 1)
            self.assertEqual(summary['valid_count'], 1)
            self.assertFalse(summary['cost_complete'])
            self.assertFalse((output / 'responses/0000.json').exists())

    def test_missing_cost_is_unknown_not_zero_measured(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            value = raw()
            del value['usage']
            dump(output / 'responses/0000.json', value)
            summary = report([item()], output, CONFIG)
            self.assertFalse(summary['cost_complete'])
            self.assertEqual(summary['cost_sources']['unknown'], 1)

    def test_changed_contract_rejected(self):
        import argparse
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / str(i) for i in range(4)]
            for path in paths:
                path.write_text('fixture')
            settings = root / 'settings'
            settings.write_text(json.dumps({**CONFIG, 'api_url': 'https://openrouter.ai/api/v1'}))
            args = argparse.Namespace(inputs=paths[0], consistency=paths[1], oof=paths[2], oof_summary=paths[3],
                                      expected_count=1, settings=settings, output_dir=root / 'output')
            with patch('supervision.run_evidence_check.select_pairs', return_value=([item()], {'union': 1})):
                prepare(args)
                paths[0].write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    prepare(args)

    def test_bad_usage_and_unresolved_dispatch_report_unknown_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            value = raw()
            value['usage']['cost'] = 'invalid'
            dump(output / 'responses/0000.json', value)
            dump(output / 'requests/0001.json', {'pair_id': 'p1'})
            summary = report([item('p0'), item('p1')], output, CONFIG)
            self.assertEqual(summary['valid_count'], 1)
            self.assertFalse(summary['cost_complete'])
            self.assertEqual(summary['unresolved_request_ids'], ['p1'])

    def test_malformed_envelope_retains_cost_and_bad_usage_is_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            value = raw()
            value['choices'] = [None]
            dump(output / 'responses/0000.json', value)
            second = raw()
            second['usage'] = ['invalid']
            dump(output / 'responses/0001.json', second)
            summary = report([item('p0'), item('p1')], output, CONFIG)
            self.assertEqual(summary['statuses']['invalid'], 1)
            self.assertEqual(summary['valid_count'], 1)
            self.assertAlmostEqual(summary['received_response_cost_usd'], .01)
            self.assertEqual(summary['cost_sources']['unknown'], 1)
            self.assertFalse(summary['cost_complete'])

    def test_retry_preserves_original_and_charges_all_received_attempts(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            bad = raw()
            bad['choices'][0]['message']['content'] = 'invalid JSON'
            dump(output / 'responses/0000.json', bad)
            before = (output / 'responses/0000.json').read_bytes()
            dump(output / 'retries/invalid-01/responses/0000.json', raw([case()]))
            dump(output / 'retries/invalid-02/responses/0000.json', bad)
            summary = report([item()], output, CONFIG)
            self.assertEqual(summary['valid_count'], 1)
            self.assertEqual(summary['failures'], {})
            self.assertAlmostEqual(summary['received_response_cost_usd'], .03)
            self.assertEqual(summary['received_response_count'], 3)
            self.assertEqual((output / 'responses/0000.json').read_bytes(), before)

    def test_missing_retry_response_is_not_complete_cost_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            dump(output / 'responses/0000.json', raw())
            dump(output / 'retries/invalid-01/requests/0000.json', {'pair_id': 'p0'})
            summary = report([item()], output, CONFIG)
            self.assertFalse(summary['cost_complete'])
            self.assertEqual(summary['unresolved_retry_requests'][0]['pair_id'], 'p0')


if __name__ == '__main__':
    unittest.main()
