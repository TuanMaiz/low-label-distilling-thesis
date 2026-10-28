"""MaDI-Bench Companies preparation on a small synthetic fixture."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from data.prepare_madi_companies import load_profile, prepare, materialize_test, render, PROFILE

FORBES = ('forbes_url,company,url,region,business_segment,asset_value,sales_figure\n'
          'http://www.forbes.com/companies/acme/,Acme,http://www.forbes.com/companies/acme/,US,Tools,10,5\n'
          'http://www.forbes.com/companies/beta/,Beta,http://www.forbes.com/companies/beta/,,Banks,20,7\n')
DBPEDIA = ('entity_uri,org_name,established,nation,headquarters,sector,keypeople_name,total_assets_val,annual_income\n'
           'http://dbpedia.org/resource/Acme_Corp,Acme Corp,1900-01-01,United States,Springfield,Tools,,,\n'
           'http://dbpedia.org/resource/Beta_Bank,Beta Bank,,Germany,,Banks,,,\n')
FULLCONTACT = ('Attribute_1,Attribute_2,Attribute_3,Attribute_4,Attribute_5,Attribute_6\n'
               'fullcontact_1,Acme Inc,United States,null,Jane Doe,1900-01-01\n'
               'fullcontact_2,Beta AG,Germany,Berlin,,\n')
MAPPING = {'mappings': [
    *({'source_dataset': 'forbes', 'source_column': c, 'target_column': t, 'label': True} for c, t in
      [('forbes_url', 'id'), ('company', 'name'), ('region', 'country'), ('business_segment', 'industry'),
       ('asset_value', 'assets'), ('sales_figure', 'revenue')]),
    *({'source_dataset': 'dbpedia', 'source_column': c, 'target_column': t, 'label': True} for c, t in
      [('entity_uri', 'id'), ('org_name', 'name'), ('established', 'founded'), ('nation', 'country'),
       ('headquarters', 'city'), ('sector', 'industry'), ('keypeople_name', 'keypeople'),
       ('total_assets_val', 'assets'), ('annual_income', 'revenue')]),
    *({'source_dataset': 'fullcontact', 'source_column': c, 'target_column': t, 'label': True} for c, t in
      [('Attribute_1', 'id'), ('Attribute_2', 'name'), ('Attribute_3', 'country'), ('Attribute_4', 'city'),
       ('Attribute_5', 'keypeople'), ('Attribute_6', 'founded')]),
    {'source_dataset': 'forbes', 'source_column': 'url', 'target_column': 'id', 'label': False}]}
A, B = 'http://www.forbes.com/companies/acme/', 'http://www.forbes.com/companies/beta/'
DA, DB, DX = 'http://dbpedia.org/resource/Acme_Corp', 'http://dbpedia.org/resource/Beta_Bank', 'http://dbpedia.org/resource/Gone'
PAIRS = {
    'train': {'dbpedia': f'{A},{DA},true\n{B},{DA},False\n{B},{DX},true\n', 'fullcontact': f'{A},fullcontact_1,TRUE\n'},
    'validation': {'dbpedia': f'{B},{DB},true\n', 'fullcontact': f'{B},fullcontact_2,true\n{A},fullcontact_2,false\n'},
    'test': {'dbpedia': f'{A},{DB},false\n', 'fullcontact': f'{B},fullcontact_1,false\n'},
}
COUNTS = {'train': {'dbpedia': (3, 2, 1), 'fullcontact': (1, 1, 0), 'pooled': (3, 2, 1)},
          'validation': {'dbpedia': (1, 1, 0), 'fullcontact': (2, 1, 1), 'pooled': (3, 2, 1)},
          'test': {'dbpedia': (1, 0, 1), 'fullcontact': (1, 0, 1), 'pooled': (2, 0, 2)}}


def fixture(tmp: Path):
    root = tmp / 'raw'
    files = {'data/forbes.csv': FORBES, 'data/dbpedia.csv': DBPEDIA, 'data/fullcontact.csv': FULLCONTACT,
             'schemamatching/sm_mapping_gold.json': json.dumps(MAPPING)}
    for split, by in PAIRS.items():
        for src, text in by.items():
            files[f'entitymatching/{split}_{src}.csv'] = text
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding='utf-8')
    meta = lambda name: {'path': name, 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest(),
                         'size_bytes': (root / name).stat().st_size}
    profile = copy.deepcopy(load_profile(PROFILE))
    for table, rows in (('forbes', 2), ('dbpedia', 2), ('fullcontact', 2)):
        profile['record_tables'][table].update(meta(f'data/{table}.csv'), row_count=rows)
    profile['record_tables']['fullcontact']['literal_null_columns'] = {'Attribute_4': 1}
    profile['schema_mapping'] = {'path': 'schemamatching/sm_mapping_gold.json',
                                 'sha256': hashlib.sha256((root / 'schemamatching/sm_mapping_gold.json').read_bytes()).hexdigest()}
    for split in PAIRS:
        for src in ('dbpedia', 'fullcontact'):
            n, m, nm = COUNTS[split][src]
            profile['splits'][split]['files'][src] = {**meta(f'entitymatching/{split}_{src}.csv'),
                                                      'row_count': n, 'match_count': m, 'non_match_count': nm}
        n, m, nm = COUNTS[split]['pooled']
        profile['splits'][split].update(pooled_row_count=n, pooled_match_count=m, pooled_non_match_count=nm,
                                        dropped_unresolved_pairs=[[B, DX]] if split == 'train' else [])
    path = tmp / 'profile.json'
    path.write_text(json.dumps(profile), encoding='utf-8')
    return path, root


class PrepareTests(unittest.TestCase):
    def test_render_mapping_cleanup_pooling_and_drops(self):
        with tempfile.TemporaryDirectory() as t:
            profile, root = fixture(Path(t))
            blobs = render(profile, root)
            train = [json.loads(l) for l in blobs['serialized/train.jsonl'].decode().splitlines()]
            self.assertEqual([r['pair_id'] for r in train], [
                'madi_companies:train:dbpedia:acme:Acme_Corp', 'madi_companies:train:dbpedia:beta:Acme_Corp',
                'madi_companies:train:fullcontact:acme:fullcontact_1'])
            self.assertEqual([r['label'] for r in train], [1, 0, 1])
            text = train[2]['input_text']
            self.assertIn('- name: Acme Inc', text)
            self.assertIn('- city: <missing>', text)        # literal null cleaned
            self.assertNotIn('forbes.com', text)            # ids and url dropped
            self.assertIn('- country: <missing>', [r for r in train if ':beta:' in r['pair_id']][0]['input_text'])
            gold = [json.loads(l) for l in blobs['full_label_targets/gold.jsonl'].decode().splitlines()]
            self.assertEqual([g['target_text'] for g in gold], ['match', 'non-match', 'match'])
            inputs = [json.loads(l) for l in blobs['teacher_labels/inputs.jsonl'].decode().splitlines()]
            self.assertEqual(set(inputs[0]), {'pair_id', 'input_text'})
            self.assertNotIn('serialized/test.jsonl', blobs)

    def test_changed_source_or_unexpected_drop_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            profile, root = fixture(Path(t))
            (root / 'data/forbes.csv').write_text(FORBES + 'x\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                render(profile, root)
        with tempfile.TemporaryDirectory() as t:
            profile, root = fixture(Path(t))
            p = json.loads(profile.read_text()); p['splits']['train']['dropped_unresolved_pairs'] = []
            profile.write_text(json.dumps(p))
            with self.assertRaises(ValueError):
                render(profile, root)

    def test_prepare_is_deterministic_and_verify_detects_changes(self):
        with tempfile.TemporaryDirectory() as t:
            profile, root = fixture(Path(t))
            out = Path(t) / 'out'
            prepare(profile, root, out)
            first = {p.name: p.read_bytes() for p in out.rglob('*') if p.is_file()}
            prepare(profile, root, out, verify_only=True)
            prepare(profile, root, out)
            self.assertEqual(first, {p.name: p.read_bytes() for p in out.rglob('*') if p.is_file()})
            (out / 'serialized/train.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                prepare(profile, root, out, verify_only=True)

    def test_test_is_materialized_only_on_request_and_once(self):
        with tempfile.TemporaryDirectory() as t:
            profile, root = fixture(Path(t))
            out = Path(t) / 'out'
            prepare(profile, root, out)
            self.assertFalse((out / 'serialized-test').exists())
            info = materialize_test(profile, root, out)
            self.assertEqual(info['pairs'], 2)
            prepare(profile, root, out, verify_only=True)   # test files do not disturb verification
            with self.assertRaises(ValueError):
                materialize_test(profile, root, out)


if __name__ == '__main__':
    unittest.main()
