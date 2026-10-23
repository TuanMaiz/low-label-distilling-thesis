"""Category-rule treatment of detected uncertain pairs (plan 261023 contract).

Applies the signed-off guideline steps to saved evidence-check diagnoses and
publishes one training bundle per arm. No API call, no gold labels, no training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from supervision.prepare_oof_folds import (_load_blinded_inputs, _load_majority_labels,
                                           _reject_forbidden_source_path)
from supervision.run_evidence_check import CATEGORY_NAMES, keyed, parse_fields

ROOT = Path(__file__).resolve().parents[1]
GUIDELINE = ROOT / 'supervision/judge_guideline.md'
PRICE_FIELDS = frozenset({'price', 'priceCurrency'})
LABELS = ('match', 'non_match')
VERDICT_LABEL = {1: 'match', 2: 'non_match'}
ARMS = ('A', 'B', 'C')
LABEL_SOURCE = {'A': 'llm_majority', 'B': 'llm_majority_detected_removed',
                'C': 'llm_majority_category_rule'}
DECISION_FIELDS = ['pair_id', 'detected', 'diagnosis_valid', 'categories', 'verdict', 'step',
                   'action', 'original_label', 'final_label', 'included_in_train', 'duplicated']


def check_source_path(path: Path, role: str) -> None:
    """Gold-free boundary: also refuse serialized splits, which carry benchmark labels."""
    _reject_forbidden_source_path(path, role)
    if 'serialized' in (part.lower() for part in path.resolve().parts):
        raise ValueError(f'{role} path is outside the gold-free boundary: {path}')


def has_real_conflict(case: dict) -> bool:
    """A conflict needs a non-price quote on both sides (price alone never decides)."""
    sides = {e['side'] for e in case['evidence'] if e['field'] not in PRICE_FIELDS}
    return case['category_id'] == 'cross_record_conflict' and sides == {'A', 'B'}


def verdict_for(cases: list[dict]) -> tuple[int, int]:
    """Contract section 2: (verdict, guideline step). 0 = not decidable, 1 = match, 2 = non-match."""
    categories = {c['category_id'] for c in cases}
    if 'intra_record_contradiction' in categories:
        return 0, 1
    if any(has_real_conflict(c) for c in cases):
        return 2, 2
    if categories & {'ambiguous_variant_bundle', 'ambiguous_role_scope'}:
        return 0, 3
    return 1, 6


def action_for(verdict: int, teacher: str) -> tuple[str, str]:
    """Contract section 3: (action, final label)."""
    if verdict == 0:
        return 'remove', ''
    label = VERDICT_LABEL[verdict]
    return ('upweight', label) if label == teacher else ('flip', label)


def load_diagnoses(path: Path, selected: list[dict]) -> tuple[dict[str, list[dict]], set[str]]:
    """Valid diagnoses keyed by pair; every evidence quote re-checked against the record."""
    check_source_path(path, 'diagnoses')
    by_id = keyed(selected)
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if {r['pair_id'] for r in rows} != set(by_id) or len(rows) != len(by_id):
        raise ValueError('diagnoses do not cover exactly the detected pairs')
    valid, invalid = {}, set()
    for row in rows:
        if row['status'] != 'categorized':
            invalid.add(row['pair_id'])
            continue
        cases = json.loads(row['evidence_json'])
        if not cases:
            raise ValueError(f'categorized diagnosis without cases: {row["pair_id"]}')
        fields = parse_fields(by_id[row['pair_id']]['input_text'])
        for case in cases:
            if case['category_id'] not in CATEGORY_NAMES or not case['evidence']:
                raise ValueError(f'invalid case in {row["pair_id"]}')
            for e in case['evidence']:
                value = fields[e['side']][e['field']]
                if value == '<missing>' or not e['quote'].strip() or e['quote'] not in value:
                    raise ValueError(f'ungrounded quote in {row["pair_id"]}: {e["quote"]!r}')
        valid[row['pair_id']] = cases
    return valid, invalid


def treat(inputs, labels, selected_ids, diagnoses, invalid, arm, dataset_id):
    """Training rows (originals in source order, then duplicates) and one decision per pair."""
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}')
    if len(inputs) != len(labels) or any(v not in LABELS for v in labels):
        raise ValueError('invalid labels or input alignment')
    if not set(diagnoses) | invalid <= selected_ids or set(diagnoses) & invalid:
        raise ValueError('diagnoses inconsistent with detected pairs')
    training, duplicates, decisions = [], [], []
    for row, teacher in zip(inputs, labels, strict=True):
        pid = row['pair_id']
        detected = pid in selected_ids
        cases = diagnoses.get(pid)
        verdict = step = ''
        action, final = 'keep', teacher
        if arm == 'B' and detected:
            action, final = 'remove', ''
        elif arm == 'C' and cases is not None:
            verdict, step = verdict_for(cases)
            action, final = action_for(verdict, teacher)
        decisions.append({
            'pair_id': pid, 'detected': detected, 'diagnosis_valid': cases is not None,
            'categories': ';'.join(sorted({c['category_id'] for c in cases})) if cases else '',
            'verdict': verdict, 'step': step, 'action': action, 'original_label': teacher,
            'final_label': final, 'included_in_train': action != 'remove',
            'duplicated': action == 'upweight'})
        if action == 'remove':
            continue
        out = {'dataset_id': dataset_id, **row, 'label': int(final == 'match'),
               'target_text': final.replace('_', '-'), 'label_source': LABEL_SOURCE[arm],
               'split': 'train'}
        training.append(out)
        if action == 'upweight':
            duplicates.append({**out, 'pair_id': f'{pid}#dup1', 'duplicate_of': pid})
    return training + duplicates, decisions


def _csv(rows, fields) -> bytes:
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(arm, inputs_path, majority_path, selected_path, diagnoses_path, dataset_id, expected_count):
    for path, role in [(inputs_path, 'inputs'), (majority_path, 'majority labels'),
                       (selected_path, 'selected pairs')]:
        check_source_path(path, role)
    inputs = _load_blinded_inputs(inputs_path, expected_count)
    labels = _load_majority_labels(majority_path, [r['pair_id'] for r in inputs])
    selected = json.loads(selected_path.read_text(encoding='utf-8'))
    source = keyed(inputs)
    for item in selected:
        if source.get(item['pair_id'], {}).get('input_text') != item['input_text']:
            raise ValueError('detected pair differs from blinded source')
    diagnoses, invalid = load_diagnoses(diagnoses_path, selected)
    training, decisions = treat(inputs, labels, {i['pair_id'] for i in selected},
                                diagnoses, invalid, arm, dataset_id)
    train_bytes = b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True).encode('utf-8') + b'\n'
                           for r in training)
    decisions_bytes = _csv(decisions, DECISION_FIELDS)
    detected = [d for d in decisions if d['detected']]
    manifest = {
        'policy': 'category-rule-v1', 'arm': arm, 'contract': 'plans/261023-2015-wdc-guideline-judge-treatment/contract.md',
        'gold_accessed': False, 'dataset_id': dataset_id,
        'sources': {name: _sha(p) for name, p in [
            ('inputs', inputs_path), ('majority_labels', majority_path), ('selected_pairs', selected_path),
            ('diagnoses', diagnoses_path), ('guideline', GUIDELINE)]},
        'source_count': len(inputs), 'detected_count': len(selected),
        'valid_diagnoses': len(diagnoses), 'invalid_diagnoses': len(invalid),
        'train_rows': len(training),
        'duplicate_rows': sum(1 for r in training if 'duplicate_of' in r),
        'actions': dict(sorted(Counter(d['action'] for d in decisions).items())),
        'actions_by_step_and_teacher_label': dict(sorted(Counter(
            f"step{d['step']}:{d['action']}:{d['original_label']}" for d in detected if d['step'] != '').items())),
        'class_counts': dict(sorted(Counter(r['target_text'] for r in training).items())),
        'train_sha256': hashlib.sha256(train_bytes).hexdigest(),
        'decisions_sha256': hashlib.sha256(decisions_bytes).hexdigest(),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    return {'train.jsonl': train_bytes, 'decisions.csv': decisions_bytes, 'manifest.json': manifest_bytes}, manifest


def build(arm, output_dir: Path, **sources):
    blobs, manifest = render(arm, **sources)
    if output_dir.exists():
        existing = {p.name: p.read_bytes() for p in output_dir.iterdir() if p.is_file()}
        if existing != blobs:
            raise ValueError('existing output differs; use a new directory')
        return manifest
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix='.category-rule-') as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        for name, data in blobs.items():
            (stage / name).write_bytes(data)
        os.rename(stage, output_dir)
    return manifest


def verify(arm, output_dir: Path, **sources):
    """Rebuild in memory and require byte-identical files plus internal consistency."""
    blobs, manifest = render(arm, **sources)
    existing = {p.name: p.read_bytes() for p in output_dir.iterdir() if p.is_file()}
    if existing != blobs:
        raise ValueError('published bundle differs from a fresh rebuild')
    rows = [json.loads(line) for line in blobs['train.jsonl'].splitlines()]
    originals = {r['pair_id']: r for r in rows if 'duplicate_of' not in r}
    for r in rows:
        if 'duplicate_of' in r:
            src = originals[r['duplicate_of']]
            if {k: v for k, v in r.items() if k not in ('pair_id', 'duplicate_of')} != \
               {k: v for k, v in src.items() if k != 'pair_id'}:
                raise ValueError(f'duplicate differs from its source: {r["pair_id"]}')
    if len(rows) != manifest['train_rows']:
        raise ValueError('row count differs from manifest')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl')
    parser.add_argument('--majority-labels', type=Path, default=ROOT / 'data/cache/wdc_products/majority_labels/llm_unrefined.csv')
    parser.add_argument('--selected-pairs', type=Path, default=ROOT / 'outputs/uncertainty_refinement/wdc-evidence-check/selected_pairs.json')
    parser.add_argument('--diagnoses', type=Path, default=ROOT / 'outputs/uncertainty_refinement/wdc-evidence-check/diagnoses.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--expected-count', type=int, default=2500)
    parser.add_argument('--dataset-id', default='wdc_products_80cc_small_100un')
    args = parser.parse_args()
    output = args.output_dir or ROOT / f'data/cache/wdc_products/refinement/category-rule-{args.arm}'
    sources = dict(inputs_path=args.inputs, majority_path=args.majority_labels, selected_path=args.selected_pairs,
                   diagnoses_path=args.diagnoses, dataset_id=args.dataset_id, expected_count=args.expected_count)
    result = (build if args.action == 'build' else verify)(args.arm, output, **sources)
    print(json.dumps({k: result[k] for k in ('arm', 'train_rows', 'duplicate_rows', 'actions', 'class_counts')}, indent=2))


if __name__ == '__main__':
    main()
