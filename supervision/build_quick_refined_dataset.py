"""Offline quick-tier refinement of majority labels; no API, gold or training."""
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
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from supervision.prepare_oof_folds import _load_blinded_inputs, _load_majority_labels
from supervision.run_evidence_check import collect, keyed

ROOT = Path(__file__).resolve().parents[1]
TIER_ORDER = ['intra_record_contradiction', 'cross_record_conflict',
              'one_sided_information', 'ambiguous_variant_bundle',
              'ambiguous_role_scope', 'noisy_or_malformed_data', 'representation_difference']


class Decision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    pair_id: str = Field(min_length=1)
    decision: Literal['keep', 'match', 'non_match']
    reason: str = Field(min_length=1)


def refine(inputs, labels, selected, diagnoses, ledger, dataset_id):
    """Semantic decisions are reviewed data, never inferred from note keywords."""
    source = keyed(inputs)
    selected_by_id = keyed(selected)
    if len(labels) != len(inputs) or any(v not in ('match', 'non_match') for v in labels):
        raise ValueError('invalid labels or input alignment')
    for item in selected:
        if item['pair_id'] not in source or source[item['pair_id']]['input_text'] != item['input_text']:
            raise ValueError('selected pair differs from blinded source')
    if set(diagnoses) - set(selected_by_id):
        raise ValueError('diagnosis outside selected pairs')
    reviewed = {}
    for value in ledger:
        d = Decision.model_validate(value)
        if d.pair_id in reviewed or d.pair_id not in diagnoses:
            raise ValueError('duplicate review or no valid diagnosis')
        reviewed[d.pair_id] = d
    training, audit = [], []
    for row, original in zip(inputs, labels, strict=True):
        pair_id = row['pair_id']
        diag = diagnoses.get(pair_id)
        cats = {c.category_id for c in diag.cases} if diag else set()
        ordered = [c for c in TIER_ORDER if c in cats]
        review = reviewed.get(pair_id)
        action, final, reason = 'keep', original, 'No clear reviewed change; retain majority label.'
        if pair_id not in selected_by_id:
            reason = 'Not selected for analysis; retain majority label.'
        elif diag is None:
            reason = 'No valid diagnosis; retain majority label.'
        if 'intra_record_contradiction' in cats:
            action, final = 'prune', ''
            reason = 'Quick pilot: hard-prune case 5 and stop; lower tiers not processed.'
        elif review:
            reason = review.reason
            if review.decision == 'non_match':
                if 'cross_record_conflict' not in cats:
                    raise ValueError('non-match review requires a conflict diagnosis')
                # Price differences cannot independently establish product identity.
                if not any(c.category_id == 'cross_record_conflict' and
                           any(e.side == 'A' and e.field not in ('price', 'priceCurrency') for e in c.evidence) and
                           any(e.side == 'B' and e.field not in ('price', 'priceCurrency') for e in c.evidence)
                           for c in diag.cases):
                    raise ValueError('price-only conflict cannot establish non-match')
                final = 'non_match'
            elif review.decision == 'match':
                if not cats.intersection({'representation_difference', 'one_sided_information'}):
                    raise ValueError('match review requires identity agreement evidence')
                if cats.intersection({'cross_record_conflict', 'ambiguous_variant_bundle', 'ambiguous_role_scope'}):
                    raise ValueError('unresolved higher-tier case blocks a match review')
                final = 'match'
            if final != original:
                action = 'relabel'
        audit.append({'pair_id': pair_id, 'selected': pair_id in selected_by_id,
                      'diagnosis_valid': diag is not None, 'ordered_cases': ';'.join(ordered),
                      'reviewed': review is not None, 'action': action,
                      'original_label': original, 'final_label': final,
                      'included_in_train': action != 'prune', 'reason': reason})
        if action != 'prune':
            training.append({'dataset_id': dataset_id, **row, 'label': int(final == 'match'),
                             'target_text': final.replace('_', '-'),
                             'label_source': 'llm_majority_quick_refined', 'split': 'train'})
    return training, audit


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def build(inputs_path, majority_path, evidence_dir, review_path, output_dir,
          dataset_id='wdc_products_80cc_small_100un', expected_count=2500):
    inputs = _load_blinded_inputs(inputs_path, expected_count)
    labels = _load_majority_labels(majority_path, [r['pair_id'] for r in inputs])
    contract = json.loads((evidence_dir / 'contract.json').read_text())
    for path, digest in contract['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('evidence source changed')
    selected = json.loads((evidence_dir / 'selected_pairs.json').read_text())
    label_by_id = dict(zip([r['pair_id'] for r in inputs], labels, strict=True))
    if any(label_by_id.get(r['pair_id']) != r['teacher']['majority_label'] for r in selected):
        raise ValueError('analysis labels differ from majority source')
    diagnoses, failures, _, _ = collect(selected, evidence_dir, contract['config'])
    ledger = json.loads(review_path.read_text())
    training, audit = refine(inputs, labels, selected, diagnoses, ledger, dataset_id)
    blobs = {
        'train.jsonl': ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n'
                               for r in training).encode('utf-8'),
        'labels.csv': csv_bytes([{'pair_id': r['pair_id'], 'result': r['target_text'].replace('-', '_')}
                                for r in training], ['pair_id', 'result']),
        'decisions.csv': csv_bytes(audit, list(audit[0])),
    }
    diagnosis_digest = hashlib.sha256(json.dumps(
        {p: d.model_dump() for p, d in diagnoses.items()}, ensure_ascii=False,
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    summary = {'policy': 'quick-tier-v1', 'decision_origin': 'assistant_review_of_existing_evidence',
               'validated_diagnoses_sha256': diagnosis_digest,
               'source_count': len(inputs), 'selected_count': len(selected),
               'valid_diagnoses': len(diagnoses), 'invalid_diagnoses': len(failures),
               'retained_count': len(training), 'actions': dict(Counter(r['action'] for r in audit)),
               'reviewed_count': len(ledger), 'class_counts': dict(Counter(r['target_text'] for r in training)),
               'changed_pairs': [r for r in audit if r['action'] != 'keep'],
               'api_calls': 0, 'gold_accessed': False, 'training_started': False,
               'limitations': ['Hard-prune trusts valid case-5 diagnoses, including possible classification errors.',
                               'Unresolved/unanalysed pairs retain majority labels; no full-dataset diagnosis.',
                               'Semantic ledger is assistant-reviewed, not automatic or verified human truth.'],
               'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (inputs_path, majority_path, review_path,
                                           evidence_dir / 'contract.json', evidence_dir / 'selected_pairs.json')},
               'output_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()}}
    blobs['summary.json'] = (json.dumps(summary, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir() or set(p.name for p in output_dir.iterdir()) != set(blobs):
            raise ValueError('existing output is not this complete bundle')
        if any((output_dir / name).is_symlink() or (output_dir / name).read_bytes() != data
               for name, data in blobs.items()):
            raise ValueError('existing output differs; use a new directory')
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix='.quick-refinement-') as temporary:
            stage = Path(temporary) / 'bundle'
            stage.mkdir()
            for name, data in blobs.items():
                (stage / name).write_bytes(data)
            os.rename(stage, output_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl')
    parser.add_argument('--majority-labels', type=Path, default=ROOT / 'data/cache/wdc_products/majority_labels/llm_unrefined.csv')
    parser.add_argument('--evidence-dir', type=Path, default=ROOT / 'outputs/uncertainty_refinement/wdc-evidence-check')
    parser.add_argument('--review-decisions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-count', type=int, default=2500)
    parser.add_argument('--dataset-id', default='wdc_products_80cc_small_100un')
    args = parser.parse_args()
    result = build(args.inputs, args.majority_labels, args.evidence_dir, args.review_decisions,
                   args.output_dir, args.dataset_id, args.expected_count)
    print(json.dumps({k: result[k] for k in ('retained_count', 'actions', 'class_counts')}, indent=2))


if __name__ == '__main__':
    main()
