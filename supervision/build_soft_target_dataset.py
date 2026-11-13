"""Arm S: soft targets for detected uncertain pairs (plan 261112; WDC, and
MaDI-Bench Companies under plan 261106 amendment A1).

Starts from the frozen arm A training file (all pairs, teacher majority labels;
its SHA-256 is checked against the arm A manifest) and adds, to every detected
pair only,

    soft_target = beta * teacher_label + (1 - beta) * p_OOF(match)

with beta fixed at 0.5 before any run (researcher, 2026-11-12). Detection is the
WDC rule: teacher 2/3 vote OR out-of-fold teacher–student disagreement (158 pairs),
computed from the stored OOF predictions, so no API call and no gold label is used.
Undetected pairs keep their hard label. ``target_text`` and ``label`` stay the
teacher's label on every row; only training reads ``soft_target``.

    python -m supervision.build_soft_target_dataset build [--dataset madi_companies]
    python -m supervision.build_soft_target_dataset verify [--dataset madi_companies]

Companies uses the same rule and beta; its base is the teacher majority file
(``majority_labels/train.jsonl``, arm A), and its detection reads the Companies
OOF predictions.
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

from supervision.run_evidence_check_v2 import PROFILES, select_pairs

ROOT = Path(__file__).resolve().parents[1]
REFINEMENT = ROOT / 'data/cache/wdc_products/refinement'
ARM_A = REFINEMENT / 'category-rule-A'
OUTPUT = REFINEMENT / 'soft-target-S'
BETA = 0.5
CONTRACT = 'plans/261023-2015-wdc-guideline-judge-treatment/contract.md (amendment A6)'
COMPANIES = ROOT / 'data/cache/madi_companies/madi-bench-companies-base-062058d'
DATASETS = {
    'wdc': dict(profile='wdc', arm_a=ARM_A, output=OUTPUT, contract=CONTRACT),
    'madi_companies': dict(profile='madi_companies', arm_a=COMPANIES / 'majority_labels',
                           output=COMPANIES / 'refinement/soft-target-S',
                           contract='plans/261106-1330-companies-treatment-arms/contract.md (amendment A1)'),
}
TEACHER_TEXT = {'match': 1, 'non-match': 0}
DECISION_FIELDS = ['pair_id', 'detected', 'two_of_three', 'disagreement', 'teacher_label',
                   'oof_match_probability', 'soft_target']


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_arm_a(arm_a: Path = ARM_A) -> tuple[list[dict], bytes, dict]:
    data = (arm_a / 'train.jsonl').read_bytes()
    manifest = json.loads((arm_a / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('arm') != 'A' or manifest.get('gold_accessed') is not False:
        raise ValueError('expected the gold-free arm A manifest')
    if _sha(data) != manifest['train_sha256']:
        raise ValueError('arm A train.jsonl differs from its manifest')
    rows = [json.loads(line) for line in data.decode('utf-8').splitlines()]
    if any('duplicate_of' in r or 'soft_target' in r for r in rows):
        raise ValueError('arm A must have one row per pair and no soft targets')
    return rows, data, manifest


def teacher_label(row: dict) -> int:
    """WDC arm A rows carry ``label``; Companies rows carry only ``target_text``."""
    from_text = TEACHER_TEXT[row['target_text']]
    if 'label' in row and int(row['label']) != from_text:
        raise ValueError(f'label and target_text disagree: {row["pair_id"]}')
    return from_text


def soft_target(teacher_label: int, p_match: float, beta: float = BETA) -> float:
    if teacher_label not in (0, 1) or not 0.0 <= p_match <= 1.0 or not 0.0 <= beta <= 1.0:
        raise ValueError('invalid teacher label, probability or beta')
    return beta * teacher_label + (1.0 - beta) * p_match


def render(beta: float = BETA, profile=None, arm_a: Path = ARM_A,
           contract: str = CONTRACT) -> tuple[dict[str, bytes], dict]:
    profile = profile or PROFILES['wdc']
    rows, a_bytes, a_manifest = load_arm_a(arm_a)
    selected, counts = select_pairs(profile)
    by_id = {item['pair_id']: item for item in selected}
    if set(by_id) - {r['pair_id'] for r in rows}:
        raise ValueError('detected pairs are missing from arm A')
    training, decisions = [], []
    for row in rows:
        item = by_id.get(row['pair_id'])
        teacher = teacher_label(row)
        if item is not None and item['oof']['given_label'] != ('match' if teacher else 'non_match'):
            raise ValueError(f'teacher label differs between arm A and the OOF file: {row["pair_id"]}')
        out = dict(row)
        decision = {'pair_id': row['pair_id'], 'detected': item is not None, 'two_of_three': '', 'disagreement': '',
                    'teacher_label': teacher, 'oof_match_probability': '', 'soft_target': ''}
        if item is not None:
            p = float(item['oof']['match_probability'])
            s = round(soft_target(teacher, p, beta), 6)
            out['soft_target'] = s
            out['label_source'] = 'llm_majority_soft_target'
            decision.update(two_of_three=item['two_of_three'], disagreement=item['disagreement'],
                            oof_match_probability=format(p, '.6f'), soft_target=s)
        training.append(out)
        decisions.append(decision)
    train_bytes = b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True).encode('utf-8') + b'\n' for r in training)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=DECISION_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(decisions)
    decisions_bytes = stream.getvalue().encode('utf-8')
    soft = [r['soft_target'] for r in training if 'soft_target' in r]
    manifest = {
        'arm': 'S', 'policy': 'soft-target-v1', 'beta': beta, 'contract': contract,
        'gold_accessed': False, 'dataset_id': a_manifest['dataset_id'],
        'base': {'arm_a_train_sha256': _sha(a_bytes)},
        'sources': {'arm_a_train': _sha(a_bytes),
                    'oof_predictions': _sha(profile.oof.read_bytes()),
                    'oof_summary': _sha(profile.oof_summary.read_bytes()),
                    'consistency': _sha(profile.consistency.read_bytes()),
                    'inputs': _sha(profile.inputs.read_bytes())},
        'selection_rule': 'teacher_2of3_OR_stored_prediction_disagreement', 'selection_counts': counts,
        'train_rows': len(training), 'soft_target_rows': len(soft),
        'soft_target_summary': {'min': min(soft), 'max': max(soft), 'mean': round(sum(soft) / len(soft), 6),
                                'teacher_match': sum(teacher_label(r) == 1 for r in training if 'soft_target' in r),
                                'teacher_non_match': sum(teacher_label(r) == 0 for r in training if 'soft_target' in r)},
        'class_counts': dict(sorted(Counter(r['target_text'] for r in training).items())),
        'train_sha256': _sha(train_bytes), 'decisions_sha256': _sha(decisions_bytes),
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode('utf-8')
    return {'train.jsonl': train_bytes, 'decisions.csv': decisions_bytes, 'manifest.json': manifest_bytes}, manifest


def build(output: Path = OUTPUT, **kwargs) -> dict:
    blobs, manifest = render(**kwargs)
    if output.exists():
        existing = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
        if existing != blobs:
            raise ValueError('existing output differs; use a new directory')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.soft-target-') as temporary:
        stage = Path(temporary) / 'bundle'
        stage.mkdir()
        for name, data in blobs.items():
            (stage / name).write_bytes(data)
        os.rename(stage, output)
    return manifest


def verify(output: Path = OUTPUT, **kwargs) -> dict:
    blobs, manifest = render(**kwargs)
    existing = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    if existing != blobs:
        raise ValueError('published bundle differs from a fresh rebuild')
    return manifest


def dataset_kwargs(name: str) -> dict:
    d = DATASETS[name]
    return dict(output=d['output'], profile=PROFILES[d['profile']], arm_a=d['arm_a'], contract=d['contract'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='wdc')
    args = parser.parse_args()
    manifest = (build if args.action == 'build' else verify)(**dataset_kwargs(args.dataset))
    print(json.dumps({k: manifest[k] for k in ('arm', 'beta', 'train_rows', 'soft_target_rows',
                                               'soft_target_summary', 'class_counts', 'selection_counts')}, indent=2))


if __name__ == '__main__':
    main()
