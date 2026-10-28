"""Frozen preparation of the MaDI-Bench Companies entity-matching subtask.

Reads the raw MaDI-Bench files (commit 062058d) under ``data/raw/madi_companies``,
checks every hash and count against ``configs/datasets/madi_companies.json``,
renames columns with the benchmark's gold schema mapping, applies light cleanup
only, pools both source pairs, and publishes train/validation JSONL plus a gold
training target and teacher-labelling inputs. Test stays locked: only its file
hashes are checked, unless ``materialize-test --confirm-test`` is run.

The source data has no licence and FullContact forbids redistribution: outputs
are written to the git-ignored ``data/cache/madi_companies/`` only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import GenericERPair, GenericERRecord
from data.serialize_pairs import serialize_pair, write_serialized_pairs

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/datasets/madi_companies.json'
MATERIALIZED = ('train', 'validation')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')


def load_profile(path: Path = PROFILE) -> dict:
    profile = json.loads(path.read_text(encoding='utf-8'))
    if profile.get('status') != 'frozen' or profile.get('dataset_id') != 'madi_companies':
        raise ValueError('expected the frozen madi_companies profile')
    if profile['source'].get('redistribution_allowed') is not False:
        raise ValueError('profile must state that redistribution is not allowed')
    if profile['splits']['test'].get('materialize') or not profile['splits']['test'].get('locked'):
        raise ValueError('test must stay locked in the profile')
    return profile


def _verify_file(root: Path, contract: dict) -> Path:
    path = root / contract['path']
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'missing or aliased source file: {contract["path"]}')
    if path.stat().st_size != contract['size_bytes'] or _sha256(path) != contract['sha256']:
        raise ValueError(f'source contract mismatch: {contract["path"]}')
    return path


def verify_sources(profile: dict, root: Path) -> None:
    for table in profile['record_tables'].values():
        _verify_file(root, table)
    for split in profile['splits'].values():
        for contract in split['files'].values():
            _verify_file(root, contract)
    mapping_path = root / profile['schema_mapping']['path']
    if _sha256(mapping_path) != profile['schema_mapping']['sha256']:
        raise ValueError('schema mapping hash mismatch')
    gold = json.loads(mapping_path.read_text(encoding='utf-8'))
    found: dict[str, dict[str, str]] = {}
    for m in gold['mappings']:
        if m.get('label') is True:
            found.setdefault(m['source_dataset'], {})[m['source_column']] = m['target_column']
    for name, table in profile['record_tables'].items():
        expected = {**table['column_mapping'], table['id_column']: 'id'}
        if found.get(name) != expected:
            raise ValueError(f'profile column mapping for {name} differs from the gold schema mapping')


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def load_records(profile: dict, root: Path) -> dict[str, dict[str, dict[str, str | None]]]:
    records: dict[str, dict[str, dict[str, str | None]]] = {}
    for name, table in profile['record_tables'].items():
        with (root / table['path']).open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != table['columns']:
                raise ValueError(f'unexpected header in {table["path"]}')
            rows = list(reader)
        if len(rows) != table['row_count']:
            raise ValueError(f'unexpected row count in {table["path"]}')
        nulls = table.get('literal_null_columns', {})
        null_counts = Counter()
        by_id: dict[str, dict[str, str | None]] = {}
        for row in rows:
            raw_id = row[table['id_column']].strip()
            if not raw_id or raw_id in by_id:
                raise ValueError(f'empty or duplicate id in {table["path"]}')
            attributes = {}
            for source_column, target in table['column_mapping'].items():
                value = _clean(row[source_column])
                if source_column in nulls and value is not None and value.lower() == 'null':
                    null_counts[source_column] += 1
                    value = None
                attributes[target] = value
            by_id[raw_id] = attributes
        if dict(null_counts) != nulls:
            raise ValueError(f'literal null counts differ in {table["path"]}: {dict(null_counts)}')
        records[name] = by_id
    return records


def _forbes_slug(profile: dict, forbes_id: str) -> str:
    prefix = profile['identity']['forbes_slug_prefix']
    if not forbes_id.startswith(prefix):
        raise ValueError(f'unexpected Forbes id: {forbes_id}')
    return forbes_id[len(prefix):].strip('/')


def _right_key(profile: dict, source: str, right_id: str) -> str:
    if source == 'dbpedia':
        prefix = profile['identity']['dbpedia_key_prefix']
        if not right_id.startswith(prefix):
            raise ValueError(f'unexpected DBpedia id: {right_id}')
        return right_id[len(prefix):]
    return right_id


def read_pair_file(path: Path, profile: dict) -> list[tuple[str, str, bool]]:
    labels = profile['pair_file_format']['labels']
    out = []
    with path.open(encoding='utf-8', newline='') as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(f'bad pair row in {path.name}: {row}')
            label = row[2].strip().lower()
            if label not in labels:
                raise ValueError(f'unknown label {row[2]!r} in {path.name}')
            out.append((row[0].strip(), row[1].strip(), labels[label]))
    return out


def load_split(profile: dict, root: Path, records: dict, split: str) -> tuple[list[GenericERPair], dict]:
    contract = profile['splits'][split]
    dropped_expected = {tuple(p) for p in contract['dropped_unresolved_pairs']}
    pairs: list[GenericERPair] = []
    audit: dict[str, Any] = {}
    seen: set[tuple[str, str, str]] = set()
    for source in profile['source_pairs']:
        file_contract = contract['files'][source]
        rows = read_pair_file(root / file_contract['path'], profile)
        counts = Counter(label for _, _, label in rows)
        if (len(rows), counts[True], counts[False]) != (
                file_contract['row_count'], file_contract['match_count'], file_contract['non_match_count']):
            raise ValueError(f'pair counts differ for {split}/{source}')
        dropped = []
        for forbes_id, right_id, label in rows:
            key = (source, forbes_id, right_id)
            if key in seen:
                raise ValueError(f'duplicate pair in {split}/{source}: {forbes_id}, {right_id}')
            seen.add(key)
            if forbes_id not in records['forbes'] or right_id not in records[source]:
                dropped.append((forbes_id, right_id))
                continue
            pair_id = profile['identity']['pair_id_template'].format(
                split=split, source_pair=source, forbes_slug=_forbes_slug(profile, forbes_id),
                right_key=_right_key(profile, source, right_id))
            pairs.append(GenericERPair(
                pair_id=pair_id, label=label, split=split,
                record_a=GenericERRecord(record_id=forbes_id, source='forbes', attributes=records['forbes'][forbes_id]),
                record_b=GenericERRecord(record_id=right_id, source=source, attributes=records[source][right_id]),
                metadata={'source_pair': f'forbes_2_{source}', 'forbes_id': forbes_id, 'right_id': right_id}))
        audit[source] = {'rows': len(rows), 'matches': counts[True], 'non_matches': counts[False], 'dropped_unresolved': len(dropped)}
        dbpedia_prefix = profile['identity']['dbpedia_key_prefix']
        expected_here = {p for p in dropped_expected if p[1].startswith(dbpedia_prefix) == (source == 'dbpedia')}
        if set(dropped) != expected_here:
            raise ValueError(f'unresolved pairs in {split}/{source} differ from the profile: {dropped}')
    pooled = Counter(p.label for p in pairs)
    if (len(pairs), pooled[True], pooled[False]) != (
            contract['pooled_row_count'], contract['pooled_match_count'], contract['pooled_non_match_count']):
        raise ValueError(f'pooled counts differ for {split}')
    if len({p.pair_id for p in pairs}) != len(pairs):
        raise ValueError(f'pair ids not unique in {split}')
    return pairs, audit


def _gold_rows(pairs: list[GenericERPair], profile: dict) -> list[dict]:
    rows = []
    for p in pairs:
        rows.append({'dataset_id': profile['dataset_id'], 'pair_id': p.pair_id,
                     'input_text': serialize_pair(p, profile['attribute_order'], profile['missing_value_token']),
                     'label_source': 'gold', 'split': p.split, 'target_text': 'match' if p.label else 'non-match'})
    return rows


def _jsonl(rows: list[dict]) -> bytes:
    return b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True).encode('utf-8') + b'\n' for r in rows)


def render(profile_path: Path, root: Path) -> dict[str, bytes]:
    """All publication files as bytes (deterministic)."""
    profile = load_profile(profile_path)
    verify_sources(profile, root)
    records = load_records(profile, root)
    splits, audits = {}, {}
    for split in MATERIALIZED:
        splits[split], audits[split] = load_split(profile, root, records, split)
    train_ids = {(p.record_a.record_id, p.record_b.record_id) for p in splits['train']}
    val_ids = {(p.record_a.record_id, p.record_b.record_id) for p in splits['validation']}
    if train_ids & val_ids:
        raise ValueError('a pair appears in both train and validation')
    blobs: dict[str, bytes] = {}
    with _tmp() as tmp:
        for split in MATERIALIZED:
            out = tmp / f'{split}.jsonl'
            write_serialized_pairs(splits[split], out, attribute_order=profile['attribute_order'],
                                   missing_value=profile['missing_value_token'])
            blobs[f'serialized/{split}.jsonl'] = out.read_bytes()
    gold = _gold_rows(splits['train'], profile)
    blobs['full_label_targets/gold.jsonl'] = _jsonl(gold)
    blobs['teacher_labels/inputs.jsonl'] = _jsonl([{'pair_id': r['pair_id'], 'input_text': r['input_text']} for r in gold])
    stats = {
        'dataset_id': profile['dataset_id'], 'logical_version': profile['logical_version'],
        'splits': {s: {'pairs': len(splits[s]), 'matches': sum(p.label for p in splits[s]),
                       'non_matches': sum(not p.label for p in splits[s]), 'by_source_pair': audits[s]}
                   for s in MATERIALIZED},
        'test': {'locked': True, 'materialized': False,
                 'files': {k: v['sha256'] for k, v in profile['splits']['test']['files'].items()}},
    }
    blobs['stats.json'] = _json_bytes(stats)
    gold_bytes = blobs['full_label_targets/gold.jsonl']
    blobs['arm-manifests/G.json'] = _json_bytes({
        'arm': 'G', 'policy': 'gold-reference', 'label_source': 'gold', 'gold_accessed': True,
        'purpose': 'Reference arm: student trained on the benchmark human training labels.',
        'dataset_id': profile['dataset_id'], 'train_path': 'full_label_targets/gold.jsonl',
        'train_rows': len(gold), 'train_sha256': hashlib.sha256(gold_bytes).hexdigest(),
        'class_counts': dict(sorted(Counter(r['target_text'] for r in gold).items())),
    })
    manifest = {
        'dataset_id': profile['dataset_id'], 'logical_version': profile['logical_version'],
        'profile_sha256': _sha256(profile_path), 'source_commit': profile['source']['commit'],
        'redistribution_allowed': False, 'gold_in_outputs': ['serialized/train.jsonl', 'serialized/validation.jsonl', 'full_label_targets/gold.jsonl'],
        'gold_free_outputs': ['teacher_labels/inputs.jsonl'],
        'outputs': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)} for name, data in sorted(blobs.items())},
    }
    blobs['manifest.json'] = _json_bytes(manifest)
    return blobs


class _tmp:
    def __enter__(self):
        import tempfile
        self._dir = tempfile.TemporaryDirectory(prefix='madi-render-')
        return Path(self._dir.name)

    def __exit__(self, *exc):
        self._dir.cleanup()


def _differences(output: Path, blobs: dict[str, bytes]) -> list[str]:
    """Publication files that are missing or differ; other files (downstream, test) are ignored."""
    return sorted(name for name, data in blobs.items()
                  if not (output / name).is_file() or (output / name).read_bytes() != data)


def prepare(profile_path: Path, root: Path, output: Path, verify_only: bool = False) -> dict:
    blobs = render(profile_path, root)
    manifest = json.loads(blobs['manifest.json'])
    if verify_only or output.exists():
        if not output.is_dir():
            raise FileNotFoundError(f'prepared output does not exist: {output}')
        diff = _differences(output, blobs)
        if diff:
            raise ValueError(f'prepared output differs from a fresh preparation: {diff}')
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f'.{output.name}.staging'
    if stage.exists():
        raise RuntimeError(f'orphan staging directory needs manual inspection: {stage}')
    try:
        for name, data in blobs.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def materialize_test(profile_path: Path, root: Path, output: Path) -> dict:
    """Write the locked test split once, for frozen-arm test evaluation only."""
    profile = load_profile(profile_path)
    verify_sources(profile, root)
    target = output / 'serialized-test/test.jsonl'
    if target.exists():
        raise ValueError(f'test already materialized: {target}')
    records = load_records(profile, root)
    pairs, audit = load_split(profile, root, records, 'test')
    train = {(r['record_a']['record_id'], r['record_b']['record_id'])
             for r in map(json.loads, (output / 'serialized/train.jsonl').read_text(encoding='utf-8').splitlines())}
    if train & {(p.record_a.record_id, p.record_b.record_id) for p in pairs}:
        raise ValueError('a test pair appears in train')
    target.parent.mkdir(parents=True, exist_ok=False)
    write_serialized_pairs(pairs, target, attribute_order=profile['attribute_order'], missing_value=profile['missing_value_token'])
    info = {'pairs': len(pairs), 'by_source_pair': audit, 'sha256': _sha256(target)}
    (target.parent / 'test-manifest.json').write_bytes(_json_bytes(info))
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify', 'materialize-test'])
    parser.add_argument('--profile', type=Path, default=PROFILE)
    parser.add_argument('--confirm-test', action='store_true')
    args = parser.parse_args()
    profile = load_profile(args.profile)
    root = ROOT / profile['source']['local_root']
    output = ROOT / profile['cache']['root']
    if args.action == 'materialize-test':
        if not args.confirm_test:
            parser.error('materialize-test requires --confirm-test')
        result = materialize_test(args.profile, root, output)
    else:
        result = prepare(args.profile, root, output, verify_only=args.action == 'verify')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
