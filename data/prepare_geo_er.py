"""Frozen preparation of Geo-ER Pittsburgh (OSM-Foursquare), dataset 2 (D24).

Reads the official split files (Geo-ER commit 83063f7) under ``data/raw/geo_er``,
checks every hash and count against ``configs/datasets/geo_er_pit.json``, parses
the ``COL <field> VAL <value>`` records, applies light cleanup only (blank and
``NULL`` become ``<missing>``), and adds one computed pair field, ``distance_m``
(haversine metres between the two coordinates; uses no labels). Teacher and
student read the same ``input_text``. Test stays locked: only its hash and counts
are checked, unless ``materialize-test --confirm-test`` is run.

The source data has no licence and Foursquare's terms are restrictive: outputs
are written to the git-ignored ``data/cache/geo_er/`` only.

    python -m data.prepare_geo_er prepare
    python -m data.prepare_geo_er verify
    python -m data.prepare_geo_er materialize-test --confirm-test   # GPU host, frozen arms only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from data.schema import GenericERPair, GenericERRecord
from data.serialize_pairs import serialize_pair

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/datasets/geo_er_pit.json'
MATERIALIZED = ('train', 'validation')
EARTH_RADIUS_M = 6371008.8
_FIELD = re.compile(r'COL (\S+) VAL ')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def _jsonl(rows: list[dict]) -> bytes:
    return b''.join(json.dumps(r, ensure_ascii=False, sort_keys=True).encode('utf-8') + b'\n' for r in rows)


def load_profile(path: Path = PROFILE) -> dict:
    profile = json.loads(path.read_text(encoding='utf-8'))
    if profile.get('status') != 'frozen' or profile.get('dataset_id') != 'geo_er_pit':
        raise ValueError('expected the frozen geo_er_pit profile')
    if profile['source']['redistribution_allowed'] is not False:
        raise ValueError('Geo-ER data must stay non-redistributable')
    if profile['splits']['test'].get('materialize') is not False:
        raise ValueError('test must stay locked in the profile')
    return profile


def verify_sources(profile: dict, root: Path) -> None:
    for split, contract in profile['splits'].items():
        path = root / contract['path']
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f'missing source file for {split}: {path}')
        if _sha256(path) != contract['sha256']:
            raise ValueError(f'source hash mismatch for {split}: {path}')


def _clean(value: str, profile: dict) -> str | None:
    value = value.strip()
    if value == '' or value in profile['cleanup']['literal_null_to_missing']:
        return None
    return value


def parse_record(text: str, profile: dict) -> dict[str, str | None]:
    parts = _FIELD.split(text)
    if parts[0].strip():
        raise ValueError(f'text before the first field: {text[:80]!r}')
    keys, values = parts[1::2], parts[2::2]
    if keys != profile['attribute_order']:
        raise ValueError(f'unexpected fields {keys}')
    return {k: _clean(v, profile) for k, v in zip(keys, values)}


def distance_m(a: dict, b: dict) -> int | None:
    """Haversine distance in whole metres, or None when a coordinate is missing or not numeric."""
    try:
        lat1, lon1, lat2, lon2 = (float(x) for x in (a['latitude'], a['longitude'], b['latitude'], b['longitude']))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (lat1, lon1, lat2, lon2)):
        return None
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return round(2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, h))))


def input_text(pair: GenericERPair, profile: dict) -> str:
    missing = profile['missing_value_token']
    distance = pair.metadata['distance_m']
    return (serialize_pair(pair, profile['attribute_order'], missing)
            + f"\n\nPair:\n- distance_m: {missing if distance is None else distance}")


def load_split(profile: dict, root: Path, split: str) -> list[GenericERPair]:
    contract = profile['splits'][split]
    labels = profile['pair_file_format']['labels']
    lines = (root / contract['path']).read_text(encoding='utf-8').splitlines()
    pairs = []
    for number, line in enumerate(lines, start=1):
        fields = line.split('\t')
        if len(fields) != 3 or fields[2].strip() not in labels:
            raise ValueError(f'bad line {number} in {split}')
        a, b = parse_record(fields[0], profile), parse_record(fields[1], profile)
        pairs.append(GenericERPair(
            pair_id=profile['identity']['pair_id_template'].format(split=split, line=number),
            label=labels[fields[2].strip()], split=split,
            record_a=GenericERRecord(record_id=f'{split}:{number}:a', source=profile['sources']['a'], attributes=a),
            record_b=GenericERRecord(record_id=f'{split}:{number}:b', source=profile['sources']['b'], attributes=b),
            metadata={'line': number, 'distance_m': distance_m(a, b)}))
    counts = Counter(p.label for p in pairs)
    if (len(pairs), counts[True], counts[False]) != (contract['row_count'], contract['match_count'], contract['non_match_count']):
        raise ValueError(f'pair counts differ for {split}')
    keys = [_content_key(p) for p in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f'duplicate pair content in {split}')
    return pairs


def _content_key(pair: GenericERPair) -> str:
    return json.dumps([pair.record_a.attributes, pair.record_b.attributes], sort_keys=True, ensure_ascii=False)


def _serialized_rows(pairs: list[GenericERPair], profile: dict) -> list[dict]:
    return [{'pair_id': p.pair_id, 'split': p.split, 'label': int(p.label),
             'target_label': 'match' if p.label else 'non-match', 'input_text': input_text(p, profile),
             'record_a': p.record_a.model_dump(), 'record_b': p.record_b.model_dump(), 'metadata': p.metadata}
            for p in pairs]


def _distance_summary(pairs: list[GenericERPair]) -> dict:
    out = {}
    for name, label in (('match', True), ('non_match', False)):
        d = sorted(p.metadata['distance_m'] for p in pairs if p.label == label and p.metadata['distance_m'] is not None)
        out[name] = {'n': len(d), 'median_m': d[len(d) // 2] if d else None, 'max_m': d[-1] if d else None}
    out['missing'] = sum(p.metadata['distance_m'] is None for p in pairs)
    return out


def render(profile_path: Path, root: Path) -> dict[str, bytes]:
    """All publication files as bytes (deterministic)."""
    profile = load_profile(profile_path)
    verify_sources(profile, root)
    splits = {s: load_split(profile, root, s) for s in MATERIALIZED}
    if {_content_key(p) for p in splits['train']} & {_content_key(p) for p in splits['validation']}:
        raise ValueError('a pair appears in both train and validation')
    blobs: dict[str, bytes] = {f'serialized/{s}.jsonl': _jsonl(_serialized_rows(splits[s], profile)) for s in MATERIALIZED}
    gold = [{'dataset_id': profile['dataset_id'], 'pair_id': p.pair_id, 'input_text': input_text(p, profile),
             'label_source': 'gold', 'split': 'train', 'target_text': 'match' if p.label else 'non-match'}
            for p in splits['train']]
    blobs['full_label_targets/gold.jsonl'] = _jsonl(gold)
    blobs['teacher_labels/inputs.jsonl'] = _jsonl([{'pair_id': r['pair_id'], 'input_text': r['input_text']} for r in gold])
    blobs['stats.json'] = _json_bytes({
        'dataset_id': profile['dataset_id'], 'logical_version': profile['logical_version'],
        'splits': {s: {'pairs': len(splits[s]), 'matches': sum(p.label for p in splits[s]),
                       'non_matches': sum(not p.label for p in splits[s]), 'distance_m': _distance_summary(splits[s])}
                   for s in MATERIALIZED},
        'test': {'locked': True, 'materialized': False, 'sha256': profile['splits']['test']['sha256']},
    })
    gold_bytes = blobs['full_label_targets/gold.jsonl']
    blobs['arm-manifests/G.json'] = _json_bytes({
        'arm': 'G', 'policy': 'gold-reference', 'label_source': 'gold', 'gold_accessed': True,
        'purpose': 'Reference arm: student trained on the benchmark human training labels.',
        'dataset_id': profile['dataset_id'], 'train_path': 'full_label_targets/gold.jsonl',
        'train_rows': len(gold), 'train_sha256': hashlib.sha256(gold_bytes).hexdigest(),
        'class_counts': dict(sorted(Counter(r['target_text'] for r in gold).items())),
    })
    blobs['manifest.json'] = _json_bytes({
        'dataset_id': profile['dataset_id'], 'logical_version': profile['logical_version'],
        'profile_sha256': _sha256(profile_path), 'source_commit': profile['source']['commit'],
        'redistribution_allowed': False,
        'gold_in_outputs': ['serialized/train.jsonl', 'serialized/validation.jsonl', 'full_label_targets/gold.jsonl'],
        'gold_free_outputs': ['teacher_labels/inputs.jsonl'],
        'outputs': {n: {'sha256': hashlib.sha256(d).hexdigest(), 'size_bytes': len(d)} for n, d in sorted(blobs.items())},
    })
    return blobs


def _differences(output: Path, blobs: dict[str, bytes]) -> list[str]:
    return sorted(n for n, d in blobs.items() if not (output / n).is_file() or (output / n).read_bytes() != d)


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
            (stage / name).parent.mkdir(parents=True, exist_ok=True)
            (stage / name).write_bytes(data)
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
    pairs = load_split(profile, root, 'test')
    train = {_content_key(p) for p in load_split(profile, root, 'train')}
    if train & {_content_key(p) for p in pairs}:
        raise ValueError('a test pair appears in train')
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(_jsonl(_serialized_rows(pairs, profile)))
    info = {'pairs': len(pairs), 'sha256': _sha256(target)}
    (target.parent / 'test-manifest.json').write_bytes(_json_bytes(info))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
