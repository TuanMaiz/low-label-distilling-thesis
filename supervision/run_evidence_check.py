"""Blinded, multi-label evidence diagnosis; never relabels training pairs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from supervision.build_confident_learning import _load_oof
from supervision.llm_providers import TokenPricing, _cost_from_usage, resolve_openrouter_api_key
from supervision.openrouter_json_schema_client import create_openrouter_json_schema_client

ROOT = Path(__file__).resolve().parents[1]
PROMPT = Path(__file__).with_name('evidence_case_prompt.md')
CATEGORY_NAMES = {
    'representation_difference': 'Khác biểu diễn tương đương',
    'one_sided_information': 'Thông tin chỉ có một phía',
    'cross_record_conflict': 'Xung đột giữa hai bản ghi',
    'ambiguous_variant_bundle': 'Không rõ phiên bản/bộ sản phẩm',
    'intra_record_contradiction': 'Mâu thuẫn nội bộ bản ghi',
    'ambiguous_role_scope': 'Không rõ vai trò/phạm vi thông tin',
    'noisy_or_malformed_data': 'Nhiễu/bất thường dữ liệu',
}
Category = Literal['representation_difference', 'one_sided_information', 'cross_record_conflict',
                   'ambiguous_variant_bundle', 'intra_record_contradiction', 'ambiguous_role_scope',
                   'noisy_or_malformed_data']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Evidence(StrictModel):
    side: Literal['A', 'B']
    field: Literal['title', 'brand', 'description', 'price', 'priceCurrency']
    quote: str = Field(min_length=1)


class Case(StrictModel):
    category_id: Category
    attribute: str = Field(min_length=1)
    note_vi: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class Diagnosis(StrictModel):
    status: Literal['categorized', 'no_finding', 'unknown']
    cases: list[Case]
    summary_vi: str = Field(min_length=1)
    uncertainties_vi: list[str]


def dump(path: Path, value) -> None:
    """Atomic derived artifact publication; source files are never modified."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def keyed(rows):
    result = {}
    for row in rows:
        key = row['pair_id']
        if not key or key in result:
            raise ValueError('empty or duplicate pair_id')
        result[key] = row
    return result


def parse_fields(text: str) -> dict:
    result = {'A': {}, 'B': {}}
    side = None
    for line in text.splitlines():
        if line in ('Record A:', 'Record B:'):
            side = line[7]
        elif line.startswith('- ') and ': ' in line and side:
            field, value = line[2:].split(': ', 1)
            if field not in Evidence.model_fields['field'].annotation.__args__ or field in result[side]:
                raise ValueError('unexpected or duplicate record field')
            result[side][field] = value
    if any(len(result[s]) != 5 for s in result):
        raise ValueError('expected five serialized fields per record')
    return result


def select_pairs(inputs: Path, consistency: Path, oof: Path, summary: Path, expected_count=2500):
    with inputs.open(encoding='utf-8') as handle:
        input_rows = list(map(json.loads, handle))
    if any(set(r) != {'pair_id', 'input_text'} for r in input_rows):
        raise ValueError('inputs must contain only pair_id and input_text')
    inp = keyed(input_rows)
    with consistency.open(encoding='utf-8', newline='') as handle:
        con = keyed(csv.DictReader(handle))
    oof_rows, _, _, _, _ = _load_oof(oof, summary, expected_count)
    if len(inp) != expected_count or set(inp) != set(con) or set(inp) != {r['pair_id'] for r in oof_rows}:
        raise ValueError('source IDs/counts are not aligned')
    selected = []
    counts = Counter()
    for row in oof_rows:
        teacher = con[row['pair_id']]
        votes = [teacher[f'pass_0{i}'] for i in range(1, 4)]
        if any(v not in ('match', 'non_match') for v in votes):
            raise ValueError('invalid teacher pass label')
        label, n = Counter(votes).most_common(1)[0]
        if (teacher['majority_label'] != label or teacher['majority_votes'] != str(n)
                or teacher['consistency'] != f'{n}/3' or row['given_label'] != label):
            raise ValueError('teacher/OOF majority mismatch')
        inconsistent = n == 2
        disagree = row['given_label'] != row['predicted_label']
        counts['two_of_three'] += inconsistent
        counts['disagreement'] += disagree
        counts['overlap'] += inconsistent and disagree
        if inconsistent or disagree:
            item = {'pair_id': row['pair_id'], 'input_text': inp[row['pair_id']]['input_text'],
                    'two_of_three': inconsistent, 'disagreement': disagree, 'oof': row, 'teacher': teacher}
            parse_fields(item['input_text'])
            selected.append(item)
    counts['union'] = len(selected)
    return selected, dict(counts)


def payload_for(item: dict, config: dict, prompt: str) -> dict:
    # Metadata, pair ID, previous labels and student scores remain local only.
    record_text = item['input_text'].split('Record A:', 1)[1]
    return {
        'model': config['settings']['sol_high']['model'],
        'messages': [{'role': 'system', 'content': prompt},
                     {'role': 'user', 'content': 'Record A:' + record_text}],
        'reasoning': config['settings']['sol_high']['reasoning'],
        'provider': config['provider_routing'],
        'max_tokens': 8192,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'evidence_case_diagnosis', 'strict': True, 'schema': Diagnosis.model_json_schema()}},
    }


def validate_diagnosis(raw: dict, item: dict) -> Diagnosis:
    if raw.get('error') or not raw.get('choices'):
        raise ValueError('missing successful completion')
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop' or choice.get('message', {}).get('refusal'):
        raise ValueError('refused or incomplete completion')
    diag = Diagnosis.model_validate_json(choice['message']['content'])
    if (diag.status == 'categorized') != bool(diag.cases):
        raise ValueError('inconsistent status and cases')
    fields = parse_fields(item['input_text'])
    for case in diag.cases:
        for evidence in case.evidence:
            value = fields[evidence.side][evidence.field]
            if value == '<missing>' or evidence.quote not in value or not evidence.quote.strip():
                raise ValueError(f'ungrounded evidence quote: {evidence.side}.{evidence.field}: {evidence.quote!r}')
        if case.category_id == 'cross_record_conflict' and {e.side for e in case.evidence} != {'A', 'B'}:
            raise ValueError('cross-record conflict requires both sides')
        if case.category_id == 'intra_record_contradiction':
            if not any(len({(e.field, e.quote) for e in case.evidence if e.side == s}) >= 2 for s in ('A', 'B')):
                raise ValueError('internal contradiction requires two quotes on one side')
    return diag


def cost_of(raw: dict, config: dict) -> tuple[float | None, str]:
    usage = raw.get('usage') or {}
    cost = _cost_from_usage(usage)
    source = 'provider_usage'
    if cost is None:
        if 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
            return None, 'unknown'
        pricing = config['pricing_snapshot']
        cost = TokenPricing(pricing['input_usd_per_million_tokens'], pricing['output_usd_per_million_tokens']).estimate(
            usage['prompt_tokens'], usage['completion_tokens'])
        source = 'configured_pricing_estimate'
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('invalid cost')
    return cost, source


def prepare(args):
    paths = [args.inputs, args.consistency, args.oof, args.oof_summary, args.settings, PROMPT]
    config = json.loads(args.settings.read_text(encoding='utf-8'))
    if (config['api_url'] != 'https://openrouter.ai/api/v1' or
            config['provider_routing']['only'] != ['openai'] or
            config['provider_routing']['allow_fallbacks'] is not False):
        raise ValueError('expected canonical OpenRouter and pinned OpenAI routing')
    items, counts = select_pairs(args.inputs, args.consistency, args.oof, args.oof_summary, args.expected_count)
    contract = {'schema_version': 1, 'selection_rule': 'teacher_2of3_OR_stored_prediction_disagreement',
                'sources': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'counts': counts, 'config': config, 'output_token_cap': 8192,
                'response_schema': Diagnosis.model_json_schema(), 'gold_accessed': False}
    for name, value in [('contract.json', contract), ('selected_pairs.json', items)]:
        target = args.output_dir / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError('existing run contract or inputs changed; use a new output directory')
        if not target.exists():
            dump(target, value)
    return items, config


def response_paths(output: Path, index: int):
    original = output / 'responses' / f'{index:04d}.json'
    return ([original] if original.exists() else []) + sorted(
        (output / 'retries').glob(f'*/responses/{index:04d}.json'))


def collect(items, output: Path, config):
    results, failures = {}, {}
    total = 0.0
    cost_sources = Counter()
    for i, item in enumerate(items):
        for path in response_paths(output, i):
            raw = json.loads(path.read_text(encoding='utf-8'))
            try:
                cost, source = cost_of(raw, config)
            except (ValueError, TypeError, OverflowError, AttributeError, KeyError):
                cost, source = None, 'unknown'
            cost_sources[source] += 1
            if cost is not None:
                total += cost
            try:
                results[item['pair_id']] = validate_diagnosis(raw, item)
                failures.pop(item['pair_id'], None)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                if item['pair_id'] not in results:
                    failures[item['pair_id']] = str(error)
    return results, failures, total, dict(cost_sources)


def report(items, output: Path, config):
    results, failures, total, cost_sources = collect(items, output, config)
    categories = Counter()
    groups = {'two_of_three': Counter(), 'disagreement': Counter()}
    statuses = Counter()
    header = ['pair_id', 'record_a_title', 'record_b_title', 'two_of_three', 'disagreement', 'teacher_label', 'teacher_consistency',
              'student_label', 'match_probability', 'status', 'categories', 'summary_vi', 'uncertainties_vi', 'evidence_json']
    lines = ['# WDC evidence-check: teacher-generated diagnoses', '',
             'Selected-set analysis only. Not verified label errors; no labels changed. Categories overlap.', '',
             '| Pair ID | 2/3 | Disagreement | Status | Categories | Observations |',
             '|---|---|---|---|---|---|']
    def escape(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    with (output / 'diagnoses.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator='\n')
        writer.writeheader()
        for i, item in enumerate(items):
            pid = item['pair_id']
            diag = results.get(pid)
            marker = output / 'requests' / f'{i:04d}.json'
            status = diag.status if diag else ('invalid' if pid in failures else
                                               ('unresolved' if marker.exists() else 'pending'))
            statuses[status] += 1
            cats = sorted({c.category_id for c in diag.cases}) if diag else []
            categories.update(cats)
            for g in groups:
                if item[g]:
                    groups[g].update(cats)
            summary = diag.summary_vi if diag else failures.get(pid, '')
            fields = parse_fields(item['input_text'])
            writer.writerow({'pair_id': pid, 'record_a_title': fields['A']['title'], 'record_b_title': fields['B']['title'],
                             'two_of_three': item['two_of_three'], 'disagreement': item['disagreement'],
                             'teacher_label': item['oof']['given_label'], 'teacher_consistency': item['teacher']['consistency'],
                             'student_label': item['oof']['predicted_label'], 'match_probability': item['oof']['match_probability'],
                             'status': status, 'categories': ';'.join(cats), 'summary_vi': summary,
                             'uncertainties_vi': '; '.join(diag.uncertainties_vi) if diag else '',
                             'evidence_json': json.dumps([c.model_dump() for c in diag.cases], ensure_ascii=False) if diag else ''})
            lines.append('| ' + ' | '.join(map(escape, [pid, item['two_of_three'], item['disagreement'], status,
                                                        ', '.join(CATEGORY_NAMES[c] for c in cats), summary])) + ' |')
    unresolved = [item['pair_id'] for i, item in enumerate(items)
                  if (output / 'requests' / f'{i:04d}.json').exists()
                  and not (output / 'responses' / f'{i:04d}.json').exists()]
    not_attempted = [item['pair_id'] for i, item in enumerate(items)
                     if not (output / 'requests' / f'{i:04d}.json').exists()
                     and not (output / 'responses' / f'{i:04d}.json').exists()]
    unresolved_retries = [{'pair_id': item['pair_id'], 'request_path': str(request)}
                         for i, item in enumerate(items)
                         for request in sorted((output / 'retries').glob(f'*/requests/{i:04d}.json'))
                         if not (request.parent.parent / 'responses' / request.name).exists()]
    dispatch_errors = [json.loads(p.read_text()) for p in sorted((output / 'errors').glob('*.json'))]
    selection_counts = {'two_of_three': sum(i['two_of_three'] for i in items),
                        'disagreement': sum(i['disagreement'] for i in items),
                        'overlap': sum(i['two_of_three'] and i['disagreement'] for i in items)}
    summary = {'selected_count': len(items), 'selection_counts': selection_counts,
               'valid_count': len(results), 'statuses': dict(statuses),
               'category_counts': dict(categories), 'category_counts_by_source': {g: dict(v) for g, v in groups.items()},
               'failures': failures, 'received_response_cost_usd': total, 'cost_sources': cost_sources,
               'received_response_count': sum(cost_sources.values()),
               'cost_complete': not cost_sources.get('unknown', 0) and not unresolved and not unresolved_retries,
               'unresolved_request_ids': unresolved,
               'unresolved_retry_requests': unresolved_retries,
               'not_attempted_pair_ids': not_attempted, 'dispatch_errors': dispatch_errors,
               'gold_accessed': False, 'labels_changed': False, 'method': 'model_generated_evidence_diagnosis'}
    dump(output / 'summary.json', summary)
    overview = ['# WDC evidence-check: teacher-generated diagnoses', '',
                'Phân loại tình huống quan sát được, không xác nhận nhãn sai và không thay đổi nhãn.', '',
                f"Đã nhận {len(results)}/{len(items)} phản hồi hợp lệ. Chi phí phản hồi đã lưu: {total:.6f} USD.", '',
                f"Trạng thái: {dict(statuses)}. Hạch toán đầy đủ: {summary['cost_complete']}. "
                'Chi phí trên không phải tổng đã xác minh nếu còn request chưa rõ kết quả.', '',
                '| Nguồn chọn | Số cặp |', '|---|---:|',
                f"| Teacher 2/3 | {selection_counts['two_of_three']} |",
                f"| Student–teacher bất đồng | {selection_counts['disagreement']} |",
                f"| Có trong cả hai nhóm | {selection_counts['overlap']} |",
                f'| Hợp, không trùng | {len(items)} |', '',
                '| Nhóm tình huống | Số cặp trong hợp | Nhóm 2/3 | Nhóm bất đồng |', '|---|---:|---:|---:|']
    overview += [f"| {name} | {categories[cat]} | {groups['two_of_three'][cat]} | {groups['disagreement'][cat]} |"
                 for cat, name in CATEGORY_NAMES.items()]
    overview += ['', 'Một cặp có thể thuộc nhiều nhóm; hai nguồn chọn có giao nhau. '
                 'Tần suất chỉ áp dụng cho tập được chọn, không phải tỷ lệ lỗi của toàn dataset.', '',
                 '## Từng cặp', '']
    lines = overview + lines[4:]
    lines += ['', '## Counts', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```',
              '', '## Full evidence and records', '']
    for item in items:
        diag = results.get(item['pair_id'])
        lines += ['### ' + item['pair_id'], '', '```text', item['input_text'], '```', '']
        if diag:
            lines += ['```json', diag.model_dump_json(indent=2), '```', '']
        else:
            lines += ['Chưa có phân loại hợp lệ: ' + failures.get(item['pair_id'], 'đang chờ phản hồi'), '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def run(items, config, output: Path, prompt, client, workers=4, limit=None, skip_unresolved=False):
    # Caller holds a filesystem lock. Per-pair markers make uncertain dispatch visible.
    pending = []
    for i, item in enumerate(items):
        response = output / 'responses' / f'{i:04d}.json'
        marker = output / 'requests' / f'{i:04d}.json'
        if response.exists():
            continue
        if marker.exists():
            if skip_unresolved:
                continue
            raise RuntimeError(f'unresolved request for {item["pair_id"]}; do not blindly retry')
        pending.append((i, item))
    if limit is not None:
        pending = pending[:limit]
    def dispatch(entry):
        i, item = entry
        payload = payload_for(item, config, prompt)
        dump(output / 'requests' / f'{i:04d}.json', {'pair_id': item['pair_id'], 'payload': payload})
        raw = client.create(payload)
        dump(output / 'responses' / f'{i:04d}.json', raw)
        # Even malformed replies are retained and charged before schema validation.
        return item['pair_id']
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(dispatch, entry): entry for entry in pending}
        for done in as_completed(futures):
            if done.cancelled():
                continue
            try:
                print('received', done.result(), flush=True)
            except Exception as error:
                errors.append(type(error).__name__)
                i, item = futures[done]
                detail = str(error) if str(error).startswith('OpenRouter ') else type(error).__name__
                dump(output / 'errors' / f'{i:04d}.json', {'pair_id': item['pair_id'],
                     'error_type': type(error).__name__, 'detail': detail})
                for future in futures:
                    future.cancel()
            report(items, output, config)
    summary = report(items, output, config)
    if errors:
        raise RuntimeError(f'dispatch failed ({errors}); report saved, inspect unresolved requests')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl')
    parser.add_argument('--consistency', type=Path, default=ROOT / 'data/cache/wdc_products/majority_labels/consistency.csv')
    parser.add_argument('--oof', type=Path, default=ROOT / 'outputs/uncertainty_refinement/wdc-qwen-oof/oof_predictions.csv')
    parser.add_argument('--oof-summary', type=Path, default=ROOT / 'outputs/uncertainty_refinement/wdc-qwen-oof/oof_summary.json')
    parser.add_argument('--settings', type=Path, default=ROOT / 'labeller-screening/settings.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/uncertainty_refinement/wdc-evidence-check')
    parser.add_argument('--expected-count', type=int, default=2500)
    parser.add_argument('--confirm-paid-evidence-check', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--skip-unresolved', action='store_true',
                        help='Dispatch only never-attempted pairs; leave uncertain requests untouched.')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or (args.limit is not None and args.limit < 1):
        parser.error('workers 1–4 and positive limit required')
    if args.action == 'run' and not args.confirm_paid_evidence_check:
        parser.error('--confirm-paid-evidence-check is required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / '.lock'
    with lock.open('a') as handle:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        items, config = prepare(args)
        if args.action == 'run':
            client = create_openrouter_json_schema_client(api_url=config['api_url'], model=config['settings']['sol_high']['model'],
                                                         timeout=config['request_timeout_seconds'], api_key_resolver=resolve_openrouter_api_key)
            result = run(items, config, args.output_dir, PROMPT.read_text(), client, args.workers, args.limit,
                         args.skip_unresolved)
        else:
            result = report(items, args.output_dir, config)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
