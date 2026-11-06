"""Blinded evidence diagnosis for MaDI-Bench Companies (plan 261106); never relabels.

Same method as the WDC evidence check (``supervision/run_evidence_check.py``):
pairs are selected by teacher 2/3 agreement OR OOF teacher–student disagreement,
the teacher model describes observable situations in the seven categories with
verbatim quotes, and nothing here changes a label. Differences: the eight company
fields, the companies prompt (``evidence_case_prompt_companies.md``), and the
OpenAI flex tier at the WDC price cap (``configs/labelers/madi_companies_sol_high.json``).

    python -m supervision.run_evidence_check_companies prepare     # free: selection + contract
    python -m supervision.run_evidence_check_companies report      # free: re-render from saved responses
    python -m supervision.run_evidence_check_companies run --confirm-paid-evidence-check

``run`` refuses while the prompt is still marked DRAFT.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from supervision.build_confident_learning import _load_oof
from supervision.llm_providers import resolve_openrouter_api_key
from supervision.openrouter_json_schema_client import create_openrouter_json_schema_client
from supervision.run_evidence_check import CATEGORY_NAMES, Category, cost_of, dump, keyed, response_paths

ROOT = Path(__file__).resolve().parents[1]
PROMPT = Path(__file__).with_name('evidence_case_prompt_companies.md')
DATA = ROOT / 'data/cache/madi_companies/madi-bench-companies-base-062058d'
OOF = ROOT / 'outputs/uncertainty_refinement/madi-companies-oof'
OUTPUT = ROOT / 'outputs/uncertainty_refinement/madi-companies-evidence-check'
SETTINGS = ROOT / 'configs/labelers/madi_companies_sol_high.json'
EXPECTED_COUNT = 1968
FIELDS = ('name', 'founded', 'country', 'city', 'industry', 'assets', 'revenue', 'keypeople')
OUTPUT_TOKEN_CAP = 8192


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Evidence(StrictModel):
    side: Literal['A', 'B']
    field: Literal['name', 'founded', 'country', 'city', 'industry', 'assets', 'revenue', 'keypeople']
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


def parse_fields(text: str) -> dict:
    """The eight serialized company fields per side, exactly once each."""
    result = {'A': {}, 'B': {}}
    side = None
    for line in text.splitlines():
        if line in ('Record A:', 'Record B:'):
            side = line[7]
        elif line.startswith('- ') and ': ' in line and side:
            field, value = line[2:].split(': ', 1)
            if field not in FIELDS or field in result[side]:
                raise ValueError('unexpected or duplicate record field')
            result[side][field] = value
    if any(len(result[s]) != len(FIELDS) for s in result):
        raise ValueError(f'expected {len(FIELDS)} serialized fields per record')
    return result


def select_pairs(inputs: Path, consistency: Path, oof: Path, summary: Path, expected_count=EXPECTED_COUNT):
    """Teacher 2/3 OR OOF teacher–student disagreement (the WDC selection rule)."""
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
    selected, counts = [], Counter()
    for row in oof_rows:
        teacher = con[row['pair_id']]
        votes = [teacher[f'pass_0{i}'] for i in range(1, 4)]
        if any(v not in ('match', 'non_match') for v in votes):
            raise ValueError('invalid teacher pass label')
        label, n = Counter(votes).most_common(1)[0]
        if (teacher['majority_label'] != label or teacher['majority_votes'] != str(n)
                or teacher['consistency'] != f'{n}/3' or row['given_label'] != label):
            raise ValueError('teacher/OOF majority mismatch')
        inconsistent, disagree = n == 2, row['given_label'] != row['predicted_label']
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
    # Pair ID, earlier labels and student scores stay local; only the two records are sent.
    record_text = item['input_text'].split('Record A:', 1)[1]
    payload = {
        'model': config['settings']['sol_high']['model'],
        'messages': [{'role': 'system', 'content': prompt},
                     {'role': 'user', 'content': 'Record A:' + record_text}],
        'reasoning': config['settings']['sol_high']['reasoning'],
        'provider': config['provider_routing'],
        'max_tokens': OUTPUT_TOKEN_CAP,
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'evidence_case_diagnosis', 'strict': True, 'schema': Diagnosis.model_json_schema()}},
    }
    if 'service_tier' in config:
        payload['service_tier'] = config['service_tier']
    return payload


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
        for e in case.evidence:
            value = fields[e.side][e.field]
            if value == '<missing>' or e.quote not in value or not e.quote.strip():
                raise ValueError(f'ungrounded evidence quote: {e.side}.{e.field}: {e.quote!r}')
        if case.category_id == 'cross_record_conflict' and {e.side for e in case.evidence} != {'A', 'B'}:
            raise ValueError('cross-record conflict requires both sides')
        if case.category_id == 'intra_record_contradiction':
            if not any(len({(e.field, e.quote) for e in case.evidence if e.side == s}) >= 2 for s in ('A', 'B')):
                raise ValueError('internal contradiction requires two quotes on one side')
    return diag


def prompt_signed_off(prompt: str) -> bool:
    return not prompt.lstrip().startswith('Status: DRAFT')


def prepare(inputs: Path, consistency: Path, oof: Path, oof_summary: Path, settings: Path, output: Path,
            expected_count: int = EXPECTED_COUNT):
    config = json.loads(settings.read_text(encoding='utf-8'))
    routing = config['provider_routing']
    if (config['api_url'] != 'https://openrouter.ai/api/v1' or routing['only'] != ['openai']
            or routing['allow_fallbacks'] is not False or routing.get('max_price') != {'prompt': 2.0, 'completion': 10.0}):
        raise ValueError('expected canonical OpenRouter, pinned OpenAI routing and the 2/10 price cap')
    items, counts = select_pairs(inputs, consistency, oof, oof_summary, expected_count)
    paths = [inputs, consistency, oof, oof_summary, settings, PROMPT]
    contract = {'schema_version': 1, 'dataset_id': 'madi_companies',
                'selection_rule': 'teacher_2of3_OR_stored_prediction_disagreement',
                'sources': {str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p):
                            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'counts': counts, 'config': config, 'output_token_cap': OUTPUT_TOKEN_CAP,
                'response_schema': Diagnosis.model_json_schema(), 'gold_accessed': False}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('contract.json', contract), ('selected_pairs.json', items)]:
        target = output / name
        if target.exists() and json.loads(target.read_text(encoding='utf-8')) != value:
            raise ValueError('existing run contract or inputs changed; use a new output directory')
        if not target.exists():
            dump(target, value)
    return items, config


def collect(items, output: Path, config):
    results, failures, total, cost_sources = {}, {}, 0.0, Counter()
    for i, item in enumerate(items):
        for path in response_paths(output, i):
            raw = json.loads(path.read_text(encoding='utf-8'))
            try:
                cost, source = cost_of(raw, config)
            except (ValueError, TypeError, OverflowError, AttributeError, KeyError):
                cost, source = None, 'unknown'
            cost_sources[source] += 1
            total += cost or 0.0
            try:
                results[item['pair_id']] = validate_diagnosis(raw, item)
                failures.pop(item['pair_id'], None)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                if item['pair_id'] not in results:
                    failures[item['pair_id']] = str(error)
    return results, failures, total, dict(cost_sources)


DIAGNOSIS_HEADER = ['pair_id', 'record_a_name', 'record_b_name', 'two_of_three', 'disagreement', 'teacher_label',
                    'teacher_consistency', 'student_label', 'match_probability', 'status', 'categories',
                    'summary_vi', 'uncertainties_vi', 'evidence_json']


def report(items, output: Path, config) -> dict:
    """diagnoses.csv (read by the category-rule builder) and summary.json. Local and git-ignored."""
    results, failures, total, cost_sources = collect(items, output, config)
    statuses, categories = Counter(), Counter()
    with (output / 'diagnoses.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSIS_HEADER, lineterminator='\n')
        writer.writeheader()
        for i, item in enumerate(items):
            pid, diag = item['pair_id'], results.get(item['pair_id'])
            status = diag.status if diag else ('invalid' if pid in failures else
                                               ('unresolved' if (output / 'requests' / f'{i:04d}.json').exists() else 'pending'))
            statuses[status] += 1
            cats = sorted({c.category_id for c in diag.cases}) if diag else []
            categories.update(cats)
            fields = parse_fields(item['input_text'])
            writer.writerow({'pair_id': pid, 'record_a_name': fields['A']['name'], 'record_b_name': fields['B']['name'],
                             'two_of_three': item['two_of_three'], 'disagreement': item['disagreement'],
                             'teacher_label': item['oof']['given_label'], 'teacher_consistency': item['teacher']['consistency'],
                             'student_label': item['oof']['predicted_label'], 'match_probability': item['oof']['match_probability'],
                             'status': status, 'categories': ';'.join(cats),
                             'summary_vi': diag.summary_vi if diag else failures.get(pid, ''),
                             'uncertainties_vi': '; '.join(diag.uncertainties_vi) if diag else '',
                             'evidence_json': json.dumps([c.model_dump() for c in diag.cases], ensure_ascii=False) if diag else ''})
    unresolved = [it['pair_id'] for i, it in enumerate(items)
                  if (output / 'requests' / f'{i:04d}.json').exists() and not (output / 'responses' / f'{i:04d}.json').exists()]
    summary = {'selected_count': len(items),
               'selection_counts': {'two_of_three': sum(i['two_of_three'] for i in items),
                                    'disagreement': sum(i['disagreement'] for i in items),
                                    'overlap': sum(i['two_of_three'] and i['disagreement'] for i in items)},
               'valid_count': len(results), 'statuses': dict(statuses), 'category_counts': dict(categories),
               'category_names': CATEGORY_NAMES, 'failures': failures, 'received_response_cost_usd': total,
               'cost_sources': cost_sources, 'unresolved_request_ids': unresolved,
               'cost_complete': not cost_sources.get('unknown', 0) and not unresolved,
               'gold_accessed': False, 'labels_changed': False, 'method': 'model_generated_evidence_diagnosis'}
    dump(output / 'summary.json', summary)
    return summary


def run(items, config, output: Path, prompt: str, client, workers=4, limit=None) -> dict:
    if not prompt_signed_off(prompt):
        raise RuntimeError('the companies evidence prompt is still marked DRAFT; sign it off first')
    pending = []
    for i, item in enumerate(items):
        if (output / 'responses' / f'{i:04d}.json').exists():
            continue
        if (output / 'requests' / f'{i:04d}.json').exists():
            raise RuntimeError(f'unresolved request for {item["pair_id"]}; inspect before retrying')
        pending.append((i, item))
    pending = pending[:limit] if limit is not None else pending

    def dispatch(entry):
        i, item = entry
        payload = payload_for(item, config, prompt)
        dump(output / 'requests' / f'{i:04d}.json', {'pair_id': item['pair_id'], 'payload': payload})
        dump(output / 'responses' / f'{i:04d}.json', client.create(payload))   # kept even if malformed
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
                dump(output / 'errors' / f'{i:04d}.json', {'pair_id': item['pair_id'], 'error_type': type(error).__name__,
                                                           'detail': str(error)[:500]})
                for future in futures:
                    future.cancel()
    summary = report(items, output, config)
    if errors:
        raise RuntimeError(f'dispatch failed ({errors}); report saved, inspect unresolved requests')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--inputs', type=Path, default=DATA / 'teacher_labels/inputs.jsonl')
    parser.add_argument('--consistency', type=Path, default=DATA / 'majority_labels/consistency.csv')
    parser.add_argument('--oof', type=Path, default=OOF / 'oof_predictions.csv')
    parser.add_argument('--oof-summary', type=Path, default=OOF / 'oof_summary.json')
    parser.add_argument('--settings', type=Path, default=SETTINGS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    parser.add_argument('--confirm-paid-evidence-check', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or (args.limit is not None and args.limit < 1):
        parser.error('workers 1–4 and a positive limit required')
    if args.action == 'run' and not args.confirm_paid_evidence_check:
        parser.error('--confirm-paid-evidence-check is required')
    items, config = prepare(args.inputs, args.consistency, args.oof, args.oof_summary, args.settings, args.output_dir)
    if args.action == 'prepare':
        print(json.dumps({'selected': len(items), 'counts': json.loads((args.output_dir / 'contract.json').read_text())['counts'],
                          'prompt_signed_off': prompt_signed_off(PROMPT.read_text(encoding='utf-8'))}, indent=2))
        return
    if args.action == 'run':
        client = create_openrouter_json_schema_client(api_url=config['api_url'], model=config['settings']['sol_high']['model'],
                                                     timeout=config['request_timeout_seconds'], api_key_resolver=resolve_openrouter_api_key)
        result = run(items, config, args.output_dir, PROMPT.read_text(encoding='utf-8'), client, args.workers, args.limit)
    else:
        result = report(items, args.output_dir, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
