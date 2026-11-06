"""Evidence check v2 (D25): one domain-neutral prompt for every dataset; never relabels.

Same method as the WDC pilot (``supervision/run_evidence_check.py``): pairs are
selected by teacher 2/3 agreement OR OOF teacher–student disagreement, and the
teacher model describes observable situations in the seven categories with
verbatim quotes. What changed (researcher, 2026-11-06):

- the prompt is ``evidence_case_prompt_generic.md`` (signed off; SHA-256 in D25),
  with ``{FIELDS}`` filled from the dataset's record fields;
- the explanations are not tied to a language (``note``, ``summary``,
  ``uncertainties``);
- every dataset uses the shared teacher settings
  ``configs/labelers/evidence_check_sol_high_flex.json`` (flex tier, 2/10 cap).

    python -m supervision.run_evidence_check_v2 prepare --dataset madi_companies   # free
    python -m supervision.run_evidence_check_v2 report  --dataset madi_companies   # free
    python -m supervision.run_evidence_check_v2 run     --dataset wdc --confirm-paid-evidence-check
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from supervision.build_confident_learning import _load_oof
from supervision.llm_providers import resolve_openrouter_api_key
from supervision.openrouter_json_schema_client import create_openrouter_json_schema_client
from supervision.run_evidence_check import CATEGORY_NAMES, Category, cost_of, dump, keyed, response_paths

ROOT = Path(__file__).resolve().parents[1]
PROMPT = Path(__file__).with_name('evidence_case_prompt_generic.md')
PROMPT_SHA256 = '48b2c4615336f09dc6240a04a8d71182b32d81c8306dd10048f8adb8dce8ec2c'   # signed off 2026-11-06 (D25)
SETTINGS = ROOT / 'configs/labelers/evidence_check_sol_high_flex.json'
OUTPUT_TOKEN_CAP = 8192
COST_PER_PAIR_ESTIMATE_USD = 0.021   # WDC pilot: USD 3.25 for 158 pairs


@dataclass(frozen=True)
class Profile:
    dataset_id: str
    fields: tuple[str, ...]
    inputs: Path
    consistency: Path
    oof: Path
    oof_summary: Path
    output: Path
    expected_count: int


_MADI = ROOT / 'data/cache/madi_companies/madi-bench-companies-base-062058d'
_MADI_OOF = ROOT / 'outputs/uncertainty_refinement/madi-companies-oof'
_WDC_OOF = ROOT / 'outputs/uncertainty_refinement/wdc-qwen-oof'
PROFILES = {
    'wdc': Profile(
        'wdc_products_80cc_small_100un', ('title', 'brand', 'description', 'price', 'priceCurrency'),
        ROOT / 'data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl',
        ROOT / 'data/cache/wdc_products/majority_labels/consistency.csv',
        _WDC_OOF / 'oof_predictions.csv', _WDC_OOF / 'oof_summary.json',
        ROOT / 'outputs/uncertainty_refinement/wdc-evidence-check-v2', 2500),
    'madi_companies': Profile(
        'madi_companies', ('name', 'founded', 'country', 'city', 'industry', 'assets', 'revenue', 'keypeople'),
        _MADI / 'teacher_labels/inputs.jsonl', _MADI / 'majority_labels/consistency.csv',
        _MADI_OOF / 'oof_predictions.csv', _MADI_OOF / 'oof_summary.json',
        ROOT / 'outputs/uncertainty_refinement/madi-companies-evidence-check', 1968),
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


@lru_cache(maxsize=None)
def models(fields: tuple[str, ...]):
    """Response schema for one dataset: the quotable field list is the dataset's record fields."""
    evidence = create_model('Evidence', __base__=StrictModel, side=(Literal['A', 'B'], ...),
                            field=(Literal[fields], ...), quote=(str, Field(min_length=1)))
    case = create_model('Case', __base__=StrictModel, category_id=(Category, ...),
                        attribute=(str, Field(min_length=1)), note=(str, Field(min_length=1)),
                        evidence=(list[evidence], Field(min_length=1)))
    return create_model('Diagnosis', __base__=StrictModel,
                        status=(Literal['categorized', 'no_finding', 'unknown'], ...), cases=(list[case], ...),
                        summary=(str, Field(min_length=1)), uncertainties=(list[str], ...))


def prompt_for(profile: Profile) -> str:
    template = PROMPT.read_text(encoding='utf-8')
    if hashlib.sha256(template.encode('utf-8')).hexdigest() != PROMPT_SHA256:
        raise ValueError('the evidence prompt differs from the signed-off version (D25)')
    if template.count('{FIELDS}') != 1:
        raise ValueError('the prompt must contain exactly one {FIELDS} placeholder')
    return template.replace('{FIELDS}', '/'.join(profile.fields))


def parse_fields(text: str, fields: tuple[str, ...]) -> dict:
    """The serialized record fields per side, exactly once each."""
    result = {'A': {}, 'B': {}}
    side = None
    for line in text.splitlines():
        if line in ('Record A:', 'Record B:'):
            side = line[7]
        elif line == 'Pair:':
            side = None
        elif line.startswith('- ') and ': ' in line and side:
            field, value = line[2:].split(': ', 1)
            if field not in fields or field in result[side]:
                raise ValueError('unexpected or duplicate record field')
            result[side][field] = value
    if any(len(result[s]) != len(fields) for s in result):
        raise ValueError(f'expected {len(fields)} serialized fields per record')
    return result


def select_pairs(profile: Profile):
    """Teacher 2/3 OR OOF teacher–student disagreement (the WDC selection rule)."""
    with profile.inputs.open(encoding='utf-8') as handle:
        input_rows = list(map(json.loads, handle))
    if any(set(r) != {'pair_id', 'input_text'} for r in input_rows):
        raise ValueError('inputs must contain only pair_id and input_text')
    inp = keyed(input_rows)
    with profile.consistency.open(encoding='utf-8', newline='') as handle:
        con = keyed(csv.DictReader(handle))
    oof_rows, _, _, _, _ = _load_oof(profile.oof, profile.oof_summary, profile.expected_count)
    if len(inp) != profile.expected_count or set(inp) != set(con) or set(inp) != {r['pair_id'] for r in oof_rows}:
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
            parse_fields(item['input_text'], profile.fields)
            selected.append(item)
    counts['union'] = len(selected)
    return selected, dict(counts)


def payload_for(item: dict, config: dict, prompt: str, profile: Profile) -> dict:
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
            'name': 'evidence_case_diagnosis', 'strict': True,
            'schema': models(profile.fields).model_json_schema()}},
    }
    if 'service_tier' in config:
        payload['service_tier'] = config['service_tier']
    return payload


def validate_diagnosis(raw: dict, item: dict, profile: Profile):
    if raw.get('error') or not raw.get('choices'):
        raise ValueError('missing successful completion')
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop' or choice.get('message', {}).get('refusal'):
        raise ValueError('refused or incomplete completion')
    diag = models(profile.fields).model_validate_json(choice['message']['content'])
    if (diag.status == 'categorized') != bool(diag.cases):
        raise ValueError('inconsistent status and cases')
    fields = parse_fields(item['input_text'], profile.fields)
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


def prepare(profile: Profile, settings: Path = SETTINGS, output: Path | None = None):
    output = output or profile.output
    config = json.loads(settings.read_text(encoding='utf-8'))
    routing = config['provider_routing']
    if (config['api_url'] != 'https://openrouter.ai/api/v1' or routing['only'] != ['openai']
            or routing['allow_fallbacks'] is not False
            or routing.get('max_price') != {'prompt': 2.0, 'completion': 10.0}):
        raise ValueError('expected canonical OpenRouter, pinned OpenAI routing and the 2/10 price cap')
    prompt = prompt_for(profile)
    items, counts = select_pairs(profile)
    paths = [profile.inputs, profile.consistency, profile.oof, profile.oof_summary, settings, PROMPT]
    contract = {'schema_version': 2, 'dataset_id': profile.dataset_id, 'decision': 'D25',
                'selection_rule': 'teacher_2of3_OR_stored_prediction_disagreement',
                'sources': {str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p):
                            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'prompt_template_sha256': PROMPT_SHA256,
                'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                'fields': list(profile.fields), 'counts': counts, 'config': config,
                'output_token_cap': OUTPUT_TOKEN_CAP,
                'response_schema': models(profile.fields).model_json_schema(), 'gold_accessed': False}
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('contract.json', contract), ('selected_pairs.json', items)]:
        target = output / name
        if target.exists() and json.loads(target.read_text(encoding='utf-8')) != value:
            raise ValueError('existing run contract or inputs changed; use a new output directory')
        if not target.exists():
            dump(target, value)
    return items, config


def collect(items, output: Path, config, profile: Profile):
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
                results[item['pair_id']] = validate_diagnosis(raw, item, profile)
                failures.pop(item['pair_id'], None)
            except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
                if item['pair_id'] not in results:
                    failures[item['pair_id']] = str(error)
    return results, failures, total, dict(cost_sources)


DIAGNOSIS_HEADER = ['pair_id', 'two_of_three', 'disagreement', 'teacher_label', 'teacher_consistency',
                    'student_label', 'match_probability', 'status', 'categories', 'summary', 'uncertainties',
                    'evidence_json']


def report(items, output: Path, config, profile: Profile) -> dict:
    """diagnoses.csv (read by the category-rule builder) and summary.json; local and git-ignored."""
    results, failures, total, cost_sources = collect(items, output, config, profile)
    statuses, categories = Counter(), Counter()
    with (output / 'diagnoses.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAGNOSIS_HEADER, lineterminator='\n')
        writer.writeheader()
        for i, item in enumerate(items):
            pid, diag = item['pair_id'], results.get(item['pair_id'])
            status = diag.status if diag else ('invalid' if pid in failures else
                                               ('unresolved' if (output / 'requests' / f'{i:04d}.json').exists()
                                                else 'pending'))
            statuses[status] += 1
            cats = sorted({c.category_id for c in diag.cases}) if diag else []
            categories.update(cats)
            writer.writerow({
                'pair_id': pid, 'two_of_three': item['two_of_three'], 'disagreement': item['disagreement'],
                'teacher_label': item['oof']['given_label'], 'teacher_consistency': item['teacher']['consistency'],
                'student_label': item['oof']['predicted_label'], 'match_probability': item['oof']['match_probability'],
                'status': status, 'categories': ';'.join(cats),
                'summary': diag.summary if diag else failures.get(pid, ''),
                'uncertainties': '; '.join(diag.uncertainties) if diag else '',
                'evidence_json': json.dumps([c.model_dump() for c in diag.cases], ensure_ascii=False) if diag else ''})
    unresolved = [it['pair_id'] for i, it in enumerate(items)
                  if (output / 'requests' / f'{i:04d}.json').exists()
                  and not (output / 'responses' / f'{i:04d}.json').exists()]
    summary = {'dataset_id': profile.dataset_id, 'selected_count': len(items),
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


def run(items, config, output: Path, client, profile: Profile, workers=4, limit=None) -> dict:
    prompt = prompt_for(profile)
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
        payload = payload_for(item, config, prompt, profile)
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
                dump(output / 'errors' / f'{i:04d}.json', {'pair_id': item['pair_id'],
                                                           'error_type': type(error).__name__,
                                                           'detail': str(error)[:500]})
                for future in futures:
                    future.cancel()
    summary = report(items, output, config, profile)
    if errors:
        raise RuntimeError(f'dispatch failed ({errors}); report saved, inspect unresolved requests')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--dataset', choices=sorted(PROFILES), required=True)
    parser.add_argument('--settings', type=Path, default=SETTINGS)
    parser.add_argument('--confirm-paid-evidence-check', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or (args.limit is not None and args.limit < 1):
        parser.error('workers 1–4 and a positive limit required')
    if args.action == 'run' and not args.confirm_paid_evidence_check:
        parser.error('--confirm-paid-evidence-check is required')
    profile = PROFILES[args.dataset]
    items, config = prepare(profile, args.settings)
    if args.action == 'prepare':
        counts = json.loads((profile.output / 'contract.json').read_text(encoding='utf-8'))['counts']
        print(json.dumps({'dataset': args.dataset, 'selected': len(items), 'counts': counts,
                          'estimated_cost_usd': round(COST_PER_PAIR_ESTIMATE_USD * len(items), 2)}, indent=2))
        return
    if args.action == 'run':
        client = create_openrouter_json_schema_client(
            api_url=config['api_url'], model=config['settings']['sol_high']['model'],
            timeout=config['request_timeout_seconds'], api_key_resolver=resolve_openrouter_api_key)
        result = run(items, config, profile.output, client, profile, args.workers, args.limit)
    else:
        result = report(items, profile.output, config, profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
