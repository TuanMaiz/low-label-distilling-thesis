"""Final-analysis check of the category rule against train gold (plan 261023, Phase 4).

Train gold is read here only, after all arms are frozen (contract section 7).
Reports, per action and guideline step, how often the teacher label and the
final label agree with gold, and how often teacher labels are wrong inside
versus outside the detected uncertain set.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / 'data/cache/wdc_products/refinement/category-rule-C/decisions.csv'
GOLD = ROOT / 'data/cache/wdc_products/full_label_targets/gold.jsonl'


def load_gold(path: Path) -> dict[str, str]:
    gold = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        gold[r['pair_id']] = r['target_text'].replace('-', '_')
    return gold


def audit(decisions: list[dict], gold: dict[str, str]) -> dict:
    if {d['pair_id'] for d in decisions} != set(gold):
        raise ValueError('decisions and gold cover different pairs')
    groups = defaultdict(lambda: {'pairs': 0, 'teacher_correct': 0, 'final_correct': 0, 'gold_match': 0})
    detection = {True: {'pairs': 0, 'teacher_wrong': 0}, False: {'pairs': 0, 'teacher_wrong': 0}}
    for d in decisions:
        g = gold[d['pair_id']]
        detected = d['detected'] == 'True'
        detection[detected]['pairs'] += 1
        detection[detected]['teacher_wrong'] += d['original_label'] != g
        if not detected:
            continue
        key = f"step{d['step'] or '-'}:{d['action']}"
        s = groups[key]
        s['pairs'] += 1
        s['teacher_correct'] += d['original_label'] == g
        s['final_correct'] += d['action'] != 'remove' and d['final_label'] == g
        s['gold_match'] += g == 'match'
    return {'by_step_and_action': dict(sorted(groups.items())),
            'teacher_error_rate': {('detected' if k else 'not_detected'):
                                   {**v, 'rate': v['teacher_wrong'] / v['pairs']} for k, v in detection.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', type=Path, default=DECISIONS)
    parser.add_argument('--gold', type=Path, default=GOLD)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.decisions.open(encoding='utf-8', newline='') as h:
        decisions = list(csv.DictReader(h))
    result = audit(decisions, load_gold(args.gold))
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
