"""Category-rule treatment for MaDI-Bench Companies (plan 261106), arms A, B and C.

The WDC builder (``supervision/build_category_treated_dataset.py``) with the
Companies settings: the eight company fields, and assets, revenue and key people
as supporting information (like price on WDC: they never decide non-match on their
own). Same guideline steps and action rule. No API call, no gold labels.

    python -m supervision.build_category_treated_dataset_companies build --arm C
    python -m supervision.build_category_treated_dataset_companies verify --arm C
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from supervision import build_category_treated_dataset as core
from supervision.run_evidence_check_companies import DATA, EXPECTED_COUNT, OUTPUT, parse_fields

ROOT = Path(__file__).resolve().parents[1]
SUPPORTING_FIELDS = frozenset({'assets', 'revenue', 'keypeople'})
CONTRACT = 'plans/261106-1330-companies-treatment-arms/contract.md'
DATASET_ID = 'madi_companies'


def sources(inputs=None, majority=None, selected=None, diagnoses=None) -> dict:
    return dict(inputs_path=inputs or DATA / 'teacher_labels/inputs.jsonl',
                majority_path=majority or DATA / 'majority_labels/llm_unrefined.csv',
                selected_path=selected or OUTPUT / 'selected_pairs.json',
                diagnoses_path=diagnoses or OUTPUT / 'diagnoses.csv',
                dataset_id=DATASET_ID, expected_count=EXPECTED_COUNT,
                field_parser=parse_fields, supporting_fields=SUPPORTING_FIELDS, contract=CONTRACT)


def output_dir(arm: str) -> Path:
    return DATA / f'refinement/category-rule-{arm}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['build', 'verify'])
    parser.add_argument('--arm', choices=core.ARMS, required=True)
    parser.add_argument('--selected-pairs', type=Path)
    parser.add_argument('--diagnoses', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    src = sources(selected=args.selected_pairs, diagnoses=args.diagnoses)
    out = args.output_dir or output_dir(args.arm)
    result = (core.build if args.action == 'build' else core.verify)(args.arm, out, **src)
    print(json.dumps({k: result[k] for k in ('arm', 'train_rows', 'duplicate_rows', 'actions', 'class_counts')}, indent=2))


if __name__ == '__main__':
    main()
