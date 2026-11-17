"""Category-rule treatment for Geo-ER Pittsburgh (plan 261111), arms A, B and C.

The WDC builder (``supervision/build_category_treated_dataset.py``) with the
Geo-ER settings: the five record fields, and latitude and longitude as supporting
information (two sources never give identical coordinates for the same place, and
the distance is given separately as ``distance_m``; like price on WDC they never
decide non-match on their own). Same guideline steps and action rule. Diagnoses
come from the evidence check v2 (D25); arm B needs none (it removes every
detected pair), so B can be built as soon as the OOF predictions and
``selected_pairs.json`` exist (amendment A1). No API call, no gold labels.

    python -m supervision.build_category_treated_dataset_geo_er build --arm B

    python -m supervision.build_category_treated_dataset_geo_er build --arm C
    python -m supervision.build_category_treated_dataset_geo_er verify --arm C
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from supervision import build_category_treated_dataset as core
from supervision.run_evidence_check_v2 import PROFILES, parse_fields

ROOT = Path(__file__).resolve().parents[1]
PROFILE = PROFILES['geo_er_pit']
DATA = ROOT / 'data/cache/geo_er/geo-er-osm-fsq-pit-83063f7'
SUPPORTING_FIELDS = frozenset({'latitude', 'longitude'})
CONTRACT = 'plans/261111-1415-geo-er-treatment-arms/contract.md'


def sources(inputs=None, majority=None, selected=None, diagnoses=None, arm=None) -> dict:
    if diagnoses is None and arm != 'B':
        diagnoses = PROFILE.output / 'diagnoses.csv'
    return dict(inputs_path=inputs or PROFILE.inputs,
                majority_path=majority or DATA / 'majority_labels/llm_unrefined.csv',
                selected_path=selected or PROFILE.output / 'selected_pairs.json',
                diagnoses_path=diagnoses,
                dataset_id=PROFILE.dataset_id, expected_count=PROFILE.expected_count,
                field_parser=partial(parse_fields, fields=PROFILE.fields),
                supporting_fields=SUPPORTING_FIELDS, contract=CONTRACT)


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
    src = sources(selected=args.selected_pairs, diagnoses=args.diagnoses, arm=args.arm)
    out = args.output_dir or output_dir(args.arm)
    result = (core.build if args.action == 'build' else core.verify)(args.arm, out, **src)
    print(json.dumps({k: result[k] for k in ('arm', 'train_rows', 'duplicate_rows', 'actions', 'class_counts')}, indent=2))


if __name__ == '__main__':
    main()
