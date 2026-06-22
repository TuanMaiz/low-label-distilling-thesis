# Rationale Pipeline

Phase 02 implements structured, attribute-grounded rationale supervision for
low-label entity resolution.

## Fixed Schema

`schema.py` defines the cached teacher-rationale contract:

- `decision`: `match` or `non-match`, required to match `gold_label`.
- `evidence`: field-grounded support items.
- `conflicts`: field-grounded disagreement or uncertainty items.
- `missing_fields`: explicit `A`/`B` field references where values are absent.
- `decision_rule`: one compact rule summarizing the decision.
- `prompt_version`, `teacher_model`, and `schema_version`: reproducibility metadata.

Allowed relation labels are fixed to:

- `exact agreement`
- `abbreviation`
- `synonym`
- `format variation`
- `numeric mismatch`
- `semantic mismatch`
- `missing`

The validator rejects invalid relation labels, invented fields, mismatched pair
IDs, decisions that disagree with the gold label, and evidence values that do
not exactly match the serialized input records.

## Commands

Generate cached rationales:

```bash
export OPENROUTER_API_KEY=...

python -m rationales.generate_teacher_rationales \
  --teacher-model openai/gpt-4o-mini \
  --input data/cache/wdc_products/low_label/train_128.jsonl \
  --output data/cache/wdc_products/rationales/train_128.openrouter.rationales.jsonl \
  --rejects data/cache/wdc_products/rationales/train_128.openrouter.rejects.jsonl
```

Validate cached rationales:

```bash
python -m rationales.validate_rationales \
  --rationales data/cache/wdc_products/rationales/train_128.openrouter.rationales.jsonl \
  --pairs data/cache/wdc_products/low_label/train_128.jsonl
```

Build student targets:

```bash
python -m rationales.build_targets \
  --pairs data/cache/wdc_products/low_label/train_128.jsonl \
  --rationales data/cache/wdc_products/rationales/train_128.openrouter.rationales.jsonl \
  --output data/cache/wdc_products/targets/train_128.structured_rationale.jsonl \
  --variant structured_rationale
```

## Teacher Providers

The generator uses `model_providers.py` to keep the model API separate from
JSONL iteration, cache reuse, and validation. The only built-in provider is
OpenRouter.

```bash
export OPENROUTER_API_KEY=...

python -m rationales.generate_teacher_rationales \
  --teacher-model openai/gpt-4o-mini \
  --input data/cache/wdc_products/low_label/train_128.jsonl \
  --output data/cache/wdc_products/rationales/train_128.openrouter.rationales.jsonl \
  --rejects data/cache/wdc_products/rationales/train_128.openrouter.rejects.jsonl
```

The OpenRouter adapter calls `/chat/completions`, requests JSON-schema
structured output, and stores cache rows with `teacher_model` formatted as
`openrouter:{model-slug}`. This keeps caches from different providers or models
separate.

Do not depend on the legacy Wikidata or mBART entrypoints from this package.
