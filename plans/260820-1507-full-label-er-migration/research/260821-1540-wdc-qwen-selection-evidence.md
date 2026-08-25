# WDC Products and Qwen3 Reranker Selection Evidence

Research timestamp: 2026-08-21 15:40 Asia/Bangkok.

## Summary

Dataset 1 is WDC Products in the pair-wise 80%-corner-case setting, using the
official small train/validation files and 100%-unseen test file. Model 1 is
`Qwen/Qwen3-Reranker-0.6B`, using the model repository previously screened in
this repository.

The selections are technically supported. Contract freeze remains blocked on
clarifying WDC dataset redistribution terms and on freezing full-label training
settings after the other datasets and models are chosen.

## Method

- Inspected the repository loader, tests, journals, raw archive, cached stats,
  Qwen config, runtime provenance, and prior screening artifacts.
- Verified dataset facts against the official WDC Products page.
- Verified model facts against the official Qwen Hugging Face repository and
  the prior screening artifacts.
- Used SHA-256 to identify the local official archive and selected compressed
  members.

## Dataset Finding

Selected identity:

```text
WDC Products
pair-wise formulation
80% corner-cases / 20% random products
small development set
100% unseen-products test
```

This is the intended hard combination because WDC defines difficulty along
corner-case ratio, unseen test entities, and development-set size. The selected
combination uses the highest corner-case ratio, highest unseen ratio, and
smallest official development set.

Selected files and expected counts:

| Split | File | Pairs | Match | Non-match |
|---|---|---:|---:|---:|
| Train | `wdcproducts80cc20rnd000un_train_small.json.gz` | 2,500 | 500 | 2,000 |
| Validation | `wdcproducts80cc20rnd000un_valid_small.json.gz` | 2,500 | 500 | 2,000 |
| Test | `wdcproducts80cc20rnd100un_gs.json.gz` | 4,500 | 500 | 4,000 |

SHA-256 evidence:

| Artifact | SHA-256 |
|---|---|
| `80pair.zip` | `b2044939cee5ea6f12148a2f3551508de3cb77660dfc91767c44daaf9d8a9c4a` |
| Train compressed member | `1915a92de76ddbf63a6c4d7ff3162df98f505033a8ad12ee7072b78b083d77c6` |
| Validation compressed member | `892e8d39cc8230dce5039a4bded16be1237d5c5c37b94a37404226d349ef3df8` |
| Test compressed member | `258f9b408e715410f07480bf4ad39788a10d5cee1512e322091de43dafae1297` |

The official page documents `title`, `description`, `brand`, `price`, and
`priceCurrency`, plus offer IDs, cluster IDs, pair ID, label, and pair-wise
hard-negative metadata. It also states that offers are split without record
overlap and reports approximately 4% estimated label noise in a manually
checked sample.

License limitation: the WDC benchmark-construction repository is BSD-3-Clause,
but the official benchmark download page does not separately state that the raw
dataset uses the same license. Academic use is clearly intended; redistribution
terms should be confirmed before freezing or sharing raw data.

## Model Finding

Selected identity:

```text
Repository: Qwen/Qwen3-Reranker-0.6B
Backend: generative_reranker
License: Apache-2.0
Scale: 0.6B parameters
Advertised context: 32K
```

The official model uses a single instruction/query/document prompt and scores
the final `yes` versus `no` answer. This jointly encodes both records and meets
the study's compact cross-encoder eligibility rule. The repository adapter maps
Record A to Query and Record B to Document.

The prior repository run verified an adapter using a 4,096-token experiment
cap without truncation, `no=0`,
`yes=1`, and LoRA with rank 8, alpha 16, dropout 0.05, attention projection
targets, and gradient checkpointing. The screening demonstrated viable
training and validation behavior; it does not freeze the new full-label
optimizer, epoch, precision, or resource contract.

Contamination risk is unknown. WDC data predates the model, but Qwen's public
training disclosure does not enumerate enough source data to rule out WDC or
derived benchmark content.

## Sources

- [WDC Products official benchmark page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/)
- [Official WDC 80% pair-wise archive](https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip)
- [WDC Products construction repository](https://github.com/wbsg-uni-mannheim/wdcproducts)
- [Qwen3-Reranker-0.6B official model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- Local `data/cache/wdc_products/stats.json`
- Local prior `student_config.json` and `runtime_provenance.json` under the
  Qwen Phase-05 result package

## Unresolved Questions

- What exact license or redistribution terms apply to the WDC Products data
  files, separately from the BSD-3-Clause construction code?
- Should the 4,096-token no-truncation cap remain after input-length audits over
  the two additional datasets?
- Which full-label optimizer, schedule, precision, GPU ceiling, and validation
  threshold metric will be shared across the gold and LLM-label arms?

## Next Steps

1. Obtain or document authoritative WDC dataset usage terms.
2. Select Datasets 2 and 3.
3. Select Models 2 and 3.
4. Run all-dataset input-length and resource audits before freezing training.
